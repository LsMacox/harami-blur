"""Hair-aware blur pipeline (modular, with graceful fallbacks).

Primary path:    SAM 3 (text="hair")  →  MatAnyone   →  alpha-blended Gaussian blur
Fallback chain:  Sapiens-seg-2B  →  SegFormer face-parsing
Matter fallback: MatAnyone  →  feathered binary mask

The pipeline always returns an output, mask and alpha — even if every "heavy"
dependency is missing — so the UI never throws at the user.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageFilter

log = logging.getLogger("harami-blur")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)


# ───────────────────────── device helper ─────────────────────────

def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ───────────────────────── protocols ─────────────────────────

class Segmenter(Protocol):
    name: str
    def segment(self, image: Image.Image) -> np.ndarray: ...  # uint8 0/255 mask


class Matter(Protocol):
    name: str
    def matte(self, image: Image.Image, mask: np.ndarray) -> np.ndarray: ...  # float32 [0,1]


# ───────────────────────── SAM 3 segmenter (text-prompted) ─────────────────────────

class Sam3Segmenter:
    """Meta SAM 3 / 3.1 with text prompt 'hair'.

    Uses Meta's official `sam3` Python package
    (https://github.com/facebookresearch/sam3) which supports both:
      • SAM 3   — `facebook/sam3`     (Nov 2025)
      • SAM 3.1 — `facebook/sam3.1`   (Mar 2026, Object Multiplex)

    Default version is "sam3.1" — same image-quality plus the better video
    backbone for when we add video later. Pass `version="sam3"` to fall back
    to the original.

    Setup once:
        git clone https://github.com/facebookresearch/sam3.git
        cd sam3 && pip install -e .
        hf download facebook/sam3.1            # gated, requires approval
    """

    def __init__(self,
                 device: str | None = None,
                 version: str = "sam3.1",
                 text_prompt: str | list[str] | dict | None = None,
                 checkpoint_path: str | None = None):
        try:
            from sam3.model_builder import (
                build_sam3_image_model,
                download_ckpt_from_hf,
            )
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "Meta `sam3` package not installed. Run:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e ."
            ) from exc

        self.device = pick_device(device)
        # SAM 3 / 3.1 internally use Metal-incompatible matmul dtype patterns
        # that crash on MPS (verified on M-series Macs). Force CPU when the
        # caller asked for MPS — better a slow result than a crash.
        if self.device == "mps":
            log.warning("SAM 3 not stable on MPS; running on CPU instead")
            self.device = "cpu"
        self.version = version
        # Normalise prompt config into {"targets": [...], "restrict_to": str|None}.
        if text_prompt is None:
            text_prompt = "hair"
        if isinstance(text_prompt, str):
            cfg = {"targets": [text_prompt], "restrict_to": None}
        elif isinstance(text_prompt, list):
            cfg = {"targets": list(text_prompt), "restrict_to": None}
        elif isinstance(text_prompt, dict):
            cfg = {
                "targets": list(text_prompt.get("targets", ["hair"])),
                "restrict_to": text_prompt.get("restrict_to"),
            }
        else:
            raise TypeError(f"unsupported text_prompt type: {type(text_prompt)}")
        self.targets: list[str] = cfg["targets"]
        self.restrict_to: str | None = cfg["restrict_to"]
        name = "+".join(self.targets)
        if self.restrict_to:
            name += f"@{self.restrict_to}"
        self.name = f"{version}:{name}"

        # Meta's builder only branches on 'cuda'; everything else is CPU.
        build_device = "cuda" if self.device == "cuda" else "cpu"

        if checkpoint_path is None:
            log.info(f"resolving {version} checkpoint via HF cache…")
            checkpoint_path = download_ckpt_from_hf(version=version)

        log.info(f"building {version} on {self.device} from {checkpoint_path}")
        model = build_sam3_image_model(
            device=build_device,
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
        # Some weights ship as bf16 (designed for CUDA autocast). On Mac
        # CPU/MPS that triggers dtype mismatches (Float vs BFloat16). Coerce
        # everything to float32 for consistent inference.
        model = model.float()
        if self.device == "mps":
            try:
                model = model.to("mps")
            except Exception as e:
                log.warning(f"MPS migration failed ({e}); staying on CPU")
                self.device = "cpu"
        self.model = model
        # Force the processor to the same device as the model — its default
        # auto-detect can pick MPS even when model is on CPU.
        self.processor = Sam3Processor(model, device=self.device)

    @staticmethod
    def _output_to_union(output, h: int, w: int) -> np.ndarray:
        """Collapse SAM 3 output['masks'] into a bool union mask."""
        masks = output.get("masks") if isinstance(output, dict) else None
        union = np.zeros((h, w), dtype=bool)
        if masks is None:
            return union
        if isinstance(masks, torch.Tensor):
            if masks.numel() == 0:
                return union
            arr = masks.bool().cpu().numpy()
            while arr.ndim > 3:
                arr = arr[:, 0]
            union = arr.any(axis=0)
        else:
            for m in masks:
                a = m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
                while a.ndim > 2:
                    a = a[0]
                union |= a.astype(bool)
        return union

    def _run_prompt_on_state(self, state, prompt: str, h: int, w: int) -> np.ndarray:
        """Run one text prompt on a pre-encoded image state; return bool mask.

        We `reset_all_prompts` first so each call gives the mask for *only*
        the new prompt, never accidentally accumulating across prompts.
        """
        self.processor.reset_all_prompts(state)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._output_to_union(output, h, w)

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        # Encode the image once; state["backbone_out"] holds the heavy ViT
        # features. All subsequent prompts are cheap text-only passes.
        state = self.processor.set_image(image)

        target_union = np.zeros((image.height, image.width), dtype=bool)
        for prompt in self.targets:
            log.info(f"sam3 target: {prompt!r}")
            target_union |= self._run_prompt_on_state(
                state, prompt, image.height, image.width
            )

        if self.restrict_to:
            log.info(f"sam3 restriction: {self.restrict_to!r}")
            restriction = self._run_prompt_on_state(
                state, self.restrict_to, image.height, image.width
            )
            cov = restriction.mean()
            if cov < 1e-4:
                log.warning(
                    f"restriction prompt {self.restrict_to!r} matched no "
                    f"pixels ({cov:.4%}); returning empty mask"
                )
                return np.zeros((image.height, image.width), dtype=np.uint8)
            target_union &= restriction

        return target_union.astype(np.uint8) * 255

    @torch.inference_mode()
    def detect(self, image: Image.Image, prompt: str) -> float:
        """Return fraction of image covered by `prompt` — for soft checks
        like 'is there a woman in this photo'."""
        state = self.processor.set_image(image)
        return float(self._run_prompt_on_state(
            state, prompt, image.height, image.width
        ).mean())


# ───────────────────────── Sapiens segmenter ─────────────────────────

SAPIENS_INPUT_H, SAPIENS_INPUT_W = 1024, 768
SAPIENS_MEAN = np.array([123.5, 116.5, 103.5], dtype=np.float32)
SAPIENS_STD = np.array([58.5, 57.0, 57.5], dtype=np.float32)

# Sapiens body-part-seg taxonomy (Goliath, 28 classes).
SAPIENS_CLASSES = (
    "Background", "Apparel", "Face_Neck", "Hair",
    "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
    "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg",
    "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
    "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm",
    "Right_Upper_Leg", "Torso", "Upper_Clothing",
    "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue",
)
SAPIENS_NAME_TO_IDX = {n: i for i, n in enumerate(SAPIENS_CLASSES)}
SAPIENS_HAIR_CLASS = SAPIENS_NAME_TO_IDX["Hair"]

# Class sets per pipeline mode. "modesty" blurs hair + exposed-skin classes
# but leaves Face_Neck (face) and clothing classes visible.
SAPIENS_CLASS_SETS: dict[str, set[str]] = {
    "hair": {"Hair"},
    "modesty": {
        "Hair", "Torso",
        "Left_Hand", "Left_Foot", "Left_Lower_Arm", "Left_Upper_Arm",
        "Left_Lower_Leg", "Left_Upper_Leg",
        "Right_Hand", "Right_Foot", "Right_Lower_Arm", "Right_Upper_Arm",
        "Right_Lower_Leg", "Right_Upper_Leg",
    },
}

# SAM 3 prompt configuration per pipeline mode.
#   targets:     prompts whose masks are union'd to form what gets blurred
#   restrict_to: optional silhouette prompt — final mask is target ∩ this.
#                Used for gender-aware modesty so men's hair / clothing
#                in mixed-gender photos isn't blurred.
SAM3_PROMPT_SETS: dict[str, dict] = {
    "hair": {
        "targets": ["hair"],
        "restrict_to": None,
    },
    "modesty": {
        # Simple noun prompts work best with SAM 3 (compound phrases like
        # "hair on a woman" return ~0%). The `restrict_to` intersection
        # with the "woman" silhouette is what filters out men. We avoid
        # prompts that match the face (e.g. "exposed skin", "skin"); face
        # remains visible by design — this matches typical hijab-style
        # modesty rules where hair and body are covered but face isn't.
        "targets": [
            "hair",
            "bare arms",
            "bare shoulders",
            "bare legs",
            "thighs",
            "midriff",
            "neckline",
            "cleavage",
            "decolletage",
            "exposed chest",
        ],
        "restrict_to": "woman",
    },
}

SAPIENS_CHECKPOINTS = {
    "0.3b": ("facebook/sapiens-seg-0.3b-torchscript",
             "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"),
    "0.6b": ("facebook/sapiens-seg-0.6b-torchscript",
             "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"),
    "1b":   ("facebook/sapiens-seg-1b-torchscript",
             "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"),
    # Note: "2b" was removed — facebook/sapiens-seg-2b-torchscript no longer
    # exists publicly on HF. Largest available torchscript variant is 1b.
}


def _letterbox(image: Image.Image, target_h: int, target_w: int):
    src_w, src_h = image.size
    scale = min(target_h / src_h, target_w / src_w)
    new_h, new_w = int(round(src_h * scale)), int(round(src_w * scale))
    resized = image.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas.paste(resized, (left, top))
    return canvas, (top, left, new_h, new_w)


class SapiensSegmenter:
    def __init__(self, size: str = "1b", device: str | None = None,
                 target_classes: set[str] | None = None):
        from huggingface_hub import hf_hub_download

        if size not in SAPIENS_CHECKPOINTS:
            raise ValueError(f"unknown sapiens size {size!r}")
        self.size = size
        self.device = pick_device(device)
        # Default target = just Hair (single-class behaviour).
        if target_classes is None:
            target_classes = SAPIENS_CLASS_SETS["hair"]
        self.target_class_names = sorted(target_classes)
        unknown = set(target_classes) - set(SAPIENS_NAME_TO_IDX)
        if unknown:
            raise ValueError(f"unknown sapiens classes: {unknown}")
        self.target_class_indices = np.array(
            [SAPIENS_NAME_TO_IDX[n] for n in self.target_class_names],
            dtype=np.int64,
        )
        self.name = f"sapiens-{size}:{'+'.join(self.target_class_names)}"
        repo_id, filename = SAPIENS_CHECKPOINTS[size]
        log.info(f"loading {self.name} from {repo_id}/{filename} on {self.device}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = torch.jit.load(path, map_location=self.device).eval()

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        # Match the official Sapiens preprocessing exactly: direct resize to
        # (W, H) with bilinear interpolation, RGB order, normalize with mean/std.
        # No letterbox — Sapiens was trained on aspect-ratio-distorted inputs.
        resized = image.resize((SAPIENS_INPUT_W, SAPIENS_INPUT_H), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        arr = (arr - SAPIENS_MEAN) / SAPIENS_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        # Upsample logits to original image size before argmax — preserves
        # boundary detail better than nearest-resizing the discrete label map.
        logits = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width),
            mode="bilinear", align_corners=False,
        )
        labels = logits.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(labels, self.target_class_indices)
        return (mask.astype(np.uint8) * 255)


# ───────────────────────── SegFormer fallback segmenter ─────────────────────────

class SegFormerSegmenter:
    """Lighter face-parsing fallback (`jonathandinu/face-parsing`, SegFormer-B5)."""

    def __init__(self, device: str | None = None):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = pick_device(device)
        self.name = "segformer-b5-faceparsing"
        log.info(f"loading {self.name} on {self.device}")
        self.processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = (SegformerForSemanticSegmentation
                      .from_pretrained("jonathandinu/face-parsing")
                      .to(self.device).eval())
        id2label = self.model.config.id2label
        self.hair_id = next((int(i) for i, n in id2label.items() if str(n).lower() == "hair"), None)
        if self.hair_id is None:
            raise RuntimeError(f"no hair label in {id2label}")

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()
        return (labels == self.hair_id).astype(np.uint8) * 255


# ───────────────────────── MatAnyone matter ─────────────────────────

class MatAnyoneMatter:
    def __init__(self):
        from matanyone import InferenceCore  # noqa: F401  (early import-check)
        self.name = "matanyone"
        self._core_cls = InferenceCore
        self._core = None

    def _get_core(self):
        if self._core is None:
            log.info("instantiating MatAnyone (PeiqingYang/MatAnyone)")
            self._core = self._core_cls("PeiqingYang/MatAnyone")
        return self._core

    def matte(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        # MatAnyone's process_video expects a *directory* of frames (or an
        # mp4), not a single image. Wrap the single image as a one-frame
        # "video" in a temp dir, ask for per-frame PNG output, and pick up
        # the alpha matte at out/frames/pha/00000.png.
        work = Path(tempfile.mkdtemp(prefix="matanyone_"))
        try:
            frames_dir = work / "frames"
            frames_dir.mkdir()
            mask_path = work / "mask.png"
            out_dir = work / "out"
            out_dir.mkdir()
            image.save(frames_dir / "0001.jpg")
            Image.fromarray(mask, mode="L").save(mask_path)

            core = self._get_core()
            core.process_video(
                input_path=str(frames_dir),
                mask_path=str(mask_path),
                output_path=str(out_dir),
                n_warmup=10,
                save_image=True,
            )
            alpha_pngs = sorted((out_dir / "frames" / "pha").glob("*.png"))
            if not alpha_pngs:
                raise RuntimeError(
                    f"MatAnyone produced no alpha PNGs under {out_dir}/frames/pha"
                )
            alpha_pil = Image.open(alpha_pngs[0]).convert("L")
            if alpha_pil.size != image.size:
                alpha_pil = alpha_pil.resize(image.size, Image.BILINEAR)
            return np.asarray(alpha_pil, dtype=np.float32) / 255.0
        finally:
            shutil.rmtree(work, ignore_errors=True)


# ───────────────────────── Feather fallback matter ─────────────────────────

class FeatherMatter:
    """Pure-PIL feather of the binary mask. No matting, but always available."""

    def __init__(self, radius: float = 4.0):
        self.name = f"feather-r{radius}"
        self.radius = radius

    def matte(self, image: Image.Image, mask: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(mask, mode="L")
        if self.radius > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return np.asarray(pil, dtype=np.float32) / 255.0


# ───────────────────────── Hybrid: Sapiens body + SAM 3 gender ──────────

class HybridSegmenter:
    """Best-quality modesty segmenter for mixed-gender / multi-person photos.

    body  = Sapiens parsing (mode-specific class set: hair + skin classes)
    gender = SAM 3 silhouette of `restrict_to` (default "woman")
    final = body ∩ gender, plus an optional SAM 3 "hair" pass union'd into
    the body mask before intersection (Sapiens hair coverage is uneven on
    portraits; SAM 3 hair is reliably strong).
    """

    def __init__(self,
                 sapiens_size: str = "1b",
                 device: str | None = None,
                 mode: str = "modesty",
                 sam3_version: str = "sam3.1",
                 restrict_to: str = "woman",
                 sam3_targets: list[str] | None = None):
        self.body = SapiensSegmenter(
            size=sapiens_size,
            device=device,
            target_classes=SAPIENS_CLASS_SETS[mode],
        )
        # SAM 3 prompts to run *in addition* to the Sapiens body parsing:
        # patch up things Sapiens misses on small bodies in crowded photos.
        if sam3_targets is None:
            sam3_targets = list(SAM3_PROMPT_SETS[mode]["targets"])
        self.sam3_targets: list[str] = sam3_targets
        # The Sam3Segmenter itself isn't used via .segment() here — we just
        # need its loaded model and processor for direct prompt calls.
        self.sam3 = Sam3Segmenter(
            device=device,
            version=sam3_version,
            text_prompt={"targets": sam3_targets, "restrict_to": None},
        )
        self.restrict_to = restrict_to
        self.name = (
            f"hybrid:{self.body.name}"
            f"+{self.sam3.version}:{'+'.join(sam3_targets)}"
            f"@{restrict_to}"
        )

    def segment(self, image: Image.Image) -> np.ndarray:
        body_mask = self.body.segment(image)

        # Single heavy ViT pass; everything below is cheap text-only forward.
        state = self.sam3.processor.set_image(image)

        sam3_union = np.zeros((image.height, image.width), dtype=bool)
        for prompt in self.sam3_targets:
            log.info(f"hybrid sam3 target: {prompt!r}")
            sam3_union |= self.sam3._run_prompt_on_state(
                state, prompt, image.height, image.width
            )

        gender = self.sam3._run_prompt_on_state(
            state, self.restrict_to, image.height, image.width
        )
        if gender.mean() < 1e-4:
            log.warning(
                f"hybrid: '{self.restrict_to}' silhouette empty; returning empty mask"
            )
            return np.zeros((image.height, image.width), dtype=np.uint8)

        target = (body_mask > 127) | sam3_union
        return ((target & gender).astype(np.uint8) * 255)


# ───────────────────────── builders with fallback ─────────────────────────

PIPELINE_MODES = ("hair", "modesty")


def build_segmenter(prefer: str = "sam3", sapiens_size: str = "1b",
                    device: str | None = None,
                    sam3_version: str = "sam3.1",
                    mode: str = "hair") -> Segmenter:
    """Try the requested model first; cascade through fallbacks on failure.

    Default cascade order for prefer="sam3": sam3 → sapiens → segformer.
    `mode` selects which classes / prompts the segmenter is configured for:
        - "hair":     blur hair only
        - "modesty":  blur hair + every exposed-skin body class.
                      SegFormer face-parsing has no body classes, so on this
                      mode it gets dropped from the fallback chain.
    """
    if mode not in PIPELINE_MODES:
        raise ValueError(f"unknown mode {mode!r}; valid: {PIPELINE_MODES}")

    if prefer == "hybrid":
        # Hybrid only makes sense for modesty (it's body parsing + gender).
        if mode != "modesty":
            log.warning("hybrid segmenter is for modesty mode; falling back "
                        "to sam3 since mode is hair-only")
            chain = ["sam3", "sapiens", "segformer"]
        else:
            chain = ["hybrid", "sam3", "sapiens"]
    elif prefer == "sam3":
        chain = ["sam3", "sapiens", "segformer"]
    elif prefer == "sapiens":
        chain = ["sapiens", "segformer"]
    elif prefer == "segformer":
        chain = ["segformer"]
    else:
        raise ValueError(f"unknown segmenter {prefer!r}")

    if mode == "modesty":
        # SegFormer face-parsing knows about hair but not about arms / legs
        # / torso, so it would silently produce a hair-only mask in modesty
        # mode and mislead the user. Drop it.
        chain = [c for c in chain if c != "segformer"]
        if not chain:
            raise RuntimeError("modesty mode requires sapiens / sam3 / hybrid")

    last_err: Exception | None = None
    for choice in chain:
        try:
            if choice == "hybrid":
                return HybridSegmenter(
                    sapiens_size=sapiens_size,
                    device=device,
                    mode=mode,
                    sam3_version=sam3_version,
                )
            if choice == "sam3":
                return Sam3Segmenter(
                    device=device,
                    version=sam3_version,
                    text_prompt=SAM3_PROMPT_SETS[mode],
                )
            if choice == "sapiens":
                return SapiensSegmenter(
                    size=sapiens_size,
                    device=device,
                    target_classes=SAPIENS_CLASS_SETS[mode],
                )
            if choice == "segformer":
                return SegFormerSegmenter(device=device)
        except Exception as e:
            log.warning(f"{choice} unavailable ({e}); trying next fallback")
            last_err = e
    raise RuntimeError(f"all segmenters failed; last error: {last_err}")


def build_matter(prefer: str = "matanyone", feather_radius: float = 4.0) -> Matter:
    if prefer == "matanyone":
        try:
            return MatAnyoneMatter()
        except Exception as e:
            log.warning(f"MatAnyone unavailable ({e}); falling back to feather")
    return FeatherMatter(radius=feather_radius)


# ───────────────────────── compositing ─────────────────────────

def composite_blur(image: Image.Image, alpha: np.ndarray, blur_radius: float) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    src = np.asarray(image, dtype=np.float32)
    blr = np.asarray(blurred, dtype=np.float32)
    a = alpha[..., None]
    out = a * blr + (1.0 - a) * src
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.0005) -> np.ndarray:
    """Drop isolated specks and fill tiny holes (cv2-based, optional)."""
    try:
        import cv2
    except ImportError:
        return mask
    binary = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(1, int(binary.size * min_area_ratio))
    keep = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel)
    return (closed * 255).astype(np.uint8)


# ───────────────────────── pipeline ─────────────────────────

@dataclass
class BlurResult:
    output: Image.Image
    mask: Image.Image          # binary semantic mask (uint8 L)
    alpha: Image.Image         # soft alpha matte (uint8 L)
    segmenter: str             # which segmenter actually ran
    matter: str                # which matter actually ran
    coverage: float            # fraction of image flagged for blur (0..1)
    mode: str                  # "hair" or "modesty"
    woman_check: float | None  # fraction "woman" was detected on, or None


class HairBlurPipeline:
    def __init__(self,
                 segmenter: Segmenter | None = None,
                 matter: Matter | None = None,
                 sapiens_size: str = "1b",
                 prefer_segmenter: str = "sam3",
                 prefer_matter: str = "matanyone",
                 device: str | None = None,
                 feather_radius: float = 4.0,
                 sam3_version: str = "sam3.1",
                 mode: str = "hair"):
        if mode not in PIPELINE_MODES:
            raise ValueError(f"unknown mode {mode!r}; valid: {PIPELINE_MODES}")
        self.mode = mode
        self.segmenter = segmenter or build_segmenter(
            prefer_segmenter, sapiens_size, device,
            sam3_version=sam3_version, mode=mode,
        )
        self.matter = matter or build_matter(prefer_matter, feather_radius)

    def __call__(self, image: Image.Image, blur_radius: float = 25.0,
                 do_clean: bool = True,
                 check_woman: bool = False,
                 woman_threshold: float = 0.01) -> BlurResult:
        if image.mode != "RGB":
            image = image.convert("RGB")

        woman_cov: float | None = None
        if check_woman:
            if not isinstance(self.segmenter, Sam3Segmenter):
                log.warning("woman check requested but only SAM 3 supports it; "
                            "skipping check")
            else:
                woman_cov = self.segmenter.detect(image, "woman")
                log.info(f"woman coverage: {woman_cov:.2%}")
                if woman_cov < woman_threshold:
                    log.warning(
                        f"no female silhouette detected (woman coverage "
                        f"{woman_cov:.2%} < {woman_threshold:.2%}); "
                        f"running blur anyway"
                    )

        mask = self.segmenter.segment(image)
        if do_clean:
            mask = clean_mask(mask)
        coverage = float((mask > 127).mean())
        if coverage < 1e-4:
            log.warning(f"target coverage {coverage:.4%}; mask may be empty")

        alpha = self.matter.matte(image, mask)
        out = composite_blur(image, alpha, blur_radius=blur_radius)
        return BlurResult(
            output=out,
            mask=Image.fromarray(mask, mode="L"),
            alpha=Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L"),
            segmenter=self.segmenter.name,
            matter=self.matter.name,
            coverage=coverage,
            mode=self.mode,
            woman_check=woman_cov,
        )
