"""Modesty-aware blur pipeline (modular, with graceful fallbacks).

Primary path:    Hybrid (Sapiens body + SAM 3 woman silhouette)
                 → MatAnyone → alpha-blended Gaussian blur
Fallback chain:  SAM 3 multi-prompt → Sapiens body classes
Matter fallback: MatAnyone  →  feathered binary mask

The pipeline always returns an output, mask and alpha — even if every "heavy"
dependency is missing — so the UI never throws at the user.

There is exactly one mode: blur hair + every exposed-skin body class on
detected women, leave face / clothing / background untouched.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
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


# ───────────────────────── helpers ─────────────────────────

def image_content_hash(image: Image.Image) -> str:
    """Fast content hash of a PIL image. ~5–10 ms for a 2 MP image.

    Used as a key for the SAM 3 image-state cache and the pipeline-level
    mask cache so re-running the same image (e.g. tweaking blur radius
    in the UI) skips the expensive ViT pass.
    """
    arr = np.asarray(image)
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


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
                 checkpoint_path: str | None = None,
                 state_cache_size: int = 4):
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

        # Cache encoded image state by content hash. The heavy ViT pass
        # behind set_image is the dominant cost on CPU (~25s on Apple
        # Silicon for a 1000×667 photo); reusing it across repeated calls
        # makes UI tweaks (blur radius, matter type) effectively free.
        self._state_cache: dict[str, dict] = {}
        self._state_cache_lock = threading.Lock()
        self._state_cache_max = state_cache_size

    def get_image_state(self, image: Image.Image):
        """Return SAM 3 image state, encoding once and caching by hash.

        On a cache hit we still call `reset_all_prompts` so the returned
        state has no leftover text/box prompts from previous queries.
        """
        h = image_content_hash(image)
        with self._state_cache_lock:
            cached = self._state_cache.get(h)
            if cached is not None:
                self.processor.reset_all_prompts(cached)
                log.info(f"sam3 state cache hit ({h[:8]})")
                return cached
        # Heavy path — compute encoding outside the lock so we don't block
        # other threads asking for unrelated images.
        log.info(f"sam3 encoding image ({h[:8]})…")
        state = self.processor.set_image(image)
        with self._state_cache_lock:
            if len(self._state_cache) >= self._state_cache_max:
                self._state_cache.pop(next(iter(self._state_cache)))
            self._state_cache[h] = state
        return state

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
    def prompt_mask_and_boxes(
        self, image: Image.Image, prompt: str,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Backend-agnostic API: return (bool union mask, list of xyxy boxes).

        Works on the cached image state. HybridSegmenter calls this so the
        same code path works against either the PyTorch or the MLX SAM 3
        backend.
        """
        state = self.get_image_state(image)
        self.processor.reset_all_prompts(state)
        output = self.processor.set_text_prompt(state=state, prompt=prompt)
        h, w = image.height, image.width
        mask = self._output_to_union(output, h, w)

        boxes: list[tuple[int, int, int, int]] = []
        boxes_t = output.get("boxes") if isinstance(output, dict) else None
        if isinstance(boxes_t, torch.Tensor) and boxes_t.numel() > 0:
            arr = boxes_t.detach().cpu().numpy()
            while arr.ndim > 2:
                arr = arr[0]
            for b in arr:
                x1, y1, x2, y2 = (int(round(v)) for v in b[:4].tolist())
                boxes.append((
                    max(0, min(w, x1)), max(0, min(h, y1)),
                    max(0, min(w, x2)), max(0, min(h, y2)),
                ))
        return mask, boxes

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        # Cached encoding — single heavy ViT pass per unique image.
        state = self.get_image_state(image)

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
        state = self.get_image_state(image)
        return float(self._run_prompt_on_state(
            state, prompt, image.height, image.width
        ).mean())


# ───────────────────────── SAM 3 / 3.1 via mlx-vlm (Apple Silicon) ──────

DEFAULT_MLX_SAM3_PATH = Path(__file__).parent / "models" / "sam3.1-mlx"


class Sam3MLXSegmenter:
    """SAM 3.1 running on Apple Silicon via mlx-vlm.

    Same public surface as the PyTorch `Sam3Segmenter` so HybridSegmenter
    can use either backend interchangeably:
        get_image_state(image)
        prompt_mask_and_boxes(image, prompt)
        segment(image)
        detect(image, prompt)

    Heavy ViT backbone is encoded once per image (cached by content hash);
    each text prompt only runs the cheap FPN neck + DETR + mask decoder.
    On M-series Macs this brings SAM 3 from ~25 s (PyTorch CPU) to ~3-4 s
    cold and ~0.5-1 s per additional prompt.
    """

    def __init__(self,
                 model_path: str | Path | None = None,
                 score_threshold: float = 0.5,
                 state_cache_size: int = 4):
        try:
            import mlx.core as mx
            from mlx_vlm.utils import load_model
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
            from mlx_vlm.models.sam3_1.generate import (
                _get_backbone_features,
                _detect_with_backbone,
            )
        except ImportError as exc:
            raise RuntimeError(
                "mlx-vlm not installed. Run:\n"
                "  pip install 'mlx-vlm>=0.4.3'"
            ) from exc

        path = Path(model_path) if model_path else DEFAULT_MLX_SAM3_PATH
        if not (path / "config.json").exists():
            raise FileNotFoundError(
                f"MLX SAM 3.1 checkpoint not found at {path}. "
                f"Download with: hf download mlx-community/sam3.1-bf16 "
                f"--local-dir {path}"
            )

        self._mx = mx
        self.path = path
        self.score_threshold = score_threshold
        self.name = f"sam3.1-mlx"

        log.info(f"loading {self.name} from {path}…")
        self.model = load_model(path)
        self.processor = Sam31Processor.from_pretrained(str(path))
        self.predictor = Sam3Predictor(
            self.model, self.processor, score_threshold=score_threshold,
        )
        self._get_backbone = _get_backbone_features
        self._detect_with_backbone = _detect_with_backbone

        self._state_cache: dict[str, dict] = {}
        self._state_cache_lock = threading.Lock()
        self._state_cache_max = state_cache_size

    def get_image_state(self, image: Image.Image) -> dict:
        """Encode ViT backbone once per unique image; cached by content hash."""
        h = image_content_hash(image)
        with self._state_cache_lock:
            cached = self._state_cache.get(h)
            if cached is not None:
                log.info(f"sam3-mlx state cache hit ({h[:8]})")
                return cached
        log.info(f"sam3-mlx encoding image ({h[:8]})…")
        inputs = self.processor.preprocess_image(image)
        pixel_values = self._mx.array(inputs["pixel_values"])
        backbone = self._get_backbone(self.model, pixel_values)
        state = {"backbone": backbone, "size": image.size}
        with self._state_cache_lock:
            if len(self._state_cache) >= self._state_cache_max:
                self._state_cache.pop(next(iter(self._state_cache)))
            self._state_cache[h] = state
        return state

    def prompt_mask_and_boxes(
        self, image: Image.Image, prompt: str,
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Same return shape as Sam3Segmenter.prompt_mask_and_boxes."""
        state = self.get_image_state(image)
        result = self._detect_with_backbone(
            self.predictor, state["backbone"], [prompt],
            state["size"], self.score_threshold,
        )

        h, w = image.height, image.width
        union = np.zeros((h, w), dtype=bool)
        masks = result.masks
        if masks is not None and len(masks) > 0:
            for m in masks:
                arr = np.asarray(m)
                bm = arr if arr.dtype == bool else (arr > 0)
                if bm.shape != (h, w):
                    bm = (
                        np.array(
                            Image.fromarray(bm.astype(np.uint8) * 255).resize((w, h))
                        ) > 127
                    )
                union |= bm

        boxes: list[tuple[int, int, int, int]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for b in np.asarray(result.boxes):
                x1, y1, x2, y2 = (int(round(float(v))) for v in b[:4])
                boxes.append((
                    max(0, min(w, x1)), max(0, min(h, y1)),
                    max(0, min(w, x2)), max(0, min(h, y2)),
                ))
        return union, boxes

    def segment(self, image: Image.Image,
                targets: list[str] | None = None,
                restrict_to: str | None = None) -> np.ndarray:
        """Standalone modesty pass — multi-prompt union ∩ optional restriction.

        HybridSegmenter doesn't call this; it's here so the MLX segmenter
        can stand on its own as a fallback for the cascade in build_segmenter.
        """
        if targets is None:
            targets = list(SAM3_MODESTY_TARGETS)
        if restrict_to is None:
            restrict_to = SAM3_MODESTY_RESTRICT_TO

        h, w = image.height, image.width
        union = np.zeros((h, w), dtype=bool)
        for prompt in targets:
            log.info(f"sam3-mlx target: {prompt!r}")
            mask, _ = self.prompt_mask_and_boxes(image, prompt)
            union |= mask
        if restrict_to:
            log.info(f"sam3-mlx restriction: {restrict_to!r}")
            r, _ = self.prompt_mask_and_boxes(image, restrict_to)
            if r.mean() < 1e-4:
                log.warning(
                    f"sam3-mlx: '{restrict_to}' silhouette empty; empty mask"
                )
                return np.zeros((h, w), dtype=np.uint8)
            union &= r
        return union.astype(np.uint8) * 255

    def detect(self, image: Image.Image, prompt: str) -> float:
        """Coverage check — 'is there a woman in this photo' style soft test."""
        mask, _ = self.prompt_mask_and_boxes(image, prompt)
        return float(mask.mean())


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

# Modesty class set — blurs hair + every exposed-skin body class but leaves
# Face_Neck (face) and clothing classes visible.
SAPIENS_MODESTY_CLASSES: set[str] = {
    "Hair", "Torso",
    "Left_Hand", "Left_Foot", "Left_Lower_Arm", "Left_Upper_Arm",
    "Left_Lower_Leg", "Left_Upper_Leg",
    "Right_Hand", "Right_Foot", "Right_Lower_Arm", "Right_Upper_Arm",
    "Right_Lower_Leg", "Right_Upper_Leg",
}

# SAM 3 prompt set for modesty. Simple noun prompts work best with SAM 3
# (compound phrases like "hair on a woman" return ~0%). The `restrict_to`
# intersection with the "woman" silhouette filters out men. We avoid
# prompts that match the face (e.g. "exposed skin", "skin") — face stays
# visible by design, matching typical hijab-style modesty rules.
#
# Pruned set: empirically `cleavage`, `decolletage`, `exposed chest` return
# ~0% coverage on most photos (they only fire on actually nude shots) and
# add ~1.5 s per image without measurable quality benefit. Kept as
# SAM3_MODESTY_TARGETS_FULL for callers who want the wider net.
SAM3_MODESTY_TARGETS: list[str] = [
    "hair",
    "bare arms",
    "bare shoulders",
    "bare legs",
    "thighs",
    "midriff",
    "neckline",
]
SAM3_MODESTY_TARGETS_FULL: list[str] = SAM3_MODESTY_TARGETS + [
    "cleavage", "decolletage", "exposed chest",
]
SAM3_MODESTY_RESTRICT_TO: str = "woman"

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
        # Default = the full modesty class set (hair + body skin).
        if target_classes is None:
            target_classes = SAPIENS_MODESTY_CLASSES
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

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Sapiens-style: direct resize to (W, H) bilinear, RGB, normalize.
        Returns CHW float32 tensor (no batch dim)."""
        resized = image.resize((SAPIENS_INPUT_W, SAPIENS_INPUT_H), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        arr = (arr - SAPIENS_MEAN) / SAPIENS_STD
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _logits_to_mask(self, logits_1chw: torch.Tensor,
                        target_h: int, target_w: int) -> np.ndarray:
        """Upsample logits to original image size, argmax, return uint8 mask."""
        logits_1chw = torch.nn.functional.interpolate(
            logits_1chw, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        labels = logits_1chw.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(labels, self.target_class_indices)
        return mask.astype(np.uint8) * 255

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        return self._logits_to_mask(logits, image.height, image.width)

    @torch.inference_mode()
    def segment_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Batch version — N images in one forward pass.

        All inputs are resized to (1024, 768) regardless of aspect ratio
        (Sapiens was trained that way). Output masks are returned at each
        image's original resolution.
        """
        if not images:
            return []
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        logits = self.model(batch)  # (B, C, H/2, W/2)
        results = []
        for i, img in enumerate(images):
            results.append(self._logits_to_mask(
                logits[i:i + 1], img.height, img.width
            ))
        return results


# ───────────────────────── SegFormer-ATR (clothes/parts) segmenter ─────────────────────────

ATR_CLASSES = (
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes",
    "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe",
    "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
    "Bag", "Scarf",
)
ATR_NAME_TO_IDX = {n: i for i, n in enumerate(ATR_CLASSES)}

# ATR has no midriff/cleavage/neckline labels — those still need SAM 3
# (or a fine-tuned head) if you care about them.
ATR_MODESTY_CLASSES: set[str] = {
    "Hair", "Left-arm", "Right-arm", "Left-leg", "Right-leg",
}


class SegformerATRSegmenter:
    """`mattmdjaga/segformer_b2_clothes` — SegFormer-B2 fine-tuned on ATR.

    18 classes (hair, face, body parts, clothing). ~80M parameters — about
    10× lighter than Sapiens-1b at comparable quality on the hair + bare
    arms/legs class set.
    """

    def __init__(self, device: str | None = None,
                 target_classes: set[str] | None = None,
                 repo_id: str = "mattmdjaga/segformer_b2_clothes",
                 compile_model: bool = False):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.device = pick_device(device)
        self.repo_id = repo_id
        if target_classes is None:
            target_classes = ATR_MODESTY_CLASSES
        unknown = set(target_classes) - set(ATR_NAME_TO_IDX)
        if unknown:
            raise ValueError(f"unknown ATR classes: {unknown}")
        self.target_class_names = sorted(target_classes)
        self.target_class_indices = np.array(
            [ATR_NAME_TO_IDX[n] for n in self.target_class_names],
            dtype=np.int64,
        )
        self.name = f"segformer-atr:{'+'.join(self.target_class_names)}"
        log.info(f"loading {self.name} from {repo_id} on {self.device}")
        self.processor = SegformerImageProcessor.from_pretrained(repo_id)
        self.model = (SegformerForSemanticSegmentation
                      .from_pretrained(repo_id)
                      .to(self.device).eval())

        # torch.compile fuses ATR forward into a single graph — typically
        # 20–30 % faster on repeated calls. First call is ~1 s slower
        # (compile), subsequent ones noticeably quicker. Wrapped in
        # try/except because MPS dynamo backend is sometimes flaky.
        if compile_model:
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False,
                )
                log.info(f"{self.name}: torch.compile enabled")
            except Exception as e:
                log.warning(f"{self.name}: torch.compile failed ({e}); using eager")

    @torch.inference_mode()
    def segment(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width),
            mode="bilinear", align_corners=False,
        )
        labels = upsampled.argmax(dim=1)[0].cpu().numpy()
        mask = np.isin(labels, self.target_class_indices)
        return mask.astype(np.uint8) * 255

    @torch.inference_mode()
    def segment_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Batch version — N images in one forward, used by HybridSegmenter
        smart-crop pass. SegformerImageProcessor handles per-image resize +
        normalization and pads tensors to the largest in the batch."""
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits  # (B, C, H/4, W/4)
        results = []
        for i, img in enumerate(images):
            upsampled = torch.nn.functional.interpolate(
                logits[i:i + 1],
                size=(img.height, img.width),
                mode="bilinear", align_corners=False,
            )
            labels = upsampled.argmax(dim=1)[0].cpu().numpy()
            results.append(np.isin(labels, self.target_class_indices)
                           .astype(np.uint8) * 255)
        return results


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

def _build_body_segmenter(backend: str, sapiens_size: str,
                          device: str | None) -> Segmenter:
    """Body parser used by HybridSegmenter. Order: ATR (small + fast) →
    Sapiens (heavy + thorough). ATR covers Hair + arms + legs which is
    enough for modesty mode; Sapiens has Torso/Hand/Foot too but is
    much slower and less accurate on hair for portraits."""
    if backend == "atr":
        try:
            return SegformerATRSegmenter(device=device)
        except Exception as e:
            log.warning(f"ATR body backend unavailable ({e}); using sapiens")
    return SapiensSegmenter(
        size=sapiens_size, device=device,
        target_classes=SAPIENS_MODESTY_CLASSES,
    )


def _build_sam3_segmenter(backend: str, sam3_version: str,
                          device: str | None,
                          mlx_model_path: str | Path | None = None):
    """SAM 3 backend selector. Order: MLX (fast on Apple Silicon) →
    PyTorch (CPU-only on Mac, slow but always works once weights are
    downloaded)."""
    if backend == "mlx":
        try:
            return Sam3MLXSegmenter(model_path=mlx_model_path)
        except Exception as e:
            log.warning(f"MLX SAM 3 unavailable ({e}); falling back to PyTorch")
    return Sam3Segmenter(
        device=device, version=sam3_version,
        text_prompt={"targets": list(SAM3_MODESTY_TARGETS),
                     "restrict_to": SAM3_MODESTY_RESTRICT_TO},
    )


class HybridSegmenter:
    """Best-quality modesty segmenter for mixed-gender / multi-person photos.

    body   = ATR or Sapiens (hair + skin classes)
    SAM 3  = multi-prompt union (hair, bare arms/legs/..., neckline, ...)
    gender = SAM 3 silhouette of `restrict_to` (default "woman")
    final  = (body ∪ SAM 3) ∩ gender

    Both `body_backend` and `sam3_backend` are pluggable. Defaults are
    chosen for Apple Silicon: ATR (~0.2 s on MPS) + MLX SAM 3 (~3-4 s
    cold + ~0.5 s per prompt).
    """

    def __init__(self,
                 body_backend: str = "atr",
                 sam3_backend: str = "mlx",
                 sapiens_size: str = "1b",
                 device: str | None = None,
                 sam3_version: str = "sam3.1",
                 mlx_model_path: str | Path | None = None,
                 restrict_to: str = SAM3_MODESTY_RESTRICT_TO,
                 sam3_targets: list[str] | None = None,
                 smart_crop: bool = False,
                 small_instance_threshold: float = 0.15,
                 crop_padding: float = 0.15,
                 min_crop_side: int = 64,
                 max_crops_per_batch: int = 4,
                 max_total_crops: int = 24):
        self.body = _build_body_segmenter(body_backend, sapiens_size, device)
        if sam3_targets is None:
            sam3_targets = list(SAM3_MODESTY_TARGETS)
        self.sam3_targets: list[str] = sam3_targets
        # The chosen SAM 3 backend exposes prompt_mask_and_boxes(image, prompt).
        self.sam3 = _build_sam3_segmenter(
            sam3_backend, sam3_version, device, mlx_model_path,
        )
        self.restrict_to = restrict_to
        self.smart_crop = smart_crop
        self.small_instance_threshold = small_instance_threshold
        self.crop_padding = crop_padding
        self.min_crop_side = min_crop_side
        self.max_crops_per_batch = max_crops_per_batch
        self.max_total_crops = max_total_crops
        self.name = (
            f"hybrid:{self.body.name}"
            f"+{self.sam3.name}:{'+'.join(sam3_targets)}"
            f"@{restrict_to}"
            + ("+smartcrop" if smart_crop else "")
        )

    def _run_sam3(self, image: Image.Image
                  ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
        """Single SAM 3 image-encode (cached) + all target prompts + gender.

        Returns:
            sam3_union: bool mask, union of target prompt masks
            gender:     bool mask, silhouette of `restrict_to` prompt
            instances:  list of per-instance (x1, y1, x2, y2) bboxes for
                        `restrict_to` — used by smart-crop pass
        """
        h, w = image.height, image.width
        sam3_union = np.zeros((h, w), dtype=bool)
        for prompt in self.sam3_targets:
            log.info(f"hybrid sam3 target: {prompt!r}")
            mask, _ = self.sam3.prompt_mask_and_boxes(image, prompt)
            sam3_union |= mask
        log.info(f"hybrid sam3 restriction: {self.restrict_to!r}")
        gender, instances = self.sam3.prompt_mask_and_boxes(image, self.restrict_to)
        return sam3_union, gender, instances

    def _smart_crop_pass(self, image: Image.Image,
                         instances: list[tuple[int, int, int, int]]
                         ) -> np.ndarray:
        """For small woman instances, run a Sapiens pass on the crop.
        On a tight portrait of one person Sapiens already sees the body
        well, so we only crop instances smaller than `small_instance_threshold`
        of the image area. All crops go through Sapiens in a single batched
        forward — N crops cost roughly the same as 1, not N×."""
        H, W = image.height, image.width
        total_area = H * W
        crops_to_process: list[tuple[int, int, int, int, Image.Image]] = []
        for x1, y1, x2, y2 in instances:
            if x2 <= x1 or y2 <= y1:
                continue
            inst_area = (x2 - x1) * (y2 - y1)
            if inst_area / total_area >= self.small_instance_threshold:
                continue
            pw = (x2 - x1) * self.crop_padding
            ph = (y2 - y1) * self.crop_padding
            cx1 = max(0, int(x1 - pw))
            cy1 = max(0, int(y1 - ph))
            cx2 = min(W, int(x2 + pw))
            cy2 = min(H, int(y2 + ph))
            if (cx2 - cx1) < self.min_crop_side or (cy2 - cy1) < self.min_crop_side:
                continue
            crops_to_process.append(
                (cx1, cy1, cx2, cy2, image.crop((cx1, cy1, cx2, cy2)))
            )

        result = np.zeros((H, W), dtype=np.uint8)
        if not crops_to_process:
            return result

        # Hard cap so a 50-person photo doesn't tie up the GPU for minutes.
        # Pick the largest instances first — small thumbnail-sized blobs are
        # the least useful to refine anyway.
        if len(crops_to_process) > self.max_total_crops:
            log.info(
                f"smart-crop: {len(crops_to_process)} candidates, "
                f"capping to {self.max_total_crops} largest"
            )
            crops_to_process.sort(
                key=lambda c: (c[2] - c[0]) * (c[3] - c[1]), reverse=True,
            )
            crops_to_process = crops_to_process[:self.max_total_crops]

        log.info(
            f"smart-crop: {len(crops_to_process)} crops, "
            f"batch={self.max_crops_per_batch}"
        )

        # Batch through Sapiens in chunks (avoid OOM on MPS / huge groups).
        import time as _time
        t0 = _time.perf_counter()
        for i in range(0, len(crops_to_process), self.max_crops_per_batch):
            chunk = crops_to_process[i:i + self.max_crops_per_batch]
            tb = _time.perf_counter()
            crop_imgs = [c[4] for c in chunk]
            crop_masks = self.body.segment_batch(crop_imgs)
            for (cx1, cy1, cx2, cy2, _img), cmask in zip(chunk, crop_masks):
                result[cy1:cy2, cx1:cx2] = np.maximum(
                    result[cy1:cy2, cx1:cx2], cmask
                )
            log.info(
                f"smart-crop batch {i // self.max_crops_per_batch + 1}: "
                f"{len(chunk)} crops in {(_time.perf_counter() - tb):.1f}s"
            )
        log.info(f"smart-crop total: {(_time.perf_counter() - t0):.1f}s")
        return result

    def segment(self, image: Image.Image) -> np.ndarray:
        H, W = image.height, image.width
        # We used to run body+SAM 3 concurrently in a ThreadPoolExecutor
        # (different devices → no contention). With ATR (~0.2 s on MPS)
        # and MLX SAM 3 (~3-10 s) the body pass is so cheap that
        # parallelism would save < 0.5 s, and MLX has thread-local GPU
        # streams that crash when called from worker threads. So we go
        # sequential — body first, then SAM 3.
        body_full = self.body.segment(image)
        sam3_union, gender, instances = self._run_sam3(image)

        if gender.mean() < 1e-4:
            log.warning(
                f"hybrid: '{self.restrict_to}' silhouette empty; returning empty mask"
            )
            return np.zeros((H, W), dtype=np.uint8)

        # Per-instance Sapiens crops fill in body-part detail on small bodies
        # in crowded photos (where a single 1024×768 Sapiens pass downsamples
        # each person too aggressively to detect arms / legs reliably).
        body_per_instance = (
            self._smart_crop_pass(image, instances)
            if self.smart_crop and instances
            else np.zeros((H, W), dtype=np.uint8)
        )

        target = (body_full > 127) | (body_per_instance > 127) | sam3_union
        return ((target & gender).astype(np.uint8) * 255)


# ───────────────────────── builders with fallback ─────────────────────────

def build_segmenter(prefer: str = "hybrid", sapiens_size: str = "1b",
                    device: str | None = None,
                    sam3_version: str = "sam3.1",
                    smart_crop: bool = False,
                    body_backend: str = "atr",
                    sam3_backend: str = "mlx",
                    mlx_model_path: str | Path | None = None) -> Segmenter:
    """Build a segmenter; cascade through fallbacks on init failure.

    body_backend  — "atr" (~80M, ~0.2 s on MPS) or "sapiens" (~4 GB, ~12 s)
    sam3_backend  — "mlx" (Apple Silicon native) or "pytorch" (CPU only)
    """
    if prefer == "hybrid":
        chain = ["hybrid", "sam3", "atr", "sapiens"]
    elif prefer == "sam3":
        chain = ["sam3", "atr", "sapiens"]
    elif prefer == "atr":
        chain = ["atr", "sapiens"]
    elif prefer == "sapiens":
        chain = ["sapiens"]
    else:
        raise ValueError(f"unknown segmenter {prefer!r}")

    last_err: Exception | None = None
    for choice in chain:
        try:
            if choice == "hybrid":
                return HybridSegmenter(
                    body_backend=body_backend,
                    sam3_backend=sam3_backend,
                    sapiens_size=sapiens_size,
                    device=device,
                    sam3_version=sam3_version,
                    mlx_model_path=mlx_model_path,
                    smart_crop=smart_crop,
                )
            if choice == "sam3":
                return _build_sam3_segmenter(
                    sam3_backend, sam3_version, device, mlx_model_path,
                )
            if choice == "atr":
                return SegformerATRSegmenter(device=device)
            if choice == "sapiens":
                return SapiensSegmenter(
                    size=sapiens_size,
                    device=device,
                    target_classes=SAPIENS_MODESTY_CLASSES,
                )
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
    woman_check: float | None  # fraction "woman" was detected on, or None


class HairBlurPipeline:
    def __init__(self,
                 segmenter: Segmenter | None = None,
                 matter: Matter | None = None,
                 sapiens_size: str = "1b",
                 prefer_segmenter: str = "hybrid",
                 prefer_matter: str = "matanyone",
                 device: str | None = None,
                 feather_radius: float = 4.0,
                 sam3_version: str = "sam3.1",
                 smart_crop: bool = False,
                 body_backend: str = "atr",
                 sam3_backend: str = "mlx",
                 mlx_model_path: str | Path | None = None,
                 mask_cache_size: int = 4):
        self.segmenter = segmenter or build_segmenter(
            prefer_segmenter, sapiens_size, device,
            sam3_version=sam3_version, smart_crop=smart_crop,
            body_backend=body_backend, sam3_backend=sam3_backend,
            mlx_model_path=mlx_model_path,
        )
        self.matter = matter or build_matter(prefer_matter, feather_radius)
        # Cache the final (mask, alpha) per image hash. The slow stages —
        # segmentation and matting — only depend on the image and the
        # `do_clean` flag, never on blur_radius. So tweaking blur radius
        # in the UI is then just a recomposite (~50 ms).
        self._mask_cache: dict[tuple, dict] = {}
        self._mask_cache_lock = threading.Lock()
        self._mask_cache_max = mask_cache_size

    def __call__(self, image: Image.Image, blur_radius: float = 25.0,
                 do_clean: bool = True,
                 check_woman: bool = False,
                 woman_threshold: float = 0.01) -> BlurResult:
        if image.mode != "RGB":
            image = image.convert("RGB")

        cache_key = (image_content_hash(image), do_clean, bool(check_woman))
        with self._mask_cache_lock:
            cached = self._mask_cache.get(cache_key)

        if cached is not None:
            log.info(f"pipeline mask cache hit ({cache_key[0][:8]})")
            mask = cached["mask"]
            alpha = cached["alpha"]
            coverage = cached["coverage"]
            woman_cov = cached["woman_cov"]
        else:
            woman_cov = None
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

            with self._mask_cache_lock:
                if len(self._mask_cache) >= self._mask_cache_max:
                    self._mask_cache.pop(next(iter(self._mask_cache)))
                self._mask_cache[cache_key] = {
                    "mask": mask, "alpha": alpha,
                    "coverage": coverage, "woman_cov": woman_cov,
                }

        # The composite is cheap; do it on every call so blur_radius takes
        # effect even on a cache hit.
        out = composite_blur(image, alpha, blur_radius=blur_radius)
        return BlurResult(
            output=out,
            mask=Image.fromarray(mask, mode="L"),
            alpha=Image.fromarray((alpha * 255).clip(0, 255).astype(np.uint8), mode="L"),
            segmenter=self.segmenter.name,
            matter=self.matter.name,
            coverage=coverage,
            woman_check=woman_cov,
        )
