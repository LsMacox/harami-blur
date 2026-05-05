"""CLI entrypoint over pipeline.HairBlurPipeline (modesty-only)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pipeline import HairBlurPipeline, SAPIENS_CHECKPOINTS


def main() -> None:
    p = argparse.ArgumentParser(
        description="Modesty blur: hair + exposed skin of detected women, "
                    "via SAM 3 (gender) + Sapiens (body) + MatAnyone (matting)."
    )
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--blur-radius", type=float, default=25.0,
                   help="Gaussian blur radius applied to the masked region")
    p.add_argument("--sapiens-size", default="1b",
                   choices=list(SAPIENS_CHECKPOINTS.keys()),
                   help="Sapiens-seg variant; bigger = more accurate, more VRAM")
    p.add_argument("--segmenter", default="hybrid",
                   choices=["hybrid", "sam3", "atr", "sapiens"],
                   help="Force a segmenter; falls back automatically on load "
                        "failure. 'hybrid' = body parser + SAM 3 multi-prompt "
                        "+ SAM 3 woman silhouette (recommended).")
    p.add_argument("--body-backend", default="atr", choices=["atr", "sapiens"],
                   help="Body parser inside hybrid: 'atr' "
                        "(mattmdjaga/segformer_b2_clothes, ~80M, ~0.2 s on MPS) "
                        "or 'sapiens' (~4 GB, ~12 s, more classes).")
    p.add_argument("--sam3-backend", default="mlx", choices=["mlx", "pytorch"],
                   help="SAM 3 backend: 'mlx' (Apple Silicon native, ~3-4 s) "
                        "or 'pytorch' (CPU only, ~25 s on Mac).")
    p.add_argument("--sam3-version", default="sam3.1",
                   choices=["sam3", "sam3.1"],
                   help="Which SAM 3 checkpoint to use via Meta's sam3 package")
    p.add_argument("--matter", default="matanyone", choices=["matanyone", "feather"],
                   help="Force a matter; falls back automatically on load failure")
    p.add_argument("--feather-radius", type=float, default=4.0,
                   help="Feather radius used by the fallback matter")
    p.add_argument("--save-mask", type=Path, default=None)
    p.add_argument("--save-alpha", type=Path, default=None)
    p.add_argument("--device", default=None, choices=[None, "cuda", "mps", "cpu"])
    p.add_argument("--no-clean", action="store_true",
                   help="Skip morphological cleanup of the binary mask")
    p.add_argument("--check-woman", action="store_true",
                   help="Run SAM 3 with prompt 'woman' first; warn if no "
                        "female silhouette is detected. Requires --segmenter sam3.")
    p.add_argument("--smart-crop", action="store_true",
                   help="Hybrid only: extra Sapiens pass on each woman crop. "
                        "Better arm/leg detail on group photos but very slow "
                        "on Mac MPS (~20 s per crop). Off by default.")
    args = p.parse_args()

    pipe = HairBlurPipeline(
        sapiens_size=args.sapiens_size,
        prefer_segmenter=args.segmenter,
        prefer_matter=args.matter,
        device=args.device,
        feather_radius=args.feather_radius,
        sam3_version=args.sam3_version,
        smart_crop=args.smart_crop,
        body_backend=args.body_backend,
        sam3_backend=args.sam3_backend,
    )
    image = Image.open(args.input).convert("RGB")
    result = pipe(image, blur_radius=args.blur_radius,
                  do_clean=not args.no_clean,
                  check_woman=args.check_woman)

    result.output.save(args.output)
    if args.save_mask:
        result.mask.save(args.save_mask)
    if args.save_alpha:
        result.alpha.save(args.save_alpha)

    woman = "n/a" if result.woman_check is None else f"{result.woman_check:.2%}"
    print(f"segmenter={result.segmenter} matter={result.matter} "
          f"coverage={result.coverage:.2%} woman={woman}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
