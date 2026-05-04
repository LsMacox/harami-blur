"""CLI entrypoint over pipeline.HairBlurPipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pipeline import HairBlurPipeline, SAPIENS_CHECKPOINTS


def main() -> None:
    p = argparse.ArgumentParser(description="Hair-aware blur (Sapiens + MatAnyone, with fallbacks).")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--blur-radius", type=float, default=25.0,
                   help="Gaussian blur radius applied to the hair region")
    p.add_argument("--sapiens-size", default="2b",
                   choices=list(SAPIENS_CHECKPOINTS.keys()),
                   help="Sapiens-seg variant; bigger = more accurate, more VRAM")
    p.add_argument("--segmenter", default="sam3",
                   choices=["sam3", "sapiens", "segformer"],
                   help="Force a segmenter; falls back automatically on load failure")
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
    args = p.parse_args()

    pipe = HairBlurPipeline(
        sapiens_size=args.sapiens_size,
        prefer_segmenter=args.segmenter,
        prefer_matter=args.matter,
        device=args.device,
        feather_radius=args.feather_radius,
        sam3_version=args.sam3_version,
    )
    image = Image.open(args.input).convert("RGB")
    result = pipe(image, blur_radius=args.blur_radius, do_clean=not args.no_clean)

    result.output.save(args.output)
    if args.save_mask:
        result.mask.save(args.save_mask)
    if args.save_alpha:
        result.alpha.save(args.save_alpha)

    print(f"segmenter={result.segmenter} matter={result.matter} "
          f"coverage={result.coverage:.2%}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
