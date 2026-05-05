"""Benchmark Sapiens-1b vs SegFormer-B2-ATR on test/input.jpg."""
import time
from pathlib import Path

from PIL import Image

from pipeline import (
    FeatherMatter,
    SapiensSegmenter,
    SegformerATRSegmenter,
    composite_blur,
)

ROOT = Path(__file__).parent
IN_PATH = ROOT / "test" / "input.jpg"
OUT_DIR = ROOT / "test"


def bench(seg, image, n=3):
    seg.segment(image)  # warm-up
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        mask = seg.segment(image)
        times.append(time.perf_counter() - t0)
    return mask, min(times)


def main():
    image = Image.open(IN_PATH).convert("RGB")
    print(f"input: {IN_PATH} ({image.size})")

    matter = FeatherMatter(radius=4.0)

    print("loading sapiens-1b…")
    sap = SapiensSegmenter(size="1b")
    print(f"  device={sap.device}")
    mask_sap, t_sap = bench(sap, image)
    Image.fromarray(mask_sap).save(OUT_DIR / "bench_sapiens_mask.png")
    composite_blur(image, matter.matte(image, mask_sap), 25.0).save(
        OUT_DIR / "bench_sapiens_out.jpg"
    )

    print("loading segformer-atr…")
    atr = SegformerATRSegmenter()
    print(f"  device={atr.device}")
    mask_atr, t_atr = bench(atr, image)
    Image.fromarray(mask_atr).save(OUT_DIR / "bench_atr_mask.png")
    composite_blur(image, matter.matte(image, mask_atr), 25.0).save(
        OUT_DIR / "bench_atr_out.jpg"
    )

    cov_sap = (mask_sap > 127).mean()
    cov_atr = (mask_atr > 127).mean()

    print()
    print(f"Sapiens-1b   ({sap.name})")
    print(f"  time:     {t_sap * 1000:7.1f} ms")
    print(f"  coverage: {cov_sap:.2%}")
    print(f"SegFormer-B2-ATR ({atr.name})")
    print(f"  time:     {t_atr * 1000:7.1f} ms")
    print(f"  coverage: {cov_atr:.2%}")
    print()
    print(f"Speedup: {t_sap / t_atr:.2f}×")
    print(f"Outputs in: {OUT_DIR}/bench_*")


if __name__ == "__main__":
    main()
