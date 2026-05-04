#!/usr/bin/env bash
# Apply the macOS / CPU compatibility patches to a fresh checkout of
# facebookresearch/sam3. Run from inside the cloned sam3 repo:
#
#     git clone https://github.com/facebookresearch/sam3.git ~/Projects/sam3
#     cd ~/Projects/sam3
#     bash /path/to/harami-blur/patches/apply_sam3_patches.sh
#
# What the patch does (see sam3-macos-cpu.patch for full diff):
#   * sam3/model/edt.py           — make `triton` optional, add scipy CPU fallback
#   * sam3/model/position_encoding.py — precompute on CPU instead of "cuda"
#   * sam3/model/decoder.py       — same
#   * sam3/model/sam3_image_processor.py — device default = auto-detect
#   * sam3/model/vl_combiner.py   — device="cuda" defaults removed
#   * sam3/model/geometry_encoders.py — drop `pin_memory()` (CUDA-only)
#   * sam3/perflib/fused.py       — addmm_act uses bf16 only on CUDA, fp32 on CPU/MPS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$SCRIPT_DIR/sam3-macos-cpu.patch"

if [ ! -f "pyproject.toml" ] || ! grep -q '"sam3"' pyproject.toml 2>/dev/null; then
    echo "Run this script from inside the cloned facebookresearch/sam3 directory." >&2
    exit 1
fi

if [ ! -f "$PATCH" ]; then
    echo "Patch not found: $PATCH" >&2
    exit 1
fi

git apply --check "$PATCH"
git apply "$PATCH"
echo "Applied: $PATCH"
