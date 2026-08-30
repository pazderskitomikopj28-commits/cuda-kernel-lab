#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
op="${1:-reduce}"
rows="${ROWS:-4096}"
cols="${COLS:-4096}"
iters="${ITERS:-100}"
k="${K:-256}"
work="${WORK:-64}"
"${build_dir}/kernel_bench" --op "${op}" --rows "${rows}" --cols "${cols}" \
  --k "${k}" --work "${work}" --iters "${iters}"
