#!/usr/bin/env bash

set -euo pipefail

MIP_SPLATTING_URL="https://github.com/autonomousvision/mip-splatting.git"
MIP_SPLATTING_COMMIT="dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
PACKAGE_NAME="trellis_diff_gaussian_rasterization"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -z "${CC:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc" ]]; then
        export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-cc"
    fi
    if [[ -z "${CXX:-}" && -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++" ]]; then
        export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++"
    fi
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-2}"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

git clone --filter=blob:none "${MIP_SPLATTING_URL}" "${TEMP_DIR}/mip-splatting"
git -C "${TEMP_DIR}/mip-splatting" checkout "${MIP_SPLATTING_COMMIT}"
git -C "${TEMP_DIR}/mip-splatting" submodule update --init --recursive

SOURCE_DIR="${TEMP_DIR}/mip-splatting/submodules/diff-gaussian-rasterization"
mv "${SOURCE_DIR}/diff_gaussian_rasterization" "${SOURCE_DIR}/${PACKAGE_NAME}"

"${PYTHON_BIN}" - "${SOURCE_DIR}/setup.py" "${PACKAGE_NAME}" <<'PY'
from pathlib import Path
import sys

setup_path = Path(sys.argv[1])
package_name = sys.argv[2]
content = setup_path.read_text(encoding="utf-8")
original_name = "diff_gaussian_rasterization"
if original_name not in content:
    raise RuntimeError(f"Expected package name is missing from {setup_path}")
setup_path.write_text(content.replace(original_name, package_name), encoding="utf-8")
PY

"${PYTHON_BIN}" -m pip install --no-build-isolation "${SOURCE_DIR}"
"${PYTHON_BIN}" -c "import ${PACKAGE_NAME}; print('${PACKAGE_NAME} installed')"
