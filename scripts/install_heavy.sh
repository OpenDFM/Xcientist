#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.pip-cache}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "missing virtualenv at $ROOT_DIR/.venv"
  echo "run ./scripts/install_base.sh first"
  exit 1
fi

ensure_pip() {
  if "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    return
  fi

  echo "pip is missing in .venv; bootstrapping with ensurepip"
  "$VENV_PYTHON" -m ensurepip --upgrade
}

memory_packages=(
  "faiss-cpu==1.12.0"
  "hdbscan==0.8.42"
  "rank-bm25==0.2.2"
)

ml_packages=(
  "accelerate==1.12.0"
  "contourpy==1.3.2"
  "cycler==0.12.1"
  "datasets==4.4.2"
  "einops==0.8.1"
  "fonttools==4.61.1"
  "fsspec==2025.10.0"
  "hf-xet==1.2.0"
  "huggingface-hub==0.36.0"
  "imageio==2.37.2"
  "joblib==1.5.3"
  "kiwisolver==1.4.9"
  "matplotlib==3.10.8"
  "multiprocess==0.70.18"
  "pandas==2.3.3"
  "peft==0.18.0"
  "pillow==12.0.0"
  "polars==1.36.1"
  "pyarrow==22.0.0"
  "pyparsing==3.3.1"
  "pytz==2025.2"
  "safetensors==0.7.0"
  "scikit-image==0.25.2"
  "scikit-learn==1.7.2"
  "scipy==1.15.3"
  "seaborn==0.13.2"
  "sentence-transformers==5.2.0"
  "sentencepiece==0.2.1"
  "shapely==2.1.2"
  "sympy==1.14.0"
  "tensorboard==2.20.0"
  "tokenizers==0.22.1"
  "torch==2.9.1"
  "torchvision==0.24.1"
  "transformers==4.57.3"
)

paper_packages=(
  "albucore==0.0.24"
  "albumentations==2.0.8"
  "av==16.0.1"
  "doclayout-yolo==0.0.4"
  "fast-langdetect==0.2.5"
  "magika==1.0.1"
  "mineru==2.7.0"
  "mineru-vl-utils==0.1.19.1"
  "modelscope==1.33.0"
  "onnxruntime==1.23.2"
  "opencv-python-headless==4.11.0.86"
  "pdfminer.six==20250506"
  "pdftext==0.6.3"
  "pypdf==6.5.0"
  "pypdfium2==4.30.0"
  "pyclipper==1.4.0"
  "qwen-vl-utils==0.0.14"
  "reportlab==4.4.7"
  "ultralytics==8.3.243"
)

usage() {
  cat <<'EOF'
Usage: ./scripts/install_heavy.sh [memory] [ml] [paper] [all]

Installs selected heavy dependency groups into the existing .venv with pip.
Examples:
  ./scripts/install_heavy.sh ml
  ./scripts/install_heavy.sh memory ml
  ./scripts/install_heavy.sh all

Pass pip mirror settings through environment variables if needed:
  PIP_INDEX_URL=...
  PIP_EXTRA_INDEX_URL=...
EOF
}

install_group() {
  local group_name="$1"
  shift
  echo "installing $group_name dependencies with pip"
  "$VENV_PYTHON" -m pip install "$@"
}

if [[ $# -eq 0 ]]; then
  set -- all
fi

declare -A selected_groups=()

for group in "$@"; do
  case "$group" in
    memory|ml|paper)
      selected_groups["$group"]=1
      ;;
    all)
      selected_groups["memory"]=1
      selected_groups["ml"]=1
      selected_groups["paper"]=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown group: $group" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ensure_pip

if [[ -n "${selected_groups[memory]:-}" ]]; then
  install_group memory "${memory_packages[@]}"
fi

if [[ -n "${selected_groups[ml]:-}" ]]; then
  install_group ml "${ml_packages[@]}"
fi

if [[ -n "${selected_groups[paper]:-}" ]]; then
  install_group paper "${paper_packages[@]}"
fi
