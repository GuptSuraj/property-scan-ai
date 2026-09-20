#!/usr/bin/env bash
set -euo pipefail
python_bin="${PYTHON_BIN:-python3.11}"
if ! command -v "$python_bin" >/dev/null 2>&1; then echo "Python 3.11 is required. Install it with: brew install python@3.11"; exit 1; fi
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
for command in ffmpeg ffprobe colmap; do
  if ! command -v "$command" >/dev/null 2>&1; then echo "Missing $command. Install system tools with: brew install ffmpeg colmap"; fi
done
if [[ ! -d .venv ]]; then "$python_bin" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [[ "${1:-}" == "--download-models" ]]; then
  .venv/bin/python scripts/download_models.py --photo --video --openings --damage
fi
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Cache models with: python scripts/download_models.py --photo --video --openings --damage"
