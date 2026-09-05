#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

uv venv .venv-serve --python 3.12 --seed
uv pip install --python .venv-serve/bin/python -r requirements-serve.in --torch-backend=auto
uv pip install --python .venv-serve/bin/python --no-deps --editable .
uv pip compile requirements-serve.in \
  --python .venv-serve/bin/python \
  --torch-backend=auto \
  --output-file requirements-serve.lock
.venv-serve/bin/python -m pip freeze > artifacts/manifests/serve_environment.txt
.venv-serve/bin/python -c \
  'import json, torch, vllm; print(json.dumps({"torch": torch.__version__, "vllm": vllm.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}))'
.venv-serve/bin/python scripts/capture_serve_environment.py
