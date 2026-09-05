#!/usr/bin/env bash
set -uo pipefail

printf '%s\n' '=== Distribution ==='
cat /etc/os-release
printf '%s\n' '=== Kernel/Python/CPU/RAM ==='
uname -r
python3 --version
lscpu
free -h
printf '%s\n' '=== GPU ==='
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits
printf '%s\n' '=== PyTorch CUDA ==='
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())' || true
printf '%s\n' '=== Environment manager ==='
uv --version || true
