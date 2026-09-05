[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

Write-Output "=== Windows ==="
[Environment]::OSVersion.VersionString
$PSVersionTable.PSVersion.ToString()
git --version
codex --version

Write-Output "=== Windows GPU visibility ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader,nounits

Write-Output "=== WSL platform ==="
wsl --version
wsl --status
wsl --list --verbose

Write-Output "=== WSL detailed inventory ==="
wsl -d Ubuntu-22.04 -e bash -lc 'bash scripts/preflight.sh'
