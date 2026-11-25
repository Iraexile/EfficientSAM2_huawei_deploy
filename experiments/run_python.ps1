param([Parameter(Mandatory=$true)][string]$Script)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
# Use bundled CPython 3.12 with existing same-version wheel installations;
# do not edit the broken historical venv or global Python configuration.
$python=Get-Item 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1123-915-0_26.1123.915.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe' -ErrorAction SilentlyContinue
if (!$python) { throw 'Bundled Python missing: supply CPython 3.12 and numpy/onnx/onnxruntime.' }
$previous=$env:PYTHONPATH
try {
    $env:PYTHONPATH=Join-Path $root 'pc/.venv/Lib/site-packages'
    & $python.FullName $Script
    if ($LASTEXITCODE) { throw "Python failed: $LASTEXITCODE" }
} finally { $env:PYTHONPATH=$previous }

