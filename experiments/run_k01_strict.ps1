param([switch]$Prepare,[switch]$Build,[switch]$Install,[switch]$Run,
      [string]$Hdc='D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    if($Prepare) {
        ./experiments/run_python.ps1 experiments/prepare_k01_fullrefs.py
        New-Item -ItemType Directory -Force entry/src/main/resources/rawfile | Out-Null
        Get-ChildItem experiments/artifacts/k01-strict -File | Where-Object Name -NotMatch 'refs-full|k01-full-ready' | Copy-Item -Destination entry/src/main/resources/rawfile
        ./experiments/run_python.ps1 experiments/package_k01_fullrefs.py
    }
    if($Build) {./experiments/build_gpu_app.ps1}
    if($Install) {
        & $Hdc install -r entry/build/default/outputs/default/entry-default-signed.hap
        if($LASTEXITCODE){throw 'Install failed'}
    }
    if($Run) {
        & $Hdc shell aa start -a MemattnK01Ability -b com.example.efficientsam2
        if($LASTEXITCODE){throw 'Start failed: inspect screen lock and device authorization'}
    }
} finally {Pop-Location}
