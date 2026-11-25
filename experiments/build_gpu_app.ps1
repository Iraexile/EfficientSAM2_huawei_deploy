param([string]$Studio='D:/DevEco Studio')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
foreach($arch in @('arm64-v8a','x86_64')) {
    $pattern=if($arch -eq 'arm64-v8a'){'aarch64'}else{'x86_64'}
    $lib=Get-ChildItem "$Studio/sdk/default/openharmony/native/llvm" -Recurse -Filter libomp.so | Where-Object FullName -Match $pattern | Select-Object -First 1
    if(!$lib){throw "Missing SDK libomp for $arch"}
    New-Item -ItemType Directory -Force "$root/entry/libs/$arch" | Out-Null
    Copy-Item $lib.FullName "$root/entry/libs/$arch/libomp.so"
}
$env:DEVECO_SDK_HOME="$Studio/sdk"
$env:PATH="$Studio/sdk/default/openharmony/previewer/common/bin;$Studio/jbr/bin;$Studio/tools/node;"+$env:PATH
Push-Location $root
try {
    & "$Studio/tools/ohpm/bin/ohpm.bat" install
    if($LASTEXITCODE){throw 'ohpm install failed'}
    & "$Studio/tools/hvigor/bin/hvigorw.bat" assembleHap --mode module -p product=default --no-daemon
    if($LASTEXITCODE){throw 'GPU HAP build failed'}
} finally {Pop-Location}
