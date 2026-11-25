param(
    [ValidateSet('gpu','nnrt','memattn')][string]$Stage='gpu',
    [string]$Target='',
    [string]$Hdc='D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe'
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
if(!$Target) {
    $targets=@(& $Hdc list targets | Where-Object {$_ -and $_ -notmatch 'Empty|\[|\]'})
    if($targets.Count -ne 1) {throw 'Connect exactly one authorized device, or supply -Target serial.'}
    $Target=$targets[0].Trim()
}
$remote='/data/local/tmp/sam2_probe_20261124'
$logs="$PSScriptRoot/logs/device-$(Get-Date -Format yyyyMMdd-HHmmss)-$Stage"
New-Item -ItemType Directory -Force $logs | Out-Null
function HdcCall([string[]]$Arguments) {
    & $Hdc -t $Target @Arguments
    if($LASTEXITCODE){throw "hdc failed: $Arguments"}
}
function Push([string]$Local,[string]$Destination) {HdcCall -Arguments @('file','send',$Local,$Destination)}
HdcCall -Arguments @('shell',"mkdir -p $remote/out")
foreach($file in Get-ChildItem "$PSScriptRoot/artifacts/device" -File) {Push $file.FullName "$remote/$($file.Name)"}
HdcCall -Arguments @('shell',"chmod 755 $remote/vulkan_probe $remote/nnrt_probe $remote/ncnn_case")
HdcCall -Arguments @('shell','uname -a') | Set-Content "$logs/uname.txt"
HdcCall -Arguments @('shell','param get const.product.software.version') | Set-Content "$logs/system-version.txt"
function RunProbe([string]$Name,[string]$Command) {
    $remoteResult=HdcCall -Arguments @('shell',"cd $remote && LD_LIBRARY_PATH=$remote $Command > $remote/out/$Name.txt 2>&1; echo REMOTE_EXIT=`$?")
    $remoteResult | Set-Content "$logs/$Name-exit.txt"
    HdcCall -Arguments @('file','recv',"$remote/out/$Name.txt","$logs/$Name.txt")
    Get-Content "$logs/$Name.txt"
    if($Stage -eq 'gpu' -and "$remoteResult" -notmatch 'REMOTE_EXIT=0\s*$') {
        throw "GPU shell probe did not succeed: $remoteResult. See GPU_DEVICE_RESULT.md for the signed GpuProbeAbility path."
    }
}
if($Stage -eq 'gpu') {
    RunProbe 'vulkan' './vulkan_probe'
} elseif($Stage -eq 'nnrt') {
    foreach($name in @('matmul','attention')) {
        Push "$PSScriptRoot/artifacts/models/$name.ms" "$remote/$name.ms"
        Push "$PSScriptRoot/artifacts/models/$($name)_case" "$remote/$($name)_case"
        foreach($mode in @('cpu','nnrt','accelerator')) {
            RunProbe "$name-$mode" "./nnrt_probe $mode $name.ms $($name)_case out/$name-$mode"
        }
    }
    foreach($model in Get-ChildItem "$root/pc/outputs/ms_opprobe" -Filter '*.ms') {
        Push $model.FullName "$remote/$($model.Name)"
        foreach($mode in @('cpu','nnrt','accelerator')) {RunProbe "$($model.BaseName)-$mode" "./nnrt_probe $mode $($model.Name)"}
    }
    HdcCall -Arguments @('shell','hilog -x') | Set-Content "$logs/hilog.txt"
} else {
    throw 'Legacy Net execution cannot prove absence of CPU fallback. Use experiments/run_k01_strict.ps1 and MemattnK01Ability for k01 validation.'
}
# Keep all outputs, including predictions, to enable offline parity/mask analysis.
HdcCall -Arguments @('file','recv',"$remote/out","$logs/raw-out")
Write-Output "Saved logs: $logs; review REMOTE_EXIT and hardware_execution fields."


