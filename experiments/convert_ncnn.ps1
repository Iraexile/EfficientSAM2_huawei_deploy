$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$pnnx="$root/dependencies/pnnx/pnnx-20261020-windows/pnnx.exe"
$out="$PSScriptRoot/artifacts/models"
New-Item -ItemType Directory -Force $out | Out-Null
foreach($job in @(
    @{name='matmul';input="$out/matmul.onnx";shape='[16,16],[16,16]'},
    @{name='attention';input="$out/attention.onnx";shape='[16,16],[16,16],[16,16]'},
    @{name='memattn_k01';input="$root/pc/outputs/onnx/memattn_k01.onnx";shape='[4096,1,256],[4096,1,256],[4100,1,64],[4100,1,64]'},
    @{name='memattn_k01_fixed';input="$out/memattn_k01.repaired.onnx";shape='[4096,1,256],[4096,1,256],[4100,1,64],[4100,1,64]'}
)) {
    $prefix="$out/$($job.name)"
    $staged="$out/$($job.name).source.onnx"
    Copy-Item $job.input $staged -Force
    $arguments=@($staged,"inputshape=$($job.shape)",'fp16=0',"pnnxparam=$prefix.pnnx.param","pnnxbin=$prefix.pnnx.bin","pnnxpy=$prefix`_pnnx.py","pnnxonnx=$prefix.pnnx.onnx","ncnnparam=$prefix.ncnn.param","ncnnbin=$prefix.ncnn.bin","ncnnpy=$prefix`_ncnn.py")
    & $pnnx @arguments 2>&1 | Tee-Object "$PSScriptRoot/logs/pnnx-$($job.name).txt"
    $code=$LASTEXITCODE
    "exit_code=$code" | Add-Content "$PSScriptRoot/logs/pnnx-$($job.name).txt"
    if($code) {Write-Warning "$($job.name) conversion failed: $code; inspect raw log"}
    if(Test-Path "$prefix.ncnn.param") {
        $unsupported=Select-String -Path "$prefix.ncnn.param" -Pattern '^ScatterND\s|^onnx\.|^pnnx\.'
        if($unsupported) {Write-Warning "$($job.name): unresolved operators; exit 0 is NOT conversion acceptance"}
    }
}

