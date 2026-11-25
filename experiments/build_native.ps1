param([string]$SdkNative='D:/DevEco Studio/sdk/default/openharmony/native')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$cmake="$SdkNative/build-tools/cmake/bin/cmake.exe"
$build="$PSScriptRoot/artifacts/native-build"
& $cmake -S "$PSScriptRoot/native" -B $build -G Ninja '-DCMAKE_MAKE_PROGRAM=D:/mingw64/bin/ninja.exe' "-DCMAKE_TOOLCHAIN_FILE=$SdkNative/build/cmake/ohos.toolchain.cmake" -DOHOS_ARCH=arm64-v8a "-Dncnn_DIR=$root/dependencies/ncnn/ncnn-20261020-harmonyos-vulkan/arm64-v8a/lib/cmake/ncnn" -DCMAKE_BUILD_TYPE=Release 2>&1 | Tee-Object "$PSScriptRoot/logs/native-configure.txt"
if($LASTEXITCODE) {throw 'CMake configure failed'}
& $cmake --build $build 2>&1 | Tee-Object "$PSScriptRoot/logs/native-build.txt"
if($LASTEXITCODE) {throw 'Native build failed'}
$bundle="$PSScriptRoot/artifacts/device"
New-Item -ItemType Directory -Force $bundle | Out-Null
Copy-Item "$build/vulkan_probe","$build/nnrt_probe","$build/ncnn_case" $bundle
Copy-Item "$SdkNative/llvm/lib/aarch64-linux-ohos/libc++_shared.so" $bundle
$omp=Get-ChildItem "$SdkNative/llvm" -Filter libomp.so -Recurse | Where-Object FullName -Match 'aarch64' | Select-Object -First 1
if(!$omp){throw 'arm64 libomp.so not found'}
Copy-Item $omp.FullName $bundle
Get-ChildItem $bundle -File | Get-FileHash | Select-Object @{n='file';e={Split-Path $_.Path -Leaf}},Hash | ConvertTo-Json | Set-Content "$PSScriptRoot/logs/device-bundle-sha256.json"
