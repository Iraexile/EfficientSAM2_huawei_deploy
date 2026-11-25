$dg = Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\Microsoft\Windows\DeviceGuard -ErrorAction SilentlyContinue
if ($dg) {
  "VBS status: $($dg.VirtualizationBasedSecurityStatus)"
  "CI policy enforcement: $($dg.CodeIntegrityPolicyEnforcementStatus)"
  "UMCI enforcement: $($dg.UsermodeCodeIntegrityPolicyEnforcementStatus)"
} else { "Win32_DeviceGuard not available" }
$mp = Get-MpComputerStatus -ErrorAction SilentlyContinue
if ($mp) { "SmartAppControlState: $($mp.SmartAppControlState)" }
$sig = Get-AuthenticodeSignature "C:\path\to\EfficientSAM2\pc\tools\mindspore-lite-2.11.0-win-x64\tools\converter\converter\converter_lite.exe"
"converter_lite.exe signature: $($sig.Status) / $($sig.SignerCertificate.Subject)"
