$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security
$credentialDir = Join-Path $env:LOCALAPPDATA 'ComfyUI-QwenAPI'
New-Item -ItemType Directory -Force -Path $credentialDir | Out-Null
$data = @{}
$credentialFile = Join-Path $credentialDir 'credentials.dpapi'
if (Test-Path -LiteralPath $credentialFile) {
    $bytes = [System.Security.Cryptography.ProtectedData]::Unprotect([IO.File]::ReadAllBytes($credentialFile), $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
    $old = [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json
    foreach ($p in $old.PSObject.Properties) { $data[$p.Name] = $p.Value }
}
function Read-SecretValue($label) {
    $secure = Read-Host $label -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}
$value = Read-SecretValue '千问 API Key（回车保留原值）'
if ($value) { $data['api_key'] = $value.Trim() }
if ((Read-Host '配置真实账户余额查询？输入 y，其余跳过') -eq 'y') {
    $data['access_key_id'] = (Read-Host '阿里云 AccessKey ID').Trim()
    $data['access_key_secret'] = Read-SecretValue '阿里云 AccessKey Secret'
}
$bytes = [Text.Encoding]::UTF8.GetBytes(($data | ConvertTo-Json -Compress))
$encrypted = [System.Security.Cryptography.ProtectedData]::Protect($bytes, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser)
[IO.File]::WriteAllBytes($credentialFile, $encrypted)
Write-Host '已加密保存。密钥不写入工作流；仅当前 Windows 用户可解密。'
