$ErrorActionPreference = 'Stop'
$secret = Read-Host 'API易 API Key（输入隐藏）' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
    if (-not $value) { throw '密钥不能为空。' }
    [Environment]::SetEnvironmentVariable('APIYI_API_KEY', $value, 'User')
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $value = $null
}
Write-Host '已保存到当前用户的 APIYI_API_KEY 环境变量；请完全退出并重新打开绘世启动器/ComfyUI。密钥不写入节点或工作流。'
$secret = Read-Host '查询费用与余额的 nap_ 令牌（留空跳过）' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
try {
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer).Trim()
    if ($value) { [Environment]::SetEnvironmentVariable('APIYI_ACCOUNT_TOKEN', $value, 'User') }
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $value = $null
}
Write-Host '查询令牌需同时勾选：个人资料·读取、用量与个人记录·读取。保存后重启绘世/ComfyUI。'
