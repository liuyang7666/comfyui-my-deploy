param(
    [Parameter(Mandatory=$true)][string]$ComfyUI,
    [string]$Python = "python",
    [switch]$DryRun,
    [switch]$InstallDeps
)
$ErrorActionPreference = 'Stop'
$deployArgs = @((Join-Path $PSScriptRoot 'scripts\deploy.py'), '--comfyui', $ComfyUI)
if ($DryRun) { $deployArgs += '--dry-run' }
if ($InstallDeps) { $deployArgs += '--install-deps' }
& $Python @deployArgs
if ($LASTEXITCODE -ne 0) { throw '部署未完成，请查看上方错误。' }
