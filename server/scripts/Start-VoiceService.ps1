param(
 [string]$BindAddress = '127.0.0.1',
 [int]$Port = 9880,
 [string]$Config
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install requirements-service.txt in the project .venv first.' }
$taskArgs = @('-B', '-m', 'feiyu_voice_service', '--host', $BindAddress, '--port', "$Port")
if ($Config) { $taskArgs += @('--config', [IO.Path]::GetFullPath($Config)) }
Push-Location $root
try { & $python @taskArgs } finally { Pop-Location }
