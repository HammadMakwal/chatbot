$logdir = Join-Path $PSScriptRoot 'logs'
if (-not (Test-Path $logdir)) {
  Write-Host "Logs directory not found: $logdir"
  exit 1
}
$latest = Get-ChildItem -Path $logdir -Filter '*.log' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $latest) {
  Write-Host "No log files found in $logdir"
  exit 1
}
notepad $latest.FullName
