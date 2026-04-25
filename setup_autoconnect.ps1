# QuantPipe — One-time VPN auto-connect setup
# RIGHT-CLICK this file and choose "Run as administrator"
# After this runs, WireGuard connects automatically on every login. No further action needed.

$ErrorActionPreference = "Stop"
$WG_EXE  = "C:\Program Files\WireGuard\wireguard.exe"
$WG_CONF = "C:\Users\micha\.wireguard\quantpipe.conf"
$TASK    = "QuantPipe-WireGuard-AutoConnect"

Write-Host "Setting up QuantPipe VPN auto-connect..." -ForegroundColor Cyan

# 1. Install the tunnel as a Windows service (survives reboots, starts before login)
Write-Host "[1/3] Installing WireGuard tunnel service..." -ForegroundColor Yellow
& $WG_EXE /installtunnelservice $WG_CONF
Start-Sleep -Seconds 3

# 2. Register scheduled task to (re-)install the tunnel at each login in case the
#    service was removed by a Windows update or WireGuard upgrade
Write-Host "[2/3] Creating scheduled task for resilient auto-connect..." -ForegroundColor Yellow
$action    = New-ScheduledTaskAction -Execute $WG_EXE -Argument "/installtunnelservice `"$WG_CONF`""
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType Interactive
Register-ScheduledTask -TaskName $TASK -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "Connects QuantPipe WireGuard VPN at login automatically" -Force | Out-Null

# 3. Verify
Write-Host "[3/3] Verifying..." -ForegroundColor Yellow
$svc  = Get-Service -Name "WireGuardTunnel*" -ErrorAction SilentlyContinue
$task = Get-ScheduledTask -TaskName $TASK -ErrorAction SilentlyContinue

Write-Host ""
if ($svc) {
    Write-Host "  Tunnel service : $($svc.Status)" -ForegroundColor Green
} else {
    Write-Host "  Tunnel service : NOT FOUND (may need a reboot)" -ForegroundColor Yellow
}
if ($task) {
    Write-Host "  Scheduled task : $($task.State)" -ForegroundColor Green
} else {
    Write-Host "  Scheduled task : NOT CREATED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Done. WireGuard will now connect automatically on every login." -ForegroundColor Green
Write-Host "Dashboard: http://10.0.0.1:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Enter to close..." -ForegroundColor Gray
Read-Host
