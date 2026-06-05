<#
.SYNOPSIS
    Background NVMe temperature monitor.
    Writes drive temperature to a JSON file that the pipeline reads.

.DESCRIPTION
    Run this in an Admin PowerShell window.  It polls the KLEVV drive
    temperature every N seconds and writes to %TEMP%\disk_temp_monitor.json.
    Stop with Ctrl+C.

    Usage:
        powershell -ExecutionPolicy Bypass -File scripts\monitor_disk_temp.ps1

.PARAMETER DriveName
    Substring to match against the drive's FriendlyName (default: KLEVV).

.PARAMETER IntervalSeconds
    Polling interval (default: 10).

.PARAMETER OutputFile
    Path to the output JSON file.
#>

param(
    [string]$DriveName = "KLEVV",
    [int]$IntervalSeconds = 10,
    [string]$OutputFile = "$env:TEMP\disk_temp_monitor.json"
)

Write-Host "Disk temperature monitor started -- polling every ${IntervalSeconds}s"
Write-Host "Matching drives containing: $DriveName"
Write-Host "Output: $OutputFile"
Write-Host "Press Ctrl+C to stop.`n"

while ($true) {
    try {
        $disk = Get-PhysicalDisk | Where-Object { $_.FriendlyName -like "*$DriveName*" }

        if (-not $disk) {
            Write-Warning "No drive matching '$DriveName' found"
            $data = @{
                timestamp   = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                temperature = $null
                error       = "Drive not found"
            }
        } else {
            $rel = $disk | Get-StorageReliabilityCounter
            $temp = $rel.Temperature

            if ($temp -gt 0) {
                Write-Host "$(Get-Date -Format 'HH:mm:ss')  $($disk.FriendlyName): ${temp}C"
            }

            $data = @{
                timestamp       = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
                temperature     = $temp
                temperature_max = $rel.TemperatureMax
                wear            = $rel.Wear
                drive_name      = $disk.FriendlyName
            }
        }

        $data | ConvertTo-Json | Set-Content -Path $OutputFile -Force
    } catch {
        $errMsg = $_.Exception.Message
        $errorData = @{
            timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            error     = $errMsg
        }
        $errorData | ConvertTo-Json | Set-Content -Path $OutputFile -Force
        Write-Warning "Monitor error: $errMsg"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
