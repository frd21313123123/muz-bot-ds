param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$entryPath = Join-Path $projectPath 'dist\src\index.js'
$runtimePath = Join-Path $projectPath '.runtime'
$pidPath = Join-Path $runtimePath 'bot.pid'
$stdoutPath = Join-Path $runtimePath 'bot.stdout.log'
$stderrPath = Join-Path $runtimePath 'bot.stderr.log'
# Match the full entry argument, never an unrelated process with a reused PID.
$entryPattern = '(?:^|\s)(?:"' + [regex]::Escape($entryPath) + '"|' + [regex]::Escape($entryPath) + ')(?=\s|$)'

function Test-BotProcess($process) {
    return $null -ne $process -and $process.Name -eq 'node.exe' -and $process.CommandLine -match $entryPattern
}

function Get-BotProcesses {
    if (Test-Path -LiteralPath $pidPath) {
        $savedProcessId = 0
        if ([int]::TryParse((Get-Content -LiteralPath $pidPath -Raw).Trim(), [ref]$savedProcessId) -and $savedProcessId -gt 0) {
            $savedProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $savedProcessId"
            if (Test-BotProcess $savedProcess) { return $savedProcess }
        }
    }
    Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" | Where-Object { Test-BotProcess $_ }
}

$hasher = [System.Security.Cryptography.SHA256]::Create()
try { $projectHash = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($projectPath.ToLowerInvariant()))).Replace('-', '') }
finally { $hasher.Dispose() }
$mutex = New-Object System.Threading.Mutex($false, "Local\muz-bot-$projectHash")
$lockTaken = $false
$exitCode = 0
try {
    try { $lockTaken = $mutex.WaitOne(10000) }
    catch [System.Threading.AbandonedMutexException] { $lockTaken = $true }
    if (-not $lockTaken) {
        throw 'Another start/stop operation is already running. Try again shortly.'
    } else {
        $bots = @(Get-BotProcesses)
        if ($Action -eq 'Stop') {
            foreach ($bot in $bots) {
                # Recheck identity immediately before terminating this process tree.
                $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($bot.ProcessId)"
                if (-not (Test-BotProcess $current) -or $current.CreationDate -ne $bot.CreationDate) { continue }
                & taskkill.exe /PID $bot.ProcessId /T /F | Out-Null
                if ($LASTEXITCODE -ne 0 -and (Test-BotProcess (Get-CimInstance Win32_Process -Filter "ProcessId = $($bot.ProcessId)"))) {
                    throw "Could not stop bot PID $($bot.ProcessId)."
                }
            }
            if (Test-Path -LiteralPath $pidPath) { Remove-Item -LiteralPath $pidPath }
            if ($bots.Count -gt 0) { Write-Host 'Bot stopped. Its Python, yt-dlp and FFmpeg child processes were stopped as well.' }
            else { Write-Host 'Bot is not running.' }
        } elseif ($bots.Count -gt 0) {
            Write-Host "Bot is already running (PID $($bots[0].ProcessId))."
        } elseif ($Action -eq 'Status') {
            Write-Host 'Bot is not running.'
        } else {
            if (-not (Test-Path -LiteralPath (Join-Path $projectPath '.env'))) { throw 'Create .env with DISCORD_TOKEN and CLIENT_ID first.' }
            if (-not (Test-Path -LiteralPath (Join-Path $projectPath 'node_modules'))) { throw 'Run npm install first.' }
            $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
            $npmPath = (Get-Command npm.cmd -ErrorAction Stop).Source
            Push-Location -LiteralPath $projectPath
            try {
                Write-Host 'Building bot...'
                & $npmPath run build
                if ($LASTEXITCODE -ne 0) { throw 'TypeScript build failed; bot was not started.' }
            } finally { Pop-Location }
            New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
            foreach ($logPath in @($stdoutPath, $stderrPath)) {
                if (Test-Path -LiteralPath $logPath) { Copy-Item -LiteralPath $logPath -Destination "$logPath.previous" -Force }
            }
            $process = Start-Process -FilePath $nodePath -ArgumentList @('"' + $entryPath + '"') -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
            Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ASCII
            # Startup can take time; allow stop-bot.bat to cancel it meanwhile.
            $mutex.ReleaseMutex()
            $lockTaken = $false
            Write-Host "Bot started in background (PID $($process.Id)). Waiting for Discord..."
            $ready = $false
            $deadline = [DateTime]::UtcNow.AddSeconds(90)
            do {
                $process.Refresh()
                if ($process.HasExited) { throw "Bot exited during startup. See $stderrPath" }
                if (Test-Path -LiteralPath $stdoutPath) {
                    # Get-Content permits the writer to keep its file open.
                    # Empty startup files yield null; -match safely treats it as false.
                    $startupLog = Get-Content -LiteralPath $stdoutPath -Raw -Encoding UTF8
                    $ready = $startupLog -match '\[Bot\] Ready'
                }
                if (-not $ready) { Start-Sleep -Milliseconds 500 }
            } while (-not $ready -and [DateTime]::UtcNow -lt $deadline)
            if ($ready) { Write-Host 'Bot connected to Discord.' }
            else { Write-Host "Bot is still initializing. Check $stdoutPath" }
            Write-Host "Logs: $stdoutPath"
            Write-Host 'Use stop-bot.bat to stop the bot.'
        }
    }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($lockTaken) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
exit $exitCode
