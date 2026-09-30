param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Start', 'Stop')]
    [string] $Action
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$entryPoint = Join-Path $projectRoot 'dist\src\index.js'
# The launcher passes an absolute entry point so other Node.js apps cannot match.
$commandPattern = '^(?:"[^"]*"|\S+)\s+"?' + [regex]::Escape($entryPoint) + '"?\s*$'

function Get-BotProcess {
    Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" |
        Where-Object { $_.CommandLine -match $commandPattern }
}

$launchMutex = $null
$ownsMutex = $false
try {
    if ($Action -eq 'Stop') {
        $botProcesses = @(Get-BotProcess)
        if ($botProcesses.Count -eq 0) {
            Write-Host 'The bot is not running.'
            exit 0
        }
        foreach ($botProcess in $botProcesses) {
            # Kill the whole bot tree, including yt-dlp and FFmpeg.
            & "$env:SystemRoot\System32\taskkill.exe" /PID $botProcess.ProcessId /T /F
            if ($LASTEXITCODE -ne 0) { throw 'Failed to stop the bot process tree.' }
        }
        Write-Host 'Bot stopped.'
        exit 0
    }

    # Hold a project-specific mutex through setup and playback to prevent duplicates.
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $rootBytes = [System.Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant())
        $rootHash = [BitConverter]::ToString($hasher.ComputeHash($rootBytes)).Replace('-', '')
    } finally {
        $hasher.Dispose()
    }
    $launchMutex = [System.Threading.Mutex]::new($false, "Local\muz-bot-ds-$rootHash")
    try { $ownsMutex = $launchMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex -or @(Get-BotProcess).Count -gt 0) {
        Write-Host 'The bot is already running or starting.'
        exit 0
    }

    Set-Location -LiteralPath $projectRoot
    $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $nodeCommand -or -not $npmCommand) {
        throw 'Install Node.js 24.17+ with npm, then run start-bot.bat again.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot '.env'))) {
        throw 'Create .env from .env.example and set DISCORD_TOKEN and CLIENT_ID first.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'node_modules'))) {
        Write-Host 'Installing dependencies...'
        & $npmCommand.Source ci
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
    Write-Host 'Building the bot...'
    & $npmCommand.Source run build
    if ($LASTEXITCODE -ne 0) { throw 'Build failed. The bot was not started.' }

    Write-Host 'Starting the bot. Press Ctrl+C for a graceful shutdown, or use stop-bot.bat.'
    & $nodeCommand.Source $entryPoint
    exit $LASTEXITCODE
} catch {
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($ownsMutex) { $launchMutex.ReleaseMutex() }
    if ($launchMutex) { $launchMutex.Dispose() }
}
