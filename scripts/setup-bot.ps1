param(
    [ValidateSet('light', 'heavy')][string]$Profile,
    [ValidateSet('cpu', 'cuda')][string]$Device,
    [switch]$CheckOnly,
    [switch]$NoDeploy
)
$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$envPath = Join-Path $projectPath '.env'
$runtimePath = Join-Path $projectPath '.runtime'

function Refresh-SetupPath {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}
function Test-SetupNode {
    if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) { return $false }
    $nodeVersion = & node.exe --version
    if ($LASTEXITCODE -ne 0) { return $false }
    return [version]($nodeVersion.Trim().TrimStart('v')) -ge [version]'24.17.0'
}
function Find-SetupPython {
    foreach ($candidate in @(@('py','-3.13'), @('py','-3.12'), @('py','-3.11'), @('python'), @('py','-3'), @('python3'))) {
        if (-not (Get-Command $candidate[0] -ErrorAction SilentlyContinue)) { continue }
        $prefix = @($candidate | Select-Object -Skip 1)
        try { & $candidate[0] @prefix -c 'import sys,venv; assert sys.version_info >= (3,11)' 2>$null }
        catch { continue }
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    return $null
}
function Install-SetupPackage([string]$Id) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw 'WinGet is missing. Install Microsoft App Installer from https://apps.microsoft.com/detail/9nblggh4nns1 and run setup-bot.bat again.'
    }
    & winget.exe install --id $Id --exact --source winget --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "WinGet could not install $Id. Install it manually and run setup again." }
    Refresh-SetupPath
}
function Get-SetupValue([string]$Key) {
    if (-not (Test-Path -LiteralPath $envPath)) { return '' }
    $content = [IO.File]::ReadAllText($envPath)
    $match = [regex]::Match($content, '(?m)^[ \t]*' + [regex]::Escape($Key) + '[ \t]*=[ \t]*(.*)$')
    if (-not $match.Success) { return '' }
    return $match.Groups[1].Value.Trim().Trim('"').Trim("'")
}
function Set-SetupValue([string]$Key, [string]$Value) {
    if ($Value -match '[\r\n]') { throw 'Multiline settings are not supported.' }
    $content = [IO.File]::ReadAllText($envPath)
    $pattern = '(?m)^[ \t]*' + [regex]::Escape($Key) + '[ \t]*=.*$'
    if ([regex]::IsMatch($content, $pattern)) {
        $replacement = "$Key=$Value"
        $content = [regex]::Replace($content, $pattern, [Text.RegularExpressions.MatchEvaluator]{ param($match) $replacement })
    } else { $content = $content.TrimEnd() + "`r`n$Key=$Value`r`n" }
    # Temporary secret-bearing files stay in the ignored runtime directory.
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    $temporary = Join-Path $runtimePath 'env.setup.tmp'
    [IO.File]::WriteAllText($temporary, $content, (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $envPath -Force
}
function Invoke-SetupNpm([string[]]$Arguments) {
    & npm.cmd @Arguments
    if ($LASTEXITCODE -ne 0) { throw "npm $($Arguments -join ' ') failed. Fix the reported error and run setup again." }
}

Push-Location -LiteralPath $projectPath
try {
    Write-Host 'muz-bot-ds setup: Windows x64, Node.js, Python, FFmpeg, voice and ready-made replies.'
    if ($CheckOnly) {
        $python = Find-SetupPython
        Write-Host "Node.js 24.17+: $(Test-SetupNode)"
        Write-Host "Python 3.11+ with venv: $($null -ne $python)"
        Write-Host "Discord credentials configured: $([bool](Get-SetupValue 'DISCORD_TOKEN') -and [bool](Get-SetupValue 'CLIENT_ID'))"
        Write-Host "Light prepared: $(Test-Path -LiteralPath (Join-Path $runtimePath 'voice-light/ready.json'))"
        Write-Host "Heavy prepared: $(Test-Path -LiteralPath (Join-Path $runtimePath 'voice/ready.json'))"
        Write-Host 'Check only; no installation or configuration changes.'
        exit 0
    }
    if (-not $Profile) {
        Write-Host '1. Light (recommended): CPU, one small Whisper, commands parsed by rules.'
        Write-Host '2. Heavy: larger Whisper + Laya, optional NVIDIA CUDA.'
        $choice = Read-Host 'Select profile [1]'
        if ($choice -eq '' -or $choice -eq '1') { $Profile = 'light' }
        elseif ($choice -eq '2') { $Profile = 'heavy' }
        else { throw 'Select 1 or 2, or use -Profile light/heavy.' }
    }
    if ($Profile -eq 'light') { $Device = 'cpu' }
    elseif (-not $Device) {
        $choice = Read-Host 'Heavy device: 1 = CPU, 2 = NVIDIA CUDA [1]'
        if ($choice -eq '' -or $choice -eq '1') { $Device = 'cpu' }
        elseif ($choice -eq '2') { $Device = 'cuda' }
        else { throw 'Select 1 or 2, or use -Device cpu/cuda.' }
    }
    if (-not (Test-SetupNode)) { Install-SetupPackage 'OpenJS.NodeJS.LTS' }
    if (-not (Test-SetupNode)) { throw 'Node.js 24.17+ is required. Reopen setup after installing Node.js.' }
    if (-not (Find-SetupPython)) { Install-SetupPackage 'Python.Python.3.13' }
    if (-not (Find-SetupPython)) { throw 'Python 3.11+ with venv is required. Reopen setup after installing Python.' }
    & (Join-Path $PSScriptRoot 'bot.ps1') -Action Stop
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the running bot for setup.' }
    if (-not (Test-Path -LiteralPath $envPath)) { Copy-Item -LiteralPath (Join-Path $projectPath '.env.example') -Destination $envPath }
    Invoke-SetupNpm -Arguments @('ci')
    # Overrides apply only to this installer process until preparation succeeds.
    $env:VOICE_PROFILE = $Profile; $env:VOICE_DEVICE = $Device; $env:VOICE_TTS_ENGINE = 'bundled'
    Invoke-SetupNpm -Arguments @('run','build')
    Invoke-SetupNpm -Arguments @('run','setup:voice')
    Invoke-SetupNpm -Arguments @('run','setup:tts')
    Invoke-SetupNpm -Arguments @('run','update:ytdlp')
    Set-SetupValue 'VOICE_PROFILE' $Profile
    if ($Profile -eq 'heavy') { Set-SetupValue 'VOICE_DEVICE' $Device }
    Set-SetupValue 'VOICE_TTS_ENGINE' 'bundled'
    if (-not $NoDeploy) {
        if (-not (Get-SetupValue 'DISCORD_TOKEN')) {
            $secureToken = Read-Host 'Discord bot token (hidden; saved only in .env)' -AsSecureString
            $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
            try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer) }
            finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer) }
            if ([string]::IsNullOrWhiteSpace($token)) { throw 'Discord token is required, or use -NoDeploy.' }
            Set-SetupValue 'DISCORD_TOKEN' $token.Trim()
            $token = $null
        }
        if (-not (Get-SetupValue 'CLIENT_ID')) {
            $applicationId = Read-Host 'Discord application ID'
            if ($applicationId -notmatch '^\d{17,22}$') { throw 'Invalid Discord application ID.' }
            Set-SetupValue 'CLIENT_ID' $applicationId
        }
        Invoke-SetupNpm -Arguments @('run','deploy')
    }
    Write-Host "Setup completed. Selected profile: $Profile ($Device). Run start-bot.bat."
    Write-Host 'Change profile by rerunning setup-bot.bat. Existing heavy models and settings are preserved.'
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
    exit 1
} finally { Pop-Location }
