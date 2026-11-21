$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installation failed; install it from https://docs.astral.sh/uv/ and rerun"
    }
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    Invoke-Checked uv @("venv", "--python", "3.12", ".venv")
}

Invoke-Checked uv @("pip", "install", "--python", $Python, "--no-deps", "-r", "requirements.txt")
Invoke-Checked $Python @("download_weights.py")
Invoke-Checked $Python @("desktop.py")
