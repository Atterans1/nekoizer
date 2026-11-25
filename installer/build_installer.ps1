# Build the single setup .exe: build one MSI from dist\Nekoizer, then chain it in a Burn bundle.
#   .\installer\build_installer.ps1 -BaseUrl https://atterans.shop/downloads [-Version 1.0.0]
# Upload dist_msi\Nekoizer-<Version>.msi to BaseUrl. The small Nekoizer-Setup exe downloads it from there.
# Nothing is deleted: a previous MSI in dist_msi with another version stays where it is.
param([Parameter(Mandatory = $true)][string]$BaseUrl, [string]$Version = "1.0.0", [string]$SourceDir = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wix = Join-Path $env:USERPROFILE ".dotnet\tools\wix.exe"
$out = Join-Path $root "dist_msi"
if (-not $SourceDir) { $SourceDir = Join-Path $root "dist\Nekoizer" }
New-Item -ItemType Directory -Force $out | Out-Null
$msi = Join-Path $out "Nekoizer-$Version.msi"
& $wix build -arch x64 -d Version=$Version -d UpgradeCode=6C1B6E0A-5E1A-4D7B-9B0E-2C8F5A7D41E3 -d "SourceDir=$SourceDir" -d "InstallerDir=$root\installer" -o $msi "$root\installer\product.wxs"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $wix build -arch x64 -ext WixToolset.BootstrapperApplications.wixext -d Version=$Version -d BaseUrl=$BaseUrl -d "InstallerDir=$root\installer" -d "MsiDir=$out" -o "$out\Nekoizer-Setup-$Version.exe" "$root\installer\bundle.wxs"
exit $LASTEXITCODE
