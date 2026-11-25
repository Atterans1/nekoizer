$ErrorActionPreference = 'Stop'
while (-not (Test-Path 'C:/Users/jeffr/s2s-voicechanger/build_run.done')) { Start-Sleep -Seconds 20 }
$b = Get-Content 'C:/Users/jeffr/s2s-voicechanger/build_run.done' -Raw
if ($b -notmatch 'exit=0') { "skipped: exe build failed: $b" | Out-File 'C:/Users/jeffr/s2s-voicechanger/msi_run.done'; exit 1 }
$wix = Join-Path $env:USERPROFILE '.dotnet\tools\wix.exe'
$out = 'C:/Users/jeffr/s2s-voicechanger/dist_msi'
New-Item -ItemType Directory -Force $out | Out-Null
$msi = Join-Path $out 'Nekoizer-1.0.0.msi'
& $wix build -arch x64 -d Version=1.0.0 -d UpgradeCode=6C1B6E0A-5E1A-4D7B-9B0E-2C8F5A7D41E3 -d 'SourceDir=C:/Users/jeffr/s2s-voicechanger/build_out_final/dist/Nekoizer' -d 'InstallerDir=C:/Users/jeffr/s2s-voicechanger/installer' -o $msi 'C:/Users/jeffr/s2s-voicechanger/installer\product.wxs' *> 'C:/Users/jeffr/s2s-voicechanger/msi_run.log'
"exit=$LASTEXITCODE" | Out-File 'C:/Users/jeffr/s2s-voicechanger/msi_run.done'
