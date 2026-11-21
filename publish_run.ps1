$ErrorActionPreference = 'Stop'
$wix = Join-Path $env:USERPROFILE '.dotnet\tools\wix.exe'
$out = 'C:/Users/jeffr/s2s-voicechanger/dist_msi'
$dl = 'C:/Users/jeffr/nekoizer-host/downloads'
& $wix build -arch x64 -ext WixToolset.BootstrapperApplications.wixext -d Version=1.0.0 -d BaseUrl=https://atterans.shop/downloads -d 'InstallerDir=C:/Users/jeffr/s2s-voicechanger/installer' -d "MsiDir=$out" -o "$out\Nekoizer-Setup-1.0.0.exe" 'C:/Users/jeffr/s2s-voicechanger/installer\bundle.wxs' *> 'C:/Users/jeffr/s2s-voicechanger/publish.log'
if ($LASTEXITCODE) { "exit=$LASTEXITCODE stage=bundle" | Out-File 'C:/Users/jeffr/s2s-voicechanger/publish.done'; exit 1 }
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force "$dl/previous" | Out-Null
Copy-Item "$dl/Nekoizer-Setup-1.0.0.exe" "$dl/previous/Nekoizer-Setup-1.0.0.$stamp.exe"
Copy-Item "$out\Nekoizer-1.0.0.msi" "$dl/Nekoizer-1.0.0.msi" -Force
Copy-Item "$out\Nekoizer-Setup-1.0.0.exe" "$dl/Nekoizer-Setup-1.0.0.exe" -Force
"exit=0 stage=published" | Out-File 'C:/Users/jeffr/s2s-voicechanger/publish.done'
