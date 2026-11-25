
$ErrorActionPreference = 'Continue'
Set-Location 'C:/Users/jeffr/s2s-voicechanger'
Remove-Item -ErrorAction SilentlyContinue 'installer/release_build.done'
& powershell -NoProfile -ExecutionPolicy Bypass -File installer/build_installer.ps1 -BaseUrl 'https://atterans.shop/downloads' -SourceDir 'C:/Users/jeffr/s2s-voicechanger/build_out3/dist/Nekoizer' *> installer/release_build.log
"exit=$LASTEXITCODE" | Out-File -Encoding ascii installer/release_build.done
