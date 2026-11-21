
$d='C:\Users\jeffr\s2s-voicechanger\third_party'; $sha='51383efd921027683c89e5348211d93ff12ac2a8'
Remove-Item "$d\getsv_done.txt" -EA SilentlyContinue
$z="$d\seed-vc-$sha.zip"
for($i=0;$i -lt 30;$i++){
  & curl.exe -L -C - --retry 5 --retry-delay 3 --connect-timeout 20 --speed-time 40 --speed-limit 1000 -o $z "https://github.com/Plachtaa/seed-vc/archive/$sha.zip" 2>> "$d\getsv.err"
  if($LASTEXITCODE -eq 0){ break }
}
$ok = ($LASTEXITCODE -eq 0)
if($ok){
  Expand-Archive -Path $z -DestinationPath "$d\_x" -Force
  $src = Get-ChildItem "$d\_x" -Directory | Select-Object -First 1
  if(Test-Path "$d\seed-vc"){ Remove-Item "$d\seed-vc" -Recurse -Force }
  Move-Item $src.FullName "$d\seed-vc"
  Remove-Item "$d\_x" -Recurse -Force -EA SilentlyContinue
  Set-Content "$d\seed-vc\PINNED_COMMIT.txt" $sha
}
Set-Content "$d\getsv_done.txt" ("ok=" + $ok + " zip_bytes=" + (Get-Item $z -EA SilentlyContinue).Length + " has_modules=" + (Test-Path "$d\seed-vc\modules"))
