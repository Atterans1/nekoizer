
$a='C:\Users\jeffr\s2s-voicechanger\assets\seedvc'
Remove-Item "$a\dl_done.txt" -EA SilentlyContinue
$m = Get-Content "$a\manifest.json" -Raw | ConvertFrom-Json
$out=@()
foreach($it in $m.items){
  $dest = $it.dest -replace '/','\'
  New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
  for($i=0;$i -lt 40;$i++){
    & curl.exe -L -C - --retry 5 --retry-delay 3 --connect-timeout 20 --speed-time 40 --speed-limit 1000 -o $dest $it.url 2>> "$a\dl.err"
    $len = (Get-Item $dest -EA SilentlyContinue).Length
    if($LASTEXITCODE -eq 0 -and ($it.size -eq $null -or $len -eq $it.size)){ break }
  }
  $len = (Get-Item $dest -EA SilentlyContinue).Length
  $shaOk = 'n/a'
  if($it.sha -and $len -eq $it.size){ $shaOk = ((Get-FileHash $dest -Algorithm SHA256).Hash.ToLower() -eq $it.sha) }
  $out += ((Split-Path $dest -Leaf) + ' size=' + $len + ' expected=' + $it.size + ' sha_ok=' + $shaOk)
  Set-Content "$a\dl_progress.txt" ($out -join '|')
}
Set-Content "$a\dl_done.txt" ($out -join '|')
