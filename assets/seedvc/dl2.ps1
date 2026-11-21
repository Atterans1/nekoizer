
$a='C:\Users\jeffr\s2s-voicechanger\assets\seedvc'
Remove-Item "$a\xlsr_done.txt" -EA SilentlyContinue
$it = Get-Content "$a\xlsr_item.json" -Raw | ConvertFrom-Json
$dest = $it.dest -replace '/','\'
for($i=0;$i -lt 400;$i++){
  & curl.exe -L -C - --retry 3 --retry-delay 5 --connect-timeout 20 --speed-time 60 --speed-limit 500 -o $dest $it.url 2>> "$a\dl2.err"
  $len=(Get-Item $dest -EA SilentlyContinue).Length
  if($len -eq $it.size){ break }
  Start-Sleep -Seconds 10
}
$len=(Get-Item $dest -EA SilentlyContinue).Length
$ok = $false
if($len -eq $it.size){ $ok = ((Get-FileHash $dest -Algorithm SHA256).Hash.ToLower() -eq $it.sha) }
$line = ('pytorch_model.bin size=' + $len + ' expected=' + $it.size + ' sha_ok=' + $ok)
$prog = (Get-Content "$a\dl_progress.txt" -Raw) -split '\|'
$prog = $prog | ForEach-Object { if($_ -like 'pytorch_model.bin*'){ $line } else { $_ } }
Set-Content "$a\dl_progress.txt" ($prog -join '|')
Set-Content "$a\xlsr_done.txt" $line
