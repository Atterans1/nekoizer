$ErrorActionPreference = 'Continue'
$env:HF_HUB_DISABLE_XET = '1'
$dst = 'C:\Users\jeffr\s2s-voicechanger\evaluation_seedvc\models\whisper-tiny.en'
$log = 'C:\Users\jeffr\s2s-voicechanger\evaluation_seedvc\models\whisper_dl.log'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$start = Get-Date
('start ' + $start.ToString('o')) | Set-Content $log
Remove-Item 'C:\Users\jeffr\s2s-voicechanger\evaluation_seedvc\models\whisper_dl_done.txt' -ErrorAction SilentlyContinue
$want = @('config.json','generation_config.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','normalizer.json','vocab.json','merges.txt','tokenizer.json','model.safetensors')
$info = Invoke-RestMethod 'https://huggingface.co/api/models/openai/whisper-tiny.en?blobs=true'
$report = @()
foreach ($s in $info.siblings) {
  if ($want -notcontains $s.rfilename) { continue }
  $out = Join-Path $dst $s.rfilename
  & curl.exe -L --fail --silent --show-error --retry 8 --retry-delay 2 --connect-timeout 20 -C - -o $out ('https://huggingface.co/openai/whisper-tiny.en/resolve/main/' + $s.rfilename) 2>> $log
  $len = 0
  if (Test-Path $out) { $len = (Get-Item $out).Length }
  $shaOk = $null
  if ($s.lfs -and (Test-Path $out) -and ($len -eq $s.size)) { $shaOk = ((Get-FileHash $out -Algorithm SHA256).Hash.ToLower() -eq $s.lfs.sha256) }
  $row = [pscustomobject]@{ file = $s.rfilename; size_expected = $s.size; size_got = $len; sha_ok = $shaOk; elapsed_s = [math]::Round(((Get-Date) - $start).TotalSeconds, 1) }
  $report += $row
  ($row | ConvertTo-Json -Compress) | Add-Content $log
}
$report | ConvertTo-Json | Set-Content 'C:\Users\jeffr\s2s-voicechanger\evaluation_seedvc\models\whisper_dl.json'
Set-Content 'C:\Users\jeffr\s2s-voicechanger\evaluation_seedvc\models\whisper_dl_done.txt' ('done elapsed_s=' + [math]::Round(((Get-Date) - $start).TotalSeconds, 1))
