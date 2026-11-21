
$o=@()
foreach($p in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*smoke_old*' })){ $g=Get-Process -Id $p.ProcessId -EA SilentlyContinue; if($g){ $o += ($p.ProcessId.ToString() + ' cpu_s=' + [int]$g.CPU + ' ws_mb=' + [int]($g.WS/1MB) + ' age_min=' + [int]((Get-Date) - $g.StartTime).TotalMinutes + ' threads=' + $g.Threads.Count) } }
$o += ('now ' + (Get-Date).ToString('HH:mm:ss'))
Set-Content 'C:\Users\jeffr\s2s-voicechanger\assets\seedvc\pq2.txt' ($o -join "|")
