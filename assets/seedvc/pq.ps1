
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { $_.ProcessId.ToString() + ' | ' + $_.CreationDate.ToString('HH:mm:ss') + ' | ' + [int]((Get-Process -Id $_.ProcessId -EA SilentlyContinue).WS/1MB) + 'MB | ' + $_.CommandLine.Substring(0,[Math]::Min(120,$_.CommandLine.Length)) } | Set-Content 'C:\Users\jeffr\s2s-voicechanger\assets\seedvc\pq.txt'
