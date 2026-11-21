
# Detached live run: starts app_seedvc.py --headless-test N, writes live_run\report.json and live_run\exit.txt
param([int]$Seconds = 60, [string]$Reference = "")
$r='C:\Users\jeffr\s2s-voicechanger'; $d="$r\live_run"
Remove-Item "$d\exit.txt","$d\report.json","$d\out.txt","$d\err.txt" -EA SilentlyContinue
$env:HF_HUB_OFFLINE='1'
$ref = if($Reference){ $Reference } else { "$r\testdata\ref_0.wav" }
$p=Start-Process "$r\.venv-seedvc\Scripts\python.exe" -ArgumentList "$r\app_seedvc.py","--headless-test","$Seconds","--no-window","--report","$d\report.json","--reference",$ref -WorkingDirectory $r -RedirectStandardOutput "$d\out.txt" -RedirectStandardError "$d\err.txt" -WindowStyle Hidden -PassThru
$p.WaitForExit()
Set-Content "$d\exit.txt" ("exit=" + $p.ExitCode)
