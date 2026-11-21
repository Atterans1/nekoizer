$r = 'C:\Users\jeffr\s2s-voicechanger'
$d = "$r\evaluation_seedvc\_scratch"
Remove-Item "$d\probe_env.done" -ErrorAction SilentlyContinue
$p = Start-Process "$r\.venv-seedvc\Scripts\python.exe" -ArgumentList @("$d\probe_env.py") -WorkingDirectory $r -RedirectStandardOutput "$d\probe_env.out" -RedirectStandardError "$d\probe_env.err" -WindowStyle Hidden -Wait -PassThru
Set-Content "$d\probe_env.done" ("exit=" + $p.ExitCode)
