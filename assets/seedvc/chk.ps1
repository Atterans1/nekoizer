
$a='C:\Users\jeffr\s2s-voicechanger\assets\seedvc'; $r='C:\Users\jeffr\s2s-voicechanger'
Remove-Item "$a\chk_done.txt" -EA SilentlyContinue
$p1=Start-Process "$r\.venv-seedvc\Scripts\python.exe" -ArgumentList "$a\f0keys.py" -RedirectStandardOutput "$a\f0_out.txt" -RedirectStandardError "$a\f0_err.txt" -WindowStyle Hidden -Wait -PassThru
$p2=Start-Process "$r\.venv\Scripts\python.exe" -ArgumentList "$a\smoke_old.py" -RedirectStandardOutput "$a\old_out.txt" -RedirectStandardError "$a\old_err.txt" -WindowStyle Hidden -Wait -PassThru
Set-Content "$a\chk_done.txt" ("f0=" + $p1.ExitCode + " old=" + $p2.ExitCode)
