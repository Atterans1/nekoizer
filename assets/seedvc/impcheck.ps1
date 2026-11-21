
$a='C:\Users\jeffr\s2s-voicechanger\assets\seedvc'
Remove-Item "$a\imp_done.txt" -EA SilentlyContinue
$p=Start-Process 'C:\Users\jeffr\s2s-voicechanger\.venv-seedvc\Scripts\python.exe' -ArgumentList "$a\impcheck.py" -RedirectStandardOutput "$a\imp_out.txt" -RedirectStandardError "$a\imp_err.txt" -WindowStyle Hidden -Wait -PassThru
Set-Content "$a\imp_done.txt" ("exit="+$p.ExitCode)
