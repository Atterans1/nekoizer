
$r='C:\Users\jeffr\s2s-voicechanger'; $d="$r\live_run"
Remove-Item "$d\engine_done.txt","$d\engine_proof.json" -EA SilentlyContinue
$p=Start-Process "$r\.venv-seedvc\Scripts\python.exe" -ArgumentList "$r\tools\engine_real_proof.py" -WorkingDirectory $r -RedirectStandardOutput "$d\engine.out" -RedirectStandardError "$d\engine.err" -WindowStyle Hidden -Wait -PassThru
Set-Content "$d\engine_done.txt" ("exit=" + $p.ExitCode)
