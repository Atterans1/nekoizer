
$r='C:\Users\jeffr\s2s-voicechanger'; $a="$r\assets\seedvc"; $t="C:\Users\jeffr\AppData\Local\Temp\audiotools_probe"
Remove-Item "$a\probe_done.txt" -EA SilentlyContinue
if(Test-Path $t){ Remove-Item $t -Recurse -Force }
$py="$r\.venv-seedvc\Scripts\python.exe"
& $py -m pip install --no-deps --disable-pip-version-check --target $t -c "$a\constraints.txt" -r "$a\audiotools_pins.txt" *> "$a\probe_pip.log"
$pipExit=$LASTEXITCODE
& $py -m pip install --no-deps --disable-pip-version-check --target $t descript-audio-codec==1.0.0 *>> "$a\probe_pip.log"
$env:PYTHONPATH=$t
$p=Start-Process $py -ArgumentList "$a\probe_import.py" -RedirectStandardOutput "$a\probe_out.txt" -RedirectStandardError "$a\probe_err.txt" -WindowStyle Hidden -Wait -PassThru
$size=[int]((Get-ChildItem $t -Recurse -File | Measure-Object Length -Sum).Sum/1MB)
Set-Content "$a\probe_done.txt" ("pip_exit=" + $pipExit + " import_exit=" + $p.ExitCode + " target_MB=" + $size)
