
$r='C:\Users\jeffr\s2s-voicechanger'; $a="$r\assets\seedvc"
Remove-Item "$a\inst_done.txt" -EA SilentlyContinue
$py="$r\.venv-seedvc\Scripts\python.exe"
& $py -m pip install --no-deps --disable-pip-version-check -c "$a\constraints.txt" -r "$a\audiotools_pins.txt" *> "$a\inst.log"
$e1=$LASTEXITCODE
& $py -m pip install --no-deps --disable-pip-version-check descript-audio-codec==1.0.0 *>> "$a\inst.log"
$e2=$LASTEXITCODE
& $py -c "import google.protobuf as p, torch, numpy; import dac.nn.quantize; print('protobuf', p.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__, 'dac ok')" > "$a\inst_check.txt" 2>&1
Set-Content "$a\inst_done.txt" ("audiotools_pins=" + $e1 + " dac=" + $e2)
