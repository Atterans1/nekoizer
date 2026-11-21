
$root='C:\Users\jeffr\s2s-voicechanger'; $a="$root\assets\seedvc"; $py="$root\.venv-seedvc\Scripts\python.exe"
Remove-Item "$a\pip_done.txt" -EA SilentlyContinue
$ver = & $py -c "import importlib.metadata as m; print('torch==' + m.version('torch')); print('torchaudio==' + m.version('torchaudio')); print('numpy==' + m.version('numpy')); print('transformers==' + m.version('transformers')); print('huggingface-hub==' + m.version('huggingface-hub')); print('tokenizers==' + m.version('tokenizers')); print('safetensors==' + m.version('safetensors'))"
Set-Content "$a\constraints.txt" $ver
& $py -m pip install --disable-pip-version-check -c "$a\constraints.txt" silero-vad onnxruntime torchfcpe munch einops *> "$a\pip.log"
Set-Content "$a\pip_done.txt" ("exit=" + $LASTEXITCODE)
