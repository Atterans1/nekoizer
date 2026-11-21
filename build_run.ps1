Set-Location 'C:/Users/jeffr/s2s-voicechanger/'
$env:PYTHONUNBUFFERED='1'
$sw=[Diagnostics.Stopwatch]::StartNew()
& .\.venv-seedvc\Scripts\python.exe build_exe.py --version 1.0.0 --out C:/Users/jeffr/s2s-voicechanger/build_out2 *> build_run.log
"exit=$LASTEXITCODE secs=$($sw.Elapsed.TotalSeconds)" | Out-File build_run.done
