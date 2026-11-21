
$r='C:\Users\jeffr\s2s-voicechanger'; $a="$r\assets\seedvc"
Remove-Item "$a\dry_done.txt" -EA SilentlyContinue
& "$r\.venv-seedvc\Scripts\python.exe" -m pip install --dry-run --disable-pip-version-check -c "$a\constraints.txt" --report "$a\dry_report.json" descript-audiotools *> "$a\dry.log"
Set-Content "$a\dry_done.txt" ("exit=" + $LASTEXITCODE)
