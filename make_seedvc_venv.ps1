
$root='C:\Users\jeffr\s2s-voicechanger'
$new="$root\.venv-seedvc"; $old="$root\.venv\Lib\site-packages"; $log="$root\assets\seedvc\venv.log"
Remove-Item "$root\assets\seedvc\venv_done.txt" -EA SilentlyContinue
if(-not (Test-Path "$new\Scripts\python.exe")){ & 'C:\Users\jeffr\apps\py312\python.exe' -m venv $new *> $log }
$dst="$new\Lib\site-packages"
$n=0; $skipped=0; $failed=0
Get-ChildItem $old -Recurse -File -Force | ForEach-Object {
  $rel=$_.FullName.Substring($old.Length)
  $t=Join-Path $dst $rel
  if(Test-Path $t){ $skipped++; return }
  $d=Split-Path $t
  if(-not (Test-Path $d)){ New-Item -ItemType Directory -Force $d | Out-Null }
  try { New-Item -ItemType HardLink -Path $t -Target $_.FullName -ErrorAction Stop | Out-Null; $n++ } catch { $failed++ }
  if(($n % 2000) -eq 0){ Set-Content "$root\assets\seedvc\venv_progress.txt" ("linked=$n skipped=$skipped failed=$failed") }
}
Set-Content "$root\assets\seedvc\venv_done.txt" ("linked=$n skipped=$skipped failed=$failed")
