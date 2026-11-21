param([string]$Target)
Start-Process powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$Target) -WindowStyle Hidden
