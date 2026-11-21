# Building Nekoizer and its installer

Run everything with the project venv. Neither the model weights nor the PyTorch CUDA runtime libraries ship inside the setup exe: both are downloaded on first run.

    .venv-seedvc\Scripts\python.exe tools\make_torch_manifest.py      # once per torch build: installer\torch_lib_manifest.json (ranged reads of the wheel, takes minutes)
    .venv-seedvc\Scripts\python.exe build_exe.py --version 1.0.0          # manifest, build_meta.py, icon, PyInstaller onedir into dist\Nekoizer, torch\lib removed
    powershell -File installer\build_installer.ps1 -BaseUrl https://atterans.shop/downloads -Version 1.0.0

Needs WiX 6 (`dotnet tool install --global wix --version 6.0.2` and `wix extension add -g WixToolset.BootstrapperApplications.wixext/6.0.2`). WiX 7 asks you to accept the OSMF EULA before it builds; this project pins 6.

## What the installer is

- `Nekoizer-Setup-1.0.0.exe` is a small online installer. It downloads `Nekoizer-1.0.0.msi` from `BaseUrl` when the user runs it and checks it against the hash recorded at build time. Upload `dist_msi\Nekoizer-1.0.0.msi` to `BaseUrl`. The exe has to be built against the final URL.
- The MSI holds the app folder only (about 0.6 GB, `EmbedCab=yes`, under the 2 GB Windows Installer limit). The 3.9 GB of CUDA runtime DLLs are not in it.
- Per user, no administrator rights, installed in `%LOCALAPPDATA%\Programs\Nekoizer`, Start Menu shortcut with the Mimi icon, an entry in Apps and features.
- WiX needs about 2.5 GB of free temp space for the MSI. A `ERROR_DISK_FULL` from WiX means free space, not a build bug.

## First run

Nothing big is in the installer. On first start Nekoizer shows one progress window covering two kinds of file:

- The model files listed in `installer/manifest.json` (about 1.5 GB) go to `%LOCALAPPDATA%\Nekoizer\models`: 8 MB ranges over 4 connections, resumable, SHA256-checked.
- The PyTorch CUDA runtime libraries (58 files, about 2.6 GB compressed in the wheel) come from the official PyTorch wheel `torch-2.14.0+cu126` on download.pytorch.org. `runtime_libs.py` reads each member with ranged requests, four connections at a time, inflates it, and checks size, CRC-32 and SHA256 against `installer/torch_lib_manifest.json`. Files go to `%LOCALAPPDATA%\Nekoizer\runtime\torch_lib`. Then `<install>\_internal\torch\lib` becomes a directory junction to that folder (`mklink /J`, no administrator rights). A damaged or missing file is downloaded again; a correct junction is left alone; a wrong junction is replaced; a non-empty real folder in the way is renamed to `lib.replaced-<time>`, never deleted.

The torch library list is part of the integrity chain: `make_manifest.py` ships `installer/torch_lib_manifest.json` as a shipped file, so its hash is in `manifest.json`, and the hash of `manifest.json` is baked into the exe.

Uninstalling leaves `%LOCALAPPDATA%\Nekoizer` alone; delete it to remove models, runtime libraries, logs and RVC voices. A partial download resumes where it stopped.

## Integrity

`installer/make_manifest.py` hashes the shipped configs and the torch library list, and lists each model's URL, size and SHA256; the manifest's own hash is baked into the exe. At start-up the app refuses a modified manifest, a modified config or a model whose hash differs. This catches damaged downloads and casual edits. It is tamper-evidence, not copy protection.

The torch library SHA256 values are taken from the local venv files, which `tools/make_torch_manifest.py` checks byte-for-byte in size and CRC-32 against the wheel's own zip directory before writing the list.

## Not provided

- The exe and MSI are not code-signed, so Windows SmartScreen will warn on first run. Signing needs a certificate.
- An NVIDIA GPU with a current driver, and the VB-Cable virtual audio cable, are requirements the installer does not supply.
