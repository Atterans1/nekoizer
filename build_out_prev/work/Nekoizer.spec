# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:/Users/jeffr/s2s-voicechanger/ui/static', 'ui/static'), ('C:/Users/jeffr/s2s-voicechanger/installer/manifest.json', '.'), ('C:/Users/jeffr/s2s-voicechanger/installer/torch_lib_manifest.json', 'installer'), ('C:/Users/jeffr/s2s-voicechanger/assets/seedvc/config_dit_mel_seed_uvit_xlsr_tiny.yml', 'assets/seedvc'), ('C:/Users/jeffr/s2s-voicechanger/assets/seedvc/wav2vec2-xls-r-300m/config.json', 'assets/seedvc/wav2vec2-xls-r-300m'), ('C:/Users/jeffr/s2s-voicechanger/assets/seedvc/wav2vec2-xls-r-300m/preprocessor_config.json', 'assets/seedvc/wav2vec2-xls-r-300m'), ('C:/Users/jeffr/s2s-voicechanger/third_party/seed-vc/configs/hifigan.yml', 'third_party/seed-vc/configs'), ('C:/Users/jeffr/s2s-voicechanger/third_party/seed-vc/LICENSE', 'third_party/seed-vc'), ('C:/Users/jeffr/s2s-voicechanger/third_party/seed-vc/modules', 'modules'), ('C:/Users/jeffr/s2s-voicechanger/.venv-seedvc/Lib/site-packages/dac', 'dac')]
binaries = []
hiddenimports = ['third_party.rvc.models', 'third_party.rvc.modules', 'third_party.rvc.attentions', 'third_party.rvc.commons', 'third_party.rvc.transforms', 'third_party.rvc.rmvpe', 'silero_vad', 'pipeline', 'seedvc_params', 'appdata', 'integrity', 'download_weights', 'runtime_libs', 'build_meta', 'soundfile', 'sounddevice', 'yaml', 'multipart']
datas += collect_data_files('audiotools')
datas += collect_data_files('silero_vad')
datas += copy_metadata('numpy')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('tokenizers')
datas += copy_metadata('safetensors')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('transformers')
datas += copy_metadata('pyyaml')
datas += copy_metadata('torch')
datas += copy_metadata('uvicorn')
datas += copy_metadata('fastapi')
datas += copy_metadata('starlette')
datas += copy_metadata('pydantic')
datas += copy_metadata('pywebview')
datas += copy_metadata('librosa')
datas += copy_metadata('scipy')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('websockets')
hiddenimports += collect_submodules('modules')
hiddenimports += collect_submodules('engine')
hiddenimports += collect_submodules('audio')
hiddenimports += collect_submodules('ui')
tmp_ret = collect_all('torchfcpe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyworld')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:/Users/jeffr/s2s-voicechanger/app_seedvc.py'],
    pathex=['C:/Users/jeffr/s2s-voicechanger', 'C:/Users/jeffr/s2s-voicechanger/third_party/seed-vc'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'notebook', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Nekoizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:/Users/jeffr/s2s-voicechanger/installer/mimi.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Nekoizer',
)
