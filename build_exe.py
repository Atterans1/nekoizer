"""Build Nekoizer (PyInstaller onedir) from app_seedvc.py.  python build_exe.py [--console] [--version 1.0.0]
Run with the project venv: .venv-seedvc\\Scripts\\python.exe build_exe.py. Model weights are NOT bundled: the first run downloads them (download_weights.py)."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "installer"))
import make_icon  # noqa: E402  (writes installer/mimi.ico from the mascot sprite)
import make_manifest  # noqa: E402

COLLECT_ALL = ["torchfcpe", "webview", "pyworld"]
COLLECT_DATA = ["audiotools", "silero_vad"]  # audiotools reads templates/headers.html at import, audio/vad.py finds silero_vad.onnx in the package
COLLECT_SUBMODULES = ["uvicorn", "websockets", "modules", "engine", "audio", "ui"]
RVC_MODULES = ["third_party.rvc." + name for name in ("models", "modules", "attentions", "commons", "transforms", "rmvpe")]  # imported lazily by engine/rvc_*.py
HIDDEN = RVC_MODULES + ["silero_vad", "pipeline", "seedvc_params", "appdata", "integrity", "download_weights", "runtime_libs", "build_meta", "soundfile", "sounddevice", "yaml", "multipart"]
COPY_METADATA = ["numpy", "tqdm", "regex", "packaging", "filelock", "tokenizers", "safetensors", "huggingface-hub", "transformers", "pyyaml", "torch",
                 "uvicorn", "fastapi", "starlette", "pydantic", "pywebview", "librosa", "scipy"]
EXCLUDES = ["tkinter", "matplotlib", "IPython", "pytest", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6"]


def Main(argv) -> int:
    version = argv[argv.index("--version") + 1] if "--version" in argv else "1.0.0"
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "build_out"
    if out.exists():
        raise SystemExit(f"refusing: {out} already exists; pass a fresh --out folder")
    make_manifest.Build(version)
    sources = json.loads(make_manifest.SOURCES.read_text(encoding="utf-8"))
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "Nekoizer", "--icon", str(make_icon.ICO),
           "--distpath", str(out / "dist"), "--workpath", str(out / "work"), "--specpath", str(out / "work"),
           "--paths", str(ROOT), "--paths", str(ROOT / "third_party" / "seed-vc"),
           "--add-data", f"{ROOT / 'ui' / 'static'};ui/static", "--add-data", f"{make_manifest.MANIFEST};."]
    for entry in sources["shipped"]:
        cmd += ["--add-data", f"{ROOT / entry['src']};{entry['dest']}"]
    # torch.jit.script reads a function's .py source: modules/commons.py and dac/nn/layers.py (snake) are scripted, so ship their .py next to the .pyc
    dac_dir = list(importlib.util.find_spec("dac").submodule_search_locations)[0]
    cmd += ["--add-data", f"{ROOT / 'third_party' / 'seed-vc' / 'modules'};modules", "--add-data", f"{dac_dir};dac"]
    cmd += ["--console"] if "--console" in argv else ["--windowed"]
    for flag, names in (("--collect-all", COLLECT_ALL), ("--collect-data", COLLECT_DATA), ("--collect-submodules", COLLECT_SUBMODULES), ("--hidden-import", HIDDEN),
                        ("--copy-metadata", COPY_METADATA), ("--exclude-module", EXCLUDES)):
        for name in names:
            cmd += [flag, name]
    cmd.append(str(ROOT / "app_seedvc.py"))
    code = subprocess.call(cmd)
    if code == 0:
        _DropBundledTorchLib(out / "dist" / "Nekoizer")
    return code


# ---------------------------------------------------------------- PRIVATE
def _DropBundledTorchLib(bundle: Path) -> None:
    """The 3.9 GB of CUDA runtime DLLs stay out of the installer: the app fetches them on first run (runtime_libs.py, installer/torch_lib_manifest.json)."""
    lib = bundle / "_internal" / "torch" / "lib"
    if lib.exists():
        shutil.rmtree(lib)


if __name__ == "__main__":
    sys.exit(Main(sys.argv[1:]))
