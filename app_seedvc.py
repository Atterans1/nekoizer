"""Desktop entry point: python app_seedvc.py [--headless-test SECONDS] [--no-window] [--port N]

Starts the Pipeline (live microphone -> Seed-VC -> CABLE Input), the local UI server and the global hotkeys, then shows the UI
window with pywebview (or just serves it with --no-window).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import appdata  # noqa: E402
import download_weights  # noqa: E402
import integrity  # noqa: E402
import runtime_libs  # noqa: E402
import seedvc_params as P  # noqa: E402

LOG = logging.getLogger("seedvc.app")


# ---------------------------------------------------------------- PUBLIC
def Main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ROOT / "assets" / "seedvc"))
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-window", action="store_true", help="serve the UI without opening the window")
    ap.add_argument("--headless-test", type=float, default=0.0, metavar="SECONDS",
                    help="run the live pipeline for SECONDS, print telemetry as JSON, then exit")
    ap.add_argument("--input-device", default=None)
    ap.add_argument("--reference", default=None, metavar="WAV", help="load this clip into reference slot 0 (same as dropping it on the Neutral slot)")
    ap.add_argument("--report", default=None, help="write the telemetry JSON of a headless test to this path")
    args = ap.parse_args(argv)
    _ConfigureLogging()
    try:
        assets = _PrepareAssets(args.assets, windowed=not (args.no_window or args.headless_test > 0))
    except (integrity.IntegrityError, download_weights.DownloadError, runtime_libs.RuntimeLibsError) as exc:
        LOG.error("%s", exc)
        _ShowError(str(exc))
        return 1

    from pipeline import Pipeline
    pipe = Pipeline(assets, input_device=args.input_device)
    info = pipe.Start()
    LOG.info("pipeline started: %s", json.dumps(info, default=str)[:400])
    janitor = pipe.janitor
    if args.reference:
        import soundfile as sf
        wav, sr = sf.read(args.reference, dtype="float32")
        LOG.info("reference slot 0: %s", json.dumps(pipe.SetSlotFromWav(0, wav.mean(axis=1) if wav.ndim > 1 else wav, sr), default=str))
    server = None
    hotkeys = None
    try:
        from ui.server import CreateApp
        import uvicorn
        config = uvicorn.Config(CreateApp(pipe), host="127.0.0.1", port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, name="seedvc-ui-server", daemon=True).start()
        janitor.Add(lambda: setattr(server, "should_exit", True))
        from ui.hotkeys import HotkeyRegistry
        hotkeys = HotkeyRegistry(pipe.SetManualSlot, janitor)
        LOG.info("hotkeys: %s", json.dumps(hotkeys.Start()))
        if args.headless_test > 0:
            return _HeadlessTest(pipe, args.headless_test, args.report)
        url = f"http://127.0.0.1:{args.port}/"
        if args.no_window:
            LOG.info("serving %s (Ctrl+C to stop)", url)
            while True:
                time.sleep(1.0)
        import webview
        webview.create_window("Nekoizer", url, width=1100, height=820, background_color="#FFF4F8")
        webview.start()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        pipe.Stop()


# ---------------------------------------------------------------- PRIVATE
class _Job(NamedTuple):
    name: str
    size: int  # bytes the job transfers, for the progress window
    run: Callable  # run(progress) where progress(name, done, size) or None


def _ConfigureLogging() -> None:
    """A packaged windowed app has no console: it logs to a file under the user-data folder."""
    if not appdata.IsFrozen():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
        return
    log_dir = appdata.DataDir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=log_dir / "app.log", format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.excepthook = lambda kind, exc, tb: LOG.error("uncaught exception", exc_info=(kind, exc, tb))


def _PrepareAssets(default_assets: str, windowed: bool = False) -> str:
    """Return the folder the engine loads from. A source tree keeps using its own assets folder. A packaged build (or S2S_DATA_DIR) first
    verifies the shipped files against the build manifest, downloads any missing model and any missing PyTorch runtime library (packaged
    build only), and checks every model's SHA256 before the engine may read it (a model hashed before is only stat-ed). The runtime
    libraries are linked into the bundle last; this runs before anything imports torch."""
    if not (appdata.IsFrozen() or os.environ.get(appdata.DATA_DIR_ENV)):
        return default_assets
    manifest = integrity.LoadManifest(appdata.ManifestPath(), appdata.BakedManifestSha256())
    integrity.VerifyShippedFiles(manifest, appdata.AppDir())
    models_dir, cache = appdata.ModelsDir(), appdata.DataDir() / integrity.CACHE_NAME
    models_dir.mkdir(parents=True, exist_ok=True)
    download_weights.CopyShipped(manifest, models_dir)
    jobs = [_Job(model["name"], model["size"], partial(download_weights.Fetch, model, models_dir, cache))
            for model in download_weights.Missing(manifest, models_dir)]
    if appdata.IsFrozen():
        jobs += [_Job(entry["member"], entry["compress_size"], partial(_FetchRuntimeEntry, entry)) for entry in runtime_libs.Missing()]
    _RunJobs(jobs, windowed)
    if appdata.IsFrozen():
        runtime_libs.Link()
    verifier = integrity.ModelVerifier(manifest, models_dir, cache)
    verifier.Start()
    verifier.Wait()
    LOG.info("models verified: %s", json.dumps(verifier.Stats()))
    return str(models_dir)


_SPLASH_HTML = """<!doctype html><meta charset="utf-8"><body style="margin:0;padding:24px;background:#FFF4F8;color:#4A2B5A;font:15px Consolas,monospace">
<h2 style="margin:0 0 8px">Nekoizer</h2><p style="margin:0 0 14px">First start: downloading the voice models and the PyTorch libraries (about __SIZE__ GB). This happens once.</p>
<div style="border:3px solid #4A2B5A;background:#fff;height:22px"><div id="bar" style="height:100%;width:0;background:#FF8FC7"></div></div>
<p id="msg" style="margin:10px 0 0" role="status">Starting ...</p>
<script>function setProgress(name,done,total,i,n){document.getElementById("bar").style.width=(100*done/total)+"%";document.getElementById("msg").textContent="File "+(i+1)+" of "+n+": "+name+" ("+Math.round(done/1048576)+" of "+Math.round(total/1048576)+" MB)"}</script>"""


def _FetchRuntimeEntry(entry: dict, progress=None) -> None:
    """One first-run runtime-library job: route the progress callback to FetchMember's own keyword (its second positional slot is runtime_dir)."""
    runtime_libs.FetchMember(entry, progress=progress)


def _RunJobs(jobs: list, windowed: bool) -> None:
    """Run the first-run downloads: one progress window when the app has a window, otherwise a plain log line per file."""
    if not jobs:
        return
    if windowed:
        _DownloadWithWindow(jobs)
        return
    for job in jobs:
        LOG.info("first run: downloading %s (%.0f MB)", job.name, job.size / 2**20)
        job.run(None)


def _DownloadWithWindow(jobs: list) -> None:
    """Fetch the missing models and runtime libraries while one progress window is open; the real window opens afterwards. A failure is re-raised."""
    import webview
    total = sum(job.size for job in jobs)
    html = _SPLASH_HTML.replace("__SIZE__", f"{total / 2**30:.1f}")
    window = webview.create_window("Nekoizer", html=html, width=560, height=230, background_color="#FFF4F8")
    failure: list[BaseException] = []
    finished: list[int] = []

    def work() -> None:
        try:
            for i, job in enumerate(jobs):
                last = [0]

                def progress(name: str, done: int, size: int, i=i) -> None:
                    if done - last[0] >= 4 << 20 or done == size:
                        last[0] = done
                        window.evaluate_js(f"setProgress({json.dumps(name)},{done},{size},{i},{len(jobs)})")

                LOG.info("first run: downloading %s (%.0f MB)", job.name, job.size / 2**20)
                job.run(progress)
                finished.append(i)
        except BaseException as exc:  # re-raised below, on the main thread
            failure.append(exc)
        window.destroy()

    webview.start(work)
    if failure:
        raise failure[0]
    if len(finished) < len(jobs):
        raise download_weights.DownloadError("The download window was closed before the models finished. Start Nekoizer again to continue; it resumes where it stopped.")


def _ShowError(message: str) -> None:
    """A windowed build has no console, so a start-up failure is shown in a message box (and is already in the log)."""
    if appdata.IsFrozen() and os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "Nekoizer", 0x10)
    else:
        print(message, file=sys.stderr)


def _HeadlessTest(pipe, seconds: float, report_path: str | None) -> int:
    t0 = time.time()
    samples = []
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        samples.append(pipe.Telemetry())
    final = pipe.Telemetry()
    out = {"seconds": seconds, "final": final, "n_samples": len(samples)}
    text = json.dumps(out, indent=1, default=str)
    if report_path:
        Path(report_path).write_text(text, encoding="utf-8")
    print("TELEMETRY", text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(Main())
