"""Native desktop shell: runs the FastAPI server in a thread and shows it in a pywebview window."""
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import uvicorn
import webview

import server

# ---------------------------------------------------------------- PUBLIC

WindowTitle = "S2S Voice Changer"
WindowWidth = 1100
WindowHeight = 760
StartupTimeoutS = 180.0


def main() -> None:
    _RedirectConsoleIfWindowed()
    config = uvicorn.Config(server.app, host=server.Host, port=server.Port, log_level="warning")
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, name="uvicorn", daemon=True)
    thread.start()
    try:
        _WaitForServer(thread)
        webview.create_window(
            WindowTitle,
            "http://" + server.Host + ":" + str(server.Port) + "/",
            width=WindowWidth,
            height=WindowHeight,
        )
        webview.start()
    finally:
        uv_server.should_exit = True
        thread.join(timeout=15)


# ---------------------------------------------------------------- PRIVATE

def _RedirectConsoleIfWindowed() -> None:
    """A windowed exe has no stdout/stderr; uvicorn and torch would crash writing to None. Log to a file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_dir = server.DataRoot
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = open(log_dir / "app.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream


def _WaitForServer(thread: threading.Thread) -> None:
    url = "http://" + server.Host + ":" + str(server.Port) + "/api/state"
    deadline = time.monotonic() + StartupTimeoutS
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("server thread exited during startup; see the traceback above")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    raise RuntimeError("server did not answer /api/state within " + str(int(StartupTimeoutS)) + " s")


if __name__ == "__main__":
    main()
