
import importlib, json
out = {}
for m in ['torch','chatterbox','pyworld','sounddevice','silero_vad','fastapi','webview','resemblyzer']:
    try: x = importlib.import_module(m); out[m] = getattr(x, '__version__', 'ok')
    except Exception as e: out[m] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:70]
print(json.dumps(out))
