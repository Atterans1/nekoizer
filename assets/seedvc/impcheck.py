
import importlib, json, sys
out = {}
for m in ['torch','torchaudio','transformers','onnxruntime','silero_vad','torchfcpe','munch','einops','sounddevice','pyworld','soundfile','scipy','librosa','fastapi','uvicorn','multipart','websockets','webview','yaml','omegaconf']:
    try:
        x = importlib.import_module(m); out[m] = getattr(x, '__version__', 'ok')
    except Exception as e:
        out[m] = 'MISSING ' + type(e).__name__ + ': ' + str(e)[:80]
import torch
out['cuda'] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
import silero_vad, os
d = os.path.dirname(silero_vad.__file__)
out['silero_onnx'] = [f for f in os.listdir(os.path.join(d, 'data'))] if os.path.isdir(os.path.join(d, 'data')) else 'no data dir'
print(json.dumps(out, indent=1))
