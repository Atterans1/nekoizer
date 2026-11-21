
import json, sys, importlib
out = {}
for m in ['audiotools', 'dac', 'dac.nn.quantize']:
    try:
        x = importlib.import_module(m); out[m] = 'ok ' + str(getattr(x, '__version__', ''))
    except Exception as e:
        out[m] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:160]
import google.protobuf as pb
out['protobuf_in_use'] = pb.__version__
import torch
out['torch'] = torch.__version__
try:
    from dac.nn.quantize import VectorQuantize
    out['VectorQuantize'] = 'importable'
except Exception as e:
    out['VectorQuantize'] = 'FAIL ' + type(e).__name__ + ': ' + str(e)[:120]
print(json.dumps(out))
