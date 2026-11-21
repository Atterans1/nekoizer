
import torch, json
sd = torch.load(r'C:\Users\jeffr\s2s-voicechanger\assets\seedvc\DiT_uvit_tat_xlsr_ema.pth', map_location='cpu', weights_only=False)
net = sd['net'] if 'net' in sd else sd
keys = []
for mod, v in net.items():
    keys += [mod + '.' + k for k in (v.keys() if hasattr(v, 'keys') else [])]
f0 = [k for k in keys if 'f0' in k.lower()]
print(json.dumps({'top_level': list(sd.keys())[:6], 'modules': list(net.keys()), 'n_tensors': len(keys), 'f0_related_keys': f0[:10], 'n_f0_keys': len(f0)}))
