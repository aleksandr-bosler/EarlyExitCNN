import torch
import torch.nn as nn
from timm import create_model
from thop import profile
import pandas as pd

class EarlyExitMobileNetV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, dropout=0.5):
        super().__init__()
        self.backbone = create_model(
            'mobilenetv2_100', pretrained=False,
            in_chans=in_chans, features_only=True
        )
        chs = [f['num_chs'] for f in self.backbone.feature_info.get_dicts()]
        c1, c2, c3 = chs[2], chs[3], chs[-1]
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.1), nn.Linear(c1, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(c2, num_classes)
        )
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(c3, num_classes)
        )

    def forward_features(self, x):
        return self.backbone(x)

def build_branch(model, exit_idx):
    info = model.backbone.feature_info.get_dicts()
    target = info[{1:2,2:3,3:len(info)-1}[exit_idx]]['module']
    parent, idx = target.split('.'); idx = int(idx)
    layers = []
    for name, child in model.backbone.named_children():
        if name == parent:
            layers.append(nn.Sequential(*list(child.children())[:idx+1]))
            break
        else:
            layers.append(child)
    head = {1: model.exit1, 2: model.exit2, 3: model.exit3}[exit_idx]
    return nn.Sequential(*layers, head), head

device = 'cpu'
model = EarlyExitMobileNetV2(in_chans=3, num_classes=2).to(device).eval()

results = []
prev_flops = 0.0
dummy = torch.randn(1, 3, 224, 224).to(device)

for k in [1,2,3]:
    branch, head = build_branch(model, exit_idx=k)
    branch = branch.to(device).eval()
    flops, params = profile(branch, inputs=(dummy,), verbose=False)

    cum_flops   = flops / 1e9
    delta_flops = (cum_flops - prev_flops) * 1e3

    total_params_kb = params * 4 / 1e3
    head_params_kb  = sum(p.numel() for p in head.parameters()) * 4 / 1e3

    results.append({
        'Exit k':           k,
        'Cum. FLOPs (GF)':  round(cum_flops, 2),
        'Δ FLOPs (MF)':     round(delta_flops, 1),
        'Total Params (kB)':round(total_params_kb, 1),
        'Δ Params (kB)':    round(head_params_kb, 1),
    })
    prev_flops = cum_flops

df = pd.DataFrame(results).set_index('Exit k')
print(df)
