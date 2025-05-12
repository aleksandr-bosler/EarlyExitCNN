import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'

import torch
import torch.nn as nn
import timm

class EarlyExitMobileNetV2(nn.Module):
    def __init__(self,
        backbone_name='mobilenetv2_100',
        in_chans=3,
        num_classes=2,
        exit_thresholds=None,
        dropout=0.5
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=in_chans,
            features_only=True
        )
        chs = [f['num_chs'] for f in self.backbone.feature_info.get_dicts()]
        c1, c2, c3 = chs[2], chs[3], chs[-1]

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(c1, num_classes)
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c2, num_classes)
        )
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes)
        )
        self.exit_thresholds = exit_thresholds or {'exit1':0.9,'exit2':0.9}

    def forward(self, x, inference=False):
        feats = self.backbone(x)
        l1 = self.exit1(feats[2])
        l2 = self.exit2(feats[3])
        l3 = self.exit3(feats[-1])

        if not inference:
            return l1, l2, l3

        p1,_ = torch.softmax(l1,1).max(1)
        if (p1>=self.exit_thresholds['exit1']).all(): return l1,1
        p2,_ = torch.softmax(l2,1).max(1)
        if (p2>=self.exit_thresholds['exit2']).all(): return l2,2
        return l3,3

PTH_WEIGHTS = 'MobileNetV2_ee.pth'
full = EarlyExitMobileNetV2().eval().half()
full.load_state_dict(torch.load(PTH_WEIGHTS, map_location='cpu'))

with torch.no_grad():
    x_dummy = torch.randn(1,3,224,224).half()
    feats   = full.backbone(x_dummy)
c1, c2, c3 = feats[2].shape[1], feats[3].shape[1], feats[-1].shape[1]
print(f"Detected feat-channels: c1={c1}, c2={c2}, c3={c3}")

class Stage1(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.backbone = full_model.backbone
        self.exit1    = full_model.exit1

    def forward(self, x):
        feats = self.backbone(x)
        return self.exit1(feats[2]), feats[3], feats[-1]

class Stage2(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.exit2 = full_model.exit2

    def forward(self, feat2):
        return self.exit2(feat2)

class Stage3(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.exit3 = full_model.exit3

    def forward(self, feat3):
        return self.exit3(feat3)

stage1 = Stage1(full).eval().half()
stage2 = Stage2(full).eval().half()
stage3 = Stage3(full).eval().half()

OUT_DIR = 'mobilev2_onnx'
os.makedirs(OUT_DIR, exist_ok=True)

def export(net, dummy, in_names, out_names):
    path = os.path.join(OUT_DIR, f"{net.__class__.__name__}.onnx")
    dyn_axes = {n:{0:'batch'} for n in in_names+out_names}
    torch.onnx.export(
        net, (dummy,), path,
        export_params=True, opset_version=13,
        input_names=in_names, output_names=out_names,
        dynamic_axes=dyn_axes, do_constant_folding=True,
    )
    print(f"✅ {os.path.basename(path)}  dummy={tuple(dummy.shape)} → {out_names}")

export(
    stage1,
    torch.randn(1,3,224,224).half(),
    ['input'],
    ['logits1','feat2','feat3']
)

export(
    stage2,
    torch.randn(1,c2, feats[3].shape[2], feats[3].shape[3]).half(),
    ['feat2'],
    ['logits2']
)

export(
    stage3,
    torch.randn(1,c3, feats[-1].shape[2], feats[-1].shape[3]).half(),
    ['feat3'],
    ['logits3']
)
