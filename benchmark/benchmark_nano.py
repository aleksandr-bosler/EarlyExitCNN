"""
sudo python3 benchmark_mobilenetv2.py \
--model MobileNetV2_ee.pth \
--thr-start 0.5 --thr-stop 0.95 --thr-step 0.05 \
--out results_mobilenetv2.csv
"""

import argparse, os, time, re, subprocess, threading, signal, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch.nn as nn
from torchvision.models import mobilenet_v2

class EarlyExitMobileNetV2(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 2,
            exit_thresholds: dict = None,
            dropout: float = 0.5,
            pretrained: bool = False,
            **load_kwargs
    ):
        super().__init__()
        backbone = mobilenet_v2(pretrained=False, **load_kwargs)
        self.features = nn.Sequential(*list(backbone.features.children())[:-1])

        if in_chans != 3:
            old = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_chans,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False
            )

        c1, c2, c3 = 32, 96, 320

        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(c1, num_classes),
        )
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c2, num_classes),
        )
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )

        self.exit_thresholds = exit_thresholds or {'exit1': 0.9, 'exit2': 0.9}

    def forward(self, x: torch.Tensor, inference: bool = False):
        l1 = l2 = None
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 6:
                l1 = self.exit1(x)
                if inference:
                    p1 = torch.softmax(l1, dim=1).max(1).values
                    if (p1 >= self.exit_thresholds['exit1']).all():
                        return l1, torch.tensor(1, device=x.device)
            elif idx == 13:
                l2 = self.exit2(x)
                if inference:
                    p2 = torch.softmax(l2, dim=1).max(1).values
                    if (p2 >= self.exit_thresholds['exit2']).all():
                        return l2, torch.tensor(2, device=x.device)

        l3 = self.exit3(x)
        if inference:
            return l3, torch.tensor(3, device=x.device)

        return l1, l2, l3

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False, **load_kwargs):
        state = torch.load(checkpoint_path, **load_kwargs)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        new_state = {}
        for k, v in state.items():
            if k.startswith('backbone.'):
                new_k = k[9:]
            else:
                new_k = k
            new_state[new_k] = v
        self.load_state_dict(new_state, strict=False)

CHANNELS = [2, 3, 5]
DATA_DIR = 'dataset'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_FRAC = 0.15
WARMUP_ITERS = 20
TEGR_INTV_MS = 100

def read_ms(p):
    a = np.load(p)
    if isinstance(a, np.lib.npyio.NpzFile):
        k = 'arr_0' if 'arr_0' in a.files else a.files[0]
        a = a[k]
    if a.ndim == 3 and a.shape[0] == 6:
        a = np.moveaxis(a, 0, -1)
    return a.astype(np.float32)

class FireTestDS(Dataset):
    def __init__(self, paths, labels): self.p, self.l = paths, labels

    def __len__(self): return len(self.p)

    def __getitem__(self, i):
        x = read_ms(self.p[i])[..., CHANNELS]
        x = torch.from_numpy(x).permute(2, 0, 1) / 255.0
        y = torch.tensor(self.l[i], dtype=torch.long)
        return x, y

def get_loader():
    f = [os.path.join(DATA_DIR, 'fire', n) for n in sorted(os.listdir(os.path.join(DATA_DIR, 'fire')))]
    n = [os.path.join(DATA_DIR, 'non_fire', n) for n in sorted(os.listdir(os.path.join(DATA_DIR, 'non_fire')))]
    p, l = f + n, [1] * len(f) + [0] * len(n)
    _, pt, _, lt = train_test_split(p, l, test_size=TEST_FRAC,
                                    stratify=l, random_state=42)
    return DataLoader(FireTestDS(pt, lt), batch_size=1, shuffle=False)

_TEG_RE = re.compile(r'POM_5V_IN\s+(\d+)/')

def _run_tegrastats(buf, stop_ev):
    proc = subprocess.Popen(['tegrastats', '--interval', str(TEGR_INTV_MS)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            universal_newlines=True, preexec_fn=os.setsid)
    try:
        while not stop_ev.is_set():
            line = proc.stdout.readline()
            if not line: break
            buf.append(line)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        proc.wait()

def start_tegrastats():
    buf, ev = [], threading.Event()
    th = threading.Thread(target=_run_tegrastats, args=(buf, ev), daemon=True)
    th.start();
    time.sleep(0.1)
    return buf, ev, th

def stop_tegrastats(ev, th): ev.set(); th.join()

def parse_power(lines):
    out = []
    for ln in lines:
        m = _TEG_RE.search(ln)
        if m: out.append(int(m.group(1)) / 1000.0)
    return np.asarray(out)

def evaluate(model, dl, buf):
    it = iter(dl)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            try:
                img, _ = next(it)
            except StopIteration:
                break
            _ = model(img.to(DEVICE).half(), inference=True)
    torch.cuda.synchronize()

    idx0 = len(buf)
    exit_cnt, per_lat = {1: 0, 2: 0, 3: 0}, {1: [], 2: [], 3: []}
    all_p, all_pr, all_gt, lats = [], [], [], []
    t0g = time.time()

    with torch.no_grad():
        for img, gt in dl:
            img = img.to(DEVICE).half()
            t0 = time.perf_counter()
            logits, idx_t = model(img, inference=True)
            torch.cuda.synchronize()
            lat = time.perf_counter() - t0
            lats.append(lat)

            idx = int(idx_t.item())
            exit_cnt[idx] += 1
            per_lat[idx].append(lat)

            prob = torch.softmax(logits, 1)[0, 1].item()
            pred = int(prob >= 0.5)
            all_p.append(prob);
            all_pr.append(pred);
            all_gt.append(int(gt.item()))

    tot_time = time.time() - t0g
    power = parse_power(buf[idx0:])
    avg_pwr = power.mean() if power.size else np.nan
    lat_arr = np.asarray(lats)
    e_inf = avg_pwr * lat_arr.mean() if power.size else np.nan
    edp = e_inf * lat_arr.mean() if power.size else np.nan

    acc = accuracy_score(all_gt, all_pr)
    prec, rec, f1, _ = precision_recall_fscore_support(all_gt, all_pr,
                                                       average='binary', zero_division=0)
    auc = roc_auc_score(all_gt, all_p)

    head = {}
    for i in (1, 2, 3):
        cnt = exit_cnt[i]
        head[i] = {
            'count': cnt,
            'lat_mean': np.mean(per_lat[i]) if cnt else np.nan,
            'lat_p95': np.percentile(per_lat[i], 95) if cnt else np.nan,
        }

    return {
        'count': len(all_gt), 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'throughput_fps': len(all_gt) / tot_time,
        'avg_power_W': avg_pwr, 'energy_J': e_inf, 'EDP': edp,
        'mean_latency': lat_arr.mean(), 'p95_latency': np.percentile(lat_arr, 95),
        'p99_latency': np.percentile(lat_arr, 99),
        **{f'exit{i}_rate': exit_cnt[i] / len(all_gt) for i in (1, 2, 3)},
        **{f'exit{i}_count': exit_cnt[i] for i in (1, 2, 3)},
        **{f'head{i}_lat_mean': head[i]['lat_mean'] for i in (1, 2, 3)},
        **{f'head{i}_lat_p95': head[i]['lat_p95'] for i in (1, 2, 3)},
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--thr-start', type=float, default=0.50)
    ap.add_argument('--thr-stop', type=float, default=0.95)
    ap.add_argument('--thr-step', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    ap.add_argument('--halt', action='store_true')
    args = ap.parse_args()

    model = EarlyExitMobileNetV2(num_classes=2).to(DEVICE).half().eval()
    sd = torch.load(args.model, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("WARNING: missing keys:", missing)
        print("WARNING: unexpected keys:", unexpected)
    print("Model loaded. Total params:", sum(p.numel() for p in model.parameters()))

    dl = get_loader()
    thr_vals = np.arange(args.thr_start,
                         args.thr_stop + 1e-9, args.thr_step).round(4)

    buf, ev, th = start_tegrastats()
    rows = []
    try:
        for t1 in thr_vals:
            for t2 in thr_vals:
                model.exit_thresholds['exit1'] = t1
                model.exit_thresholds['exit2'] = t2
                print(f'>> thresholds (exit1, exit2) = ({t1:.2f}, {t2:.2f})')
                rows.append({
                    'model': os.path.basename(args.model),
                    'exit1': t1, 'exit2': t2,
                    **evaluate(model, dl, buf)
                })
                torch.cuda.empty_cache()
    finally:
        stop_tegrastats(ev, th)

    pd.DataFrame(rows).to_csv(args.out, index=False)

    if args.halt:
        subprocess.call(['shutdown', '-h', '+2', 'benchmark finished'])

