import os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

SEED       = 42
BATCH      = 16
LR         = 1.5e-4
EPOCHS     = 100
PATIENCE   = 10
CHANNELS   = [2,3,5]
exit_thresholds = {'exit1': 0.85, 'exit2': 0.85}
PERSISTENT_WORKERS = True
DATA_DIR   = '../dataset'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH  = 'MobileNetV2_100_ee.pth'

W1, W2, W3 = 0.3, 0.3, 0.4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def read_ms(path):
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        k = 'arr_0' if 'arr_0' in arr.files else arr.files[0]
        img = arr[k]
    else:
        img = arr
    if img.ndim==3 and img.shape[0]==6:
        img = np.moveaxis(img,0,-1)
    return img.astype(np.float32)

train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2(),
])
val_aug = A.Compose([
    A.Normalize(mean=[0,0,0], std=[1,1,1]),
    ToTensorV2(),
])

class FireDS(Dataset):
    def __init__(self, paths, labels, aug=None):
        self.p, self.l, self.aug = paths, labels, aug
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        x = read_ms(self.p[i])[..., CHANNELS]
        if self.aug: x = self.aug(image=x)['image']
        return x, torch.tensor(self.l[i], dtype=torch.long)

def get_loaders():
    F = sorted(os.listdir(os.path.join(DATA_DIR,'fire')))
    N = sorted(os.listdir(os.path.join(DATA_DIR,'non_fire')))
    F = [os.path.join(DATA_DIR,'fire',f) for f in F]
    N = [os.path.join(DATA_DIR,'non_fire',n) for n in N]
    P, L = F+N, [1]*len(F)+[0]*len(N)
    trp, tmp, trl, tmpl = train_test_split(P,L, test_size=0.3, stratify=L, random_state=SEED)
    vp, tp, vl, tl     = train_test_split(tmp,tmpl, test_size=0.5, stratify=tmpl, random_state=SEED)
    ds_tr = FireDS(trp, trl, train_aug)
    ds_vl = FireDS(vp,  vl,  val_aug)
    ds_te = FireDS(tp,  tl,  val_aug)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=6, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    dl_vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=6, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    return dl_tr, dl_vl, dl_te, (tp, tl)

dl_tr, dl_vl, dl_te, _ = get_loaders()

class EarlyExitMobileNetV2(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 2,
        exit_thresholds: dict = None,
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super().__init__()
        backbone = mobilenet_v2(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.features.children())[:-1])
        if in_chans != 3:
            stem_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_chans,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
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
            if idx == 4:
                l1 = self.exit1(x)
                if inference:
                    p1 = torch.softmax(l1, 1).max(1).values
                    if (p1 >= self.exit_thresholds['exit1']).all():
                        return l1, 1
            elif idx == 11:
                l2 = self.exit2(x)
                if inference:
                    p2 = torch.softmax(l2, 1).max(1).values
                    if (p2 >= self.exit_thresholds['exit2']).all():
                        return l2, 2

        l3 = self.exit3(x)
        if inference:
            return l3, 3

        return l1, l2, l3

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True

model = EarlyExitMobileNetV2(
    in_chans=len(CHANNELS),
    num_classes=2,
    exit_thresholds=None,
    dropout=0.5,
    pretrained=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_f1, no_imp = 0.0, 0

for ep in range(1, EPOCHS+1):
    model.train()
    tot_loss = 0.0
    for x,y in tqdm(dl_tr, desc=f"Train Ep{ep:02d}"):
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out1,out2,out3 = model(x, inference=False)
        loss = W1*criterion(out1,y) + W2*criterion(out2,y) + W3*criterion(out3,y)
        loss.backward(); optimizer.step()
        tot_loss += loss.item()*x.size(0)
    train_loss = tot_loss/len(dl_tr.dataset)

    model.eval()
    vl_loss, all_ps, all_gt = 0.0, [], []
    with torch.no_grad():
        for x,y in dl_vl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            _,_,logits = model(x, inference=False)
            vl_loss += criterion(logits,y).item()*x.size(0)
            ps = torch.softmax(logits,1)[:,1].cpu().numpy()
            all_ps.append(ps); all_gt.append(y.cpu().numpy())
    vl_loss /= len(dl_vl.dataset)
    all_ps = np.concatenate(all_ps)
    all_gt = np.concatenate(all_gt)
    preds = (all_ps>=0.5).astype(int)

    acc  = accuracy_score(all_gt, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_gt, preds, average='binary', zero_division=0)
    auc  = roc_auc_score(all_gt, all_ps)

    print(
        f"Ep{ep:02d} | TrL {train_loss:.4f} | VlL {vl_loss:.4f} | "
        f"Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | "
        f"F1 {f1:.3f} | AUC {auc:.3f}"
    )

    if f1 > best_f1:
        best_f1, no_imp = f1, 0
        torch.save(model.state_dict(), CKPT_PATH)
        print("Model improved, saved")
    else:
        no_imp += 1
        print(f"No improvement ({no_imp}/{PATIENCE})")
        if no_imp >= PATIENCE:
            print(f"Early stopping at epoch {ep:02d}")
            break

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.exit_thresholds = exit_thresholds
model.to(DEVICE).eval()

exit_counts = {1: 0, 2: 0, 3: 0}

all_probs = []
all_preds = []
all_gts   = []

per_exit = {
    1: {'probs': [], 'preds': [], 'gts': []},
    2: {'probs': [], 'preds': [], 'gts': []},
    3: {'probs': [], 'preds': [], 'gts': []},
}

with torch.no_grad():
    for x_batch, y_batch in dl_te:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        for i in range(x_batch.size(0)):
            x = x_batch[i:i+1]
            y = int(y_batch[i].item())

            logits, exit_idx = model(x, inference=True)
            prob = torch.softmax(logits, dim=1)[0,1].item()
            pred = int(prob >= 0.5)

            all_probs.append(prob)
            all_preds.append(pred)
            all_gts.append(y)

            exit_counts[exit_idx] += 1
            per_exit[exit_idx]['probs'].append(prob)
            per_exit[exit_idx]['preds'].append(pred)
            per_exit[exit_idx]['gts'].append(y)


acc_all  = accuracy_score(all_gts, all_preds)
prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(
    all_gts, all_preds, average='binary', zero_division=0
)
auc_all  = roc_auc_score(all_gts, all_probs)
cm_all   = confusion_matrix(all_gts, all_preds)

print("=== Early-exit evaluation ===")
print(f"Thresholds: exit1={exit_thresholds['exit1']}, exit2={exit_thresholds['exit2']}")
print(f"Exit counts: {exit_counts} (from {len(all_gts)} examples)\n")

print("--- Overall metrics (from all examples) ---")
print(f"Accuracy : {acc_all:.4f}")
print(f"Precision: {prec_all:.4f}")
print(f"Recall   : {rec_all:.4f}")
print(f"F1-score : {f1_all:.4f}")
print(f"AUC      : {auc_all:.4f}")
print("Confusion matrix:")
print(cm_all, "\n")

for exit_idx in (1, 2, 3):
    probs = per_exit[exit_idx]['probs']
    preds = per_exit[exit_idx]['preds']
    gts   = per_exit[exit_idx]['gts']

    acc = accuracy_score(gts, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        gts, preds, average='binary', zero_division=0
    )
    auc = roc_auc_score(gts, probs) if len(set(gts)) > 1 else float('nan')
    cm  = confusion_matrix(gts, preds)

    print(f"--- Head {exit_idx} (out of {len(gts)} examples) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("Confusion matrix:")
    print(cm, "\n")
