import os, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from torchvision.models import mobilenet_v2
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

SEED       = 42
BATCH      = 16
LR         = 1.5e-4
EPOCHS     = 100
PATIENCE   = 15
CHANNELS   = [2,3,5]
NUM_WORKERS = 8
PERSISTENT_WORKERS = True
DATA_DIR   = '../final_dataset'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH  = 'MobileNetV2_no_ee.pth'

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
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    dl_vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=PERSISTENT_WORKERS)
    return dl_tr, dl_vl, dl_te, (tp, tl)

dl_tr, dl_vl, dl_te, _ = get_loaders()

class MobileNetV2NoEE(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 2,
    ):
        super().__init__()
        self.model = mobilenet_v2(pretrained=False, num_classes=num_classes)

        if in_chans != 3:
            old_conv = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_chans,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )

    def forward(self, x):
        return self.model(x)

model = MobileNetV2NoEE(
    in_chans=len(CHANNELS),
    num_classes=2
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_f1, no_imp = 0.0, 0

for ep in range(1, EPOCHS+1):
    model.train()
    tot_loss = 0.0
    for x,y in tqdm(dl_tr, desc=f"Train Ep{ep:02d}"):
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()*x.size(0)
    train_loss = tot_loss/len(dl_tr.dataset)

    model.eval()
    vl_loss, all_ps, all_gt = 0.0, [], []
    with torch.no_grad():
        for x,y in dl_vl:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
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
model.eval()

test_loss, all_ps, all_gt = 0.0, [], []
with torch.no_grad():
    for x,y in dl_te:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        test_loss += criterion(logits,y).item()*x.size(0)
        ps = torch.softmax(logits,1)[:,1].cpu().numpy()
        all_ps.append(ps); all_gt.append(y.cpu().numpy())

test_loss /= len(dl_te.dataset)
all_ps = np.concatenate(all_ps)
all_gt = np.concatenate(all_gt)
preds  = (all_ps>=0.5).astype(int)

acc  = accuracy_score(all_gt, preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_gt, preds, average='binary', zero_division=0)
auc  = roc_auc_score(all_gt, all_ps)

print("\n— Final Test —")
print(f"Loss {test_loss:.4f} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")
print("Confusion Matrix:\n", confusion_matrix(all_gt, preds))
