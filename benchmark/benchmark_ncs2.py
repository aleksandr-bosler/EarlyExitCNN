import os
import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from openvino.inference_engine import IECore
from scipy.special import softmax
import csv
import time

MODEL1_XML    = "openvino_mobilenetv2/exit1/stage1.xml"
MODEL1_BIN    = "openvino_mobilenetv2/exit1/stage1.bin"
MODEL2_XML    = "openvino_mobilenetv2/exit2/stage2.xml"
MODEL2_BIN    = "openvino_mobilenetv2/exit2/stage2.bin"
MODEL3_XML    = "openvino_mobilenetv2/exit3/stage3.xml"
MODEL3_BIN    = "openvino_mobilenetv2/exit3/stage3.bin"

DATA_DIR      = "D:/Programming/Thesis/final_dataset"
CHANNELS      = [2, 3, 5]
TEST_FRAC     = 0.15
BATCH_SIZE    = 1

THRESHOLD1    = 1.00
THRESHOLD2    = 1.00

WARMUP_ITERS  = 10
OUTPUT_CSV    = "results_ncs2_mobilenetv2.csv"
TIMES_FILE_NAME = 'times_mobilenetv2/times.csv'


def read_ms(path):
    a = np.load(path)
    if isinstance(a, np.lib.npyio.NpzFile):
        key = "arr_0" if "arr_0" in a.files else a.files[0]
        a = a[key]
    if a.ndim == 3 and a.shape[0] == 6:
        a = np.moveaxis(a, 0, -1)
    return a.astype(np.float32)

class FireTestDS(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        x = read_ms(self.paths[idx])[..., CHANNELS]
        x = torch.from_numpy(x).permute(2,0,1).float() / 255.0
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_loader(bs):
    fire_dir = os.path.join(DATA_DIR, "fire")
    nonf_dir = os.path.join(DATA_DIR, "non_fire")
    F = sorted(os.listdir(fire_dir))
    N = sorted(os.listdir(nonf_dir))
    P = [os.path.join(fire_dir, f) for f in F] + \
        [os.path.join(nonf_dir, n) for n in N]
    L = [1]*len(F) + [0]*len(N)
    _, pt, _, lt = train_test_split(P, L, test_size=TEST_FRAC,
                                    stratify=L, random_state=42)
    return DataLoader(FireTestDS(pt, lt), batch_size=bs,
                      shuffle=False, pin_memory=True)


ie   = IECore()
net1 = ie.read_network(MODEL1_XML, MODEL1_BIN)
net2 = ie.read_network(MODEL2_XML, MODEL2_BIN)
net3 = ie.read_network(MODEL3_XML, MODEL3_BIN)

exec1 = ie.load_network(net1, device_name="MYRIAD")
exec2 = ie.load_network(net2, device_name="MYRIAD")
exec3 = ie.load_network(net3, device_name="MYRIAD")

input_name1    = list(net1.input_info.keys())[0]
output_names1  = list(net1.outputs.keys())

input_name2    = list(net2.input_info.keys())[0]
output_name2   = list(net2.outputs.keys())[0]

input_name3    = list(net3.input_info.keys())[0]
output_name3   = list(net3.outputs.keys())[0]


loader = get_loader(BATCH_SIZE)
it = iter(loader)
for _ in range(WARMUP_ITERS):
    try:
        imgs, _ = next(it)
    except StopIteration:
        break
    dummy1 = imgs.numpy()
    exec1.infer({input_name1: dummy1})
    dummy2 = np.zeros(exec2.input_info[input_name2].input_data.shape, dtype=np.float32)
    dummy3 = np.zeros(exec3.input_info[input_name3].input_data.shape, dtype=np.float32)
    exec2.infer({input_name2: dummy2})
    exec3.infer({input_name3: dummy3})


all_preds, all_probs, all_labels = [], [], []
stage1_lats, stage2_lats, stage3_lats = [], [], []
all_lats = []
head_data = {
    1:{"preds":[], "probs":[], "labels":[]},
    2:{"preds":[], "probs":[], "labels":[]},
    3:{"preds":[], "probs":[], "labels":[]},
}
exit_counts = {1:0,2:0,3:0}

times_file   = open(TIMES_FILE_NAME, 'w', newline='')
times_writer = csv.writer(times_file)
times_writer.writerow(['start','end'])


total_start = time.time()
for imgs, labels in loader:
    B       = imgs.shape[0]
    x_np    = imgs.numpy()
    y_np    = labels.numpy()
    start_ts = time.time()
    t_start = time.perf_counter()

    t0   = time.perf_counter()
    res1 = exec1.infer({input_name1: x_np})
    t1   = time.perf_counter()
    stage1_lats.append((t1-t0)/B)

    out1  = res1['logits1']
    feat2 = res1['feat2']
    feat3 = res1['feat3']

    p1 = np.max(softmax(out1, axis=1), axis=1)
    m1 = p1 >= THRESHOLD1

    preds_batch = np.zeros(B, dtype=int)
    probs_batch = np.zeros(B, dtype=float)

    idx1 = np.where(m1)[0]
    exit_counts[1] += idx1.size
    preds1 = out1[idx1].argmax(axis=1)
    probs1 = softmax(out1[idx1], axis=1)[:,1]
    for i, idx in enumerate(idx1):
        preds_batch[idx] = preds1[i]; probs_batch[idx] = probs1[i]
        head_data[1]["preds"].append(int(preds1[i]))
        head_data[1]["probs"].append(float(probs1[i]))
        head_data[1]["labels"].append(int(y_np[idx]))

    surv1 = np.where(~m1)[0]
    if surv1.size > 0:
        x2   = feat2[surv1]
        t2_0 = time.perf_counter()
        out2 = exec2.infer({input_name2: x2})[output_name2]
        t2_1 = time.perf_counter()
        stage2_lats.append((t2_1-t2_0)/surv1.size)

        p2 = np.max(softmax(out2, axis=1), axis=1)
        m2 = p2 >= THRESHOLD2

        idx2 = surv1[m2]
        exit_counts[2] += idx2.size
        preds2 = out2[m2].argmax(axis=1)
        probs2 = softmax(out2[m2], axis=1)[:,1]
        for i, idx in enumerate(idx2):
            preds_batch[idx] = preds2[i]; probs_batch[idx] = probs2[i]
            head_data[2]["preds"].append(int(preds2[i]))
            head_data[2]["probs"].append(float(probs2[i]))
            head_data[2]["labels"].append(int(y_np[idx]))

        surv2 = surv1[~m2]
        if surv2.size > 0:
            x3   = feat3[surv2]
            t3_0 = time.perf_counter()
            out3 = exec3.infer({input_name3: x3})[output_name3]
            t3_1 = time.perf_counter()
            stage3_lats.append((t3_1-t3_0)/surv2.size)

            exit_counts[3] += surv2.size
            preds3 = out3.argmax(axis=1)
            probs3 = softmax(out3, axis=1)[:,1]
            for i, idx in enumerate(surv2):
                preds_batch[idx] = preds3[i]; probs_batch[idx] = probs3[i]
                head_data[3]["preds"].append(int(preds3[i]))
                head_data[3]["probs"].append(float(probs3[i]))
                head_data[3]["labels"].append(int(y_np[idx]))

    t_end = time.perf_counter()
    all_lats.append((t_end - t_start)/B)

    end_ts = time.time()
    times_writer.writerow([start_ts, end_ts])

    all_preds.extend(preds_batch.tolist())
    all_probs.extend(probs_batch.tolist())
    all_labels.extend(y_np.tolist())

total_time = time.time() - total_start

times_file.close()

acc   = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="binary"
)
try:
    auc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auc = float("nan")
throughput = len(all_labels) / total_time

lat_all = np.array(all_lats)
mean_lat = lat_all.mean()
p95_lat  = np.percentile(lat_all, 95)
p99_lat  = np.percentile(lat_all, 99)


rows = {
    "threshold1": THRESHOLD1,
    "threshold2": THRESHOLD2,
    "count": len(all_labels),
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc,
    "throughput_fps": throughput,
    "mean_latency": mean_lat,
    "p95_latency": p95_lat,
    "p99_latency": p99_lat,
    "exit1_rate": exit_counts[1]/len(all_labels),
    "exit2_rate": exit_counts[2]/len(all_labels),
    "exit3_rate": exit_counts[3]/len(all_labels),
    "exit1_count": exit_counts[1],
    "exit2_count": exit_counts[2],
    "exit3_count": exit_counts[3],
}

if stage1_lats:
    rows.update({
        "head1_lat_mean": np.mean(stage1_lats),
        "head1_lat_p95":  np.percentile(stage1_lats,95),
        "head1_lat_p99":  np.percentile(stage1_lats,99),
    })
if stage2_lats:
    rows.update({
        "head2_lat_mean": np.mean(stage2_lats),
        "head2_lat_p95":  np.percentile(stage2_lats,95),
        "head2_lat_p99":  np.percentile(stage2_lats,99),
    })
if stage3_lats:
    rows.update({
        "head3_lat_mean": np.mean(stage3_lats),
        "head3_lat_p95":  np.percentile(stage3_lats,95),
        "head3_lat_p99":  np.percentile(stage3_lats,99),
    })

for h in (1,2,3):
    lbls  = head_data[h]["labels"]
    preds = head_data[h]["preds"]
    probs = head_data[h]["probs"]
    key   = f"head{h}_"
    if lbls:
        acc_h = accuracy_score(lbls, preds)
        prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(
            lbls, preds, average="binary", zero_division=0
        )
        try:
            auc_h = roc_auc_score(lbls, probs)
        except ValueError:
            auc_h = float("nan")
    else:
        acc_h = prec_h = rec_h = f1_h = auc_h = float("nan")
    rows.update({
        f"{key}accuracy":  acc_h,
        f"{key}precision": prec_h,
        f"{key}recall":    rec_h,
        f"{key}f1":        f1_h,
        f"{key}auc":       auc_h,
    })

df_row = pd.DataFrame([rows])
if os.path.isfile(OUTPUT_CSV):
    df_row.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
else:
    df_row.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
