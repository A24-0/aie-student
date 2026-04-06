"""Generate HW10-11, HW12, HW13, HW14 notebooks + report stubs. Run: python homeworks/generate_remaining_hw.py"""
import os
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = os.path.dirname(os.path.abspath(__file__))


def save_nb(rel_path, cells):
    path = os.path.join(ROOT, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nb = new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }
    )
    nb["cells"] = cells
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Wrote", path)


def hw1011():
    code = textwrap.dedent(
        r'''
import os, json, csv, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms, models
from torchvision.datasets import STL10, VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

# --- cwd: HW10-11 ---
ROOT = Path.cwd().resolve()
if ROOT.name != "HW10-11":
    cand = ROOT / "homeworks" / "HW10-11"
    if cand.is_dir():
        os.chdir(cand)
ART = Path("artifacts")
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = "data"
DATASET_A = "STL10"
NUM_CLASSES = 10
BATCH = 32

# STL10: train 5000, test 8000 — val из train 80/20
base_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
aug_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=8),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
imagenet_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
imagenet_aug_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_full = STL10(DATA_ROOT, split="train", download=True, transform=base_tf)
test_ds = STL10(DATA_ROOT, split="test", download=True, transform=base_tf)
n_val = int(0.2 * len(train_full))
n_tr = len(train_full) - n_val
g = torch.Generator().manual_seed(SEED)
train_base, val_base = random_split(train_full, [n_tr, n_val], generator=g)

train_full_aug = STL10(DATA_ROOT, split="train", download=True, transform=aug_tf)
train_aug = Subset(train_full_aug, train_base.indices)

test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
val_loader_base = DataLoader(val_base, batch_size=BATCH, shuffle=False, num_workers=0)

def loaders_for_c12(aug_train: bool):
    tr = DataLoader(train_aug if aug_train else train_base, batch_size=BATCH, shuffle=True, num_workers=0)
    return tr, val_loader_base

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256 * 12 * 12, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))

    def forward(self, x):
        return self.fc(self.f(x))

def train_one_epoch(model, loader, crit, opt):
    model.train()
    tot_l = tot = cor = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        tot_l += loss.item() * x.size(0)
        tot += x.size(0)
        cor += (logits.argmax(1) == y).float().sum().item()
    return tot_l / tot, cor / tot

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tot_l = tot = cor = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        tot_l += loss.item() * x.size(0)
        tot += x.size(0)
        cor += (logits.argmax(1) == y).float().sum().item()
    return tot_l / tot, cor / tot

def train_cnn_epochs(model, train_loader, epochs, tag):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist = {k: [] for k in ("tl", "ta", "vl", "va")}
    best_va, best_state = 0.0, None
    for ep in range(epochs):
        tl, ta = train_one_epoch(model, train_loader, crit, opt)
        vl, va = evaluate(model, val_loader_base, crit)
        for k, v in zip(hist.keys(), (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best_va:
            best_va = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, epochs // 3) == 0:
            print(tag, "ep", ep + 1, "val_acc", va)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_va, min(hist["vl"]), hist

EPOCHS_CNN = 12

# C1
print("C1 simple CNN, no aug")
tr, _ = loaders_for_c12(False)
m1 = SmallCNN().to(device)
c1_acc, c1_loss, h1 = train_cnn_epochs(m1, tr, EPOCHS_CNN, "C1")

# C2
print("C2 simple CNN + aug")
tr2, _ = loaders_for_c12(True)
m2 = SmallCNN().to(device)
c2_acc, c2_loss, h2 = train_cnn_epochs(m2, tr2, EPOCHS_CNN, "C2")

# --- ResNet18: train/val/test с imagenet transforms ---
train_rn = STL10(DATA_ROOT, split="train", download=True, transform=imagenet_aug_tf)
val_rn = STL10(DATA_ROOT, split="train", download=True, transform=imagenet_tf)
test_rn = STL10(DATA_ROOT, split="test", download=True, transform=imagenet_tf)
train_rn_s = Subset(train_rn, train_base.indices)
val_rn_s = Subset(val_rn, val_base.indices)
tr_rn = DataLoader(train_rn_s, batch_size=BATCH, shuffle=True, num_workers=0)
val_rn_loader = DataLoader(val_rn_s, batch_size=BATCH, shuffle=False, num_workers=0)
test_rn_loader = DataLoader(test_rn, batch_size=BATCH, shuffle=False, num_workers=0)

weights = models.ResNet18_Weights.DEFAULT
def make_resnet():
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(device)

def train_resnet(model, freeze_backbone: bool, epochs, tag, lr=1e-3):
    crit = nn.CrossEntropyLoss()
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(model.fc.parameters(), lr=lr)
    else:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if n.startswith("layer4") or n.startswith("fc"):
                p.requires_grad = True
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr * 0.1)
    hist = {k: [] for k in ("tl", "ta", "vl", "va")}
    best_va, best_state = 0.0, None
    for ep in range(epochs):
        tl, ta = train_one_epoch(model, tr_rn, crit, opt)
        vl, va = evaluate(model, val_rn_loader, crit)
        for k, v in zip(hist.keys(), (tl, ta, vl, va)):
            hist[k].append(v)
        if va > best_va:
            best_va = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(tag, "ep", ep + 1, "val_acc", va)
    if best_state:
        model.load_state_dict(best_state)
    return best_va, min(hist["vl"]), hist

EPOCHS_RN = 8
print("C3 ResNet18 head only")
m3 = make_resnet()
c3_acc, c3_loss, h3 = train_resnet(m3, True, EPOCHS_RN, "C3")

print("C4 ResNet18 partial finetune layer4+fc")
m4 = make_resnet()
c4_acc, c4_loss, h4 = train_resnet(m4, False, EPOCHS_RN, "C4")

best_tag = max(
    [("C1", c1_acc), ("C2", c2_acc), ("C3", c3_acc), ("C4", c4_acc)],
    key=lambda x: x[1],
)[0]
models_map = {"C1": (m1, "SmallCNN", h1), "C2": (m2, "SmallCNN", h2), "C3": (m3, "ResNet18 head", h3), "C4": (m4, "ResNet18 finetune", h4)}
best_model, best_name, best_hist = models_map[best_tag]

crit = nn.CrossEntropyLoss()
if best_tag in ("C1", "C2"):
    test_loader_final = test_loader
else:
    best_model = best_model  # already resnet
    test_loader_final = test_rn_loader

@torch.no_grad()
def acc_on_loader(model, loader):
    model.eval()
    tot = cor = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cor += (model(x).argmax(1) == y).sum().item()
        tot += y.size(0)
    return cor / tot

if best_tag in ("C1", "C2"):
    test_acc = acc_on_loader(best_model, test_loader)
else:
    test_acc = acc_on_loader(best_model, test_rn_loader)

print("Best by val:", best_tag, "test_acc", test_acc)

torch.save(best_model.state_dict(), ART / "best_classifier.pt")
with open(ART / "best_classifier_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": DATASET_A,
        "seed": SEED,
        "best_experiment": best_tag,
        "best_val_accuracy": float(max(c1_acc, c2_acc, c3_acc, c4_acc)),
        "test_accuracy": float(test_acc),
        "model": best_name,
    }, f, indent=2)

# figures
fig, ax = plt.subplots()
ax.bar(["C1", "C2", "C3", "C4"], [c1_acc, c2_acc, c3_acc, c4_acc])
ax.set_ylabel("best val acc (reload per run)")
ax.set_title("STL10 classifiers")
plt.tight_layout()
plt.savefig(FIG / "classification_compare.png", dpi=120)
plt.close()

# curves best (use hist of winner)
h_best = best_hist
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ep = range(1, len(h_best["tl"]) + 1)
ax[0].plot(ep, h_best["tl"], label="train")
ax[0].plot(ep, h_best["vl"], label="val")
ax[0].set_title("Loss " + best_tag)
ax[0].legend()
ax[1].plot(ep, h_best["ta"], label="train")
ax[1].plot(ep, h_best["va"], label="val")
ax[1].set_title("Acc " + best_tag)
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG / "classification_curves_best.png", dpi=120)
plt.close()

# aug preview
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
it = iter(DataLoader(train_aug, batch_size=1, shuffle=True))
for i in range(8):
    x, _ = next(it)
    axes.flat[i].imshow(x[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axes.flat[i].axis("off")
plt.suptitle("Augmented STL10 samples")
plt.tight_layout()
plt.savefig(FIG / "augmentations_preview.png", dpi=120)
plt.close()

# ========= Part B: VOC Segmentation (FCN-ResNet50, те же классы PASCAL) =========
seg_w = FCN_ResNet50_Weights.DEFAULT
seg_model = fcn_resnet50(weights=seg_w).to(device)
seg_model.eval()
seg_pre = seg_w.transforms()

voc_root = Path(DATA_ROOT) / "voc"
voc_val = VOCSegmentation(str(voc_root), year="2012", image_set="val", download=True)


def miou_np(pred, tgt):
    pred = pred.reshape(-1)
    tgt = tgt.reshape(-1)
    m = tgt != 255
    pred, tgt = pred[m], tgt[m]
    ious = []
    for c in range(21):
        p = pred == c
        t = tgt == c
        u = (p | t).sum()
        if u > 0:
            ious.append((p & t).sum() / u)
    return float(np.mean(ious)) if ious else 0.0


def pixel_pr(pred, tgt):
    m = tgt != 255
    pred, tgt = pred[m], tgt[m]
    return float((pred == tgt).mean())


def eval_seg(n_img=30, postprocess=None):
    rng = np.random.default_rng(SEED)
    ix = rng.choice(len(voc_val), size=min(n_img, len(voc_val)), replace=False)
    m_ious, pxs = [], []
    for i in ix:
        img, mask = voc_val[int(i)]
        mask = np.array(mask, dtype=np.int64)
        inp = seg_pre(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = seg_model(inp)["out"]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
        if postprocess is not None:
            pred = postprocess(pred)
        m_ious.append(miou_np(pred, mask))
        pxs.append(pixel_pr(pred, mask))
    return float(np.mean(m_ious)), float(np.mean(pxs))


def post_median(pred):
    from scipy.ndimage import median_filter
    return median_filter(pred, size=5)


m1, px1 = eval_seg(postprocess=None)
m2, px2 = eval_seg(postprocess=post_median)
print("V1 argmax mIoU", m1, "pixel_acc", px1)
print("V2 median-filter mIoU", m2, "pixel_acc", px2)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for k in range(4):
    img, mask = voc_val[k]
    mask = np.array(mask)
    inp = seg_pre(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(seg_model(inp)["out"], dim=1).squeeze(0).cpu().numpy()
    axes[0, k].imshow(np.array(img))
    axes[0, k].set_title("image")
    axes[0, k].axis("off")
    axes[1, k].imshow(pred, vmin=0, vmax=20)
    axes[1, k].set_title("pred")
    axes[1, k].axis("off")
plt.tight_layout()
plt.savefig(FIG / "segmentation_examples.png", dpi=110)
plt.close()

fig, ax = plt.subplots()
ax.bar(["V1 mIoU", "V2 mIoU", "V1 pix", "V2 pix"], [m1, m2, px1, px2])
ax.set_ylim(0, 1)
plt.title("Segmentation metrics (val subset)")
plt.tight_layout()
plt.savefig(FIG / "segmentation_metrics.png", dpi=120)
plt.close()

rows = [
    {"experiment_id": "C1", "task": "classification", "dataset": DATASET_A, "seed": SEED,
     "model_summary": "SmallCNN no aug", "optimizer": "Adam", "lr": 1e-3, "epochs_trained": EPOCHS_CNN,
     "best_val_accuracy": c1_acc, "test_accuracy": float(test_acc) if best_tag == "C1" else "", "precision": "", "recall": "", "mean_iou": "", "notes": ""},
    {"experiment_id": "C2", "task": "classification", "dataset": DATASET_A, "seed": SEED,
     "model_summary": "SmallCNN aug", "optimizer": "Adam", "lr": 1e-3, "epochs_trained": EPOCHS_CNN,
     "best_val_accuracy": c2_acc, "test_accuracy": float(test_acc) if best_tag == "C2" else "", "precision": "", "recall": "", "mean_iou": "", "notes": ""},
    {"experiment_id": "C3", "task": "classification", "dataset": DATASET_A, "seed": SEED,
     "model_summary": "ResNet18 head", "optimizer": "Adam", "lr": 1e-3, "epochs_trained": EPOCHS_RN,
     "best_val_accuracy": c3_acc, "test_accuracy": float(test_acc) if best_tag == "C3" else "", "precision": "", "recall": "", "mean_iou": "", "notes": ""},
    {"experiment_id": "C4", "task": "classification", "dataset": DATASET_A, "seed": SEED,
     "model_summary": "ResNet18 layer4+fc", "optimizer": "Adam", "lr": 1e-4, "epochs_trained": EPOCHS_RN,
     "best_val_accuracy": c4_acc, "test_accuracy": float(test_acc) if best_tag == "C4" else "", "precision": "", "recall": "", "mean_iou": "", "notes": "test только для лучшего по val"},
    {"experiment_id": "V1", "task": "segmentation", "dataset": "VOC2012-seg", "seed": SEED,
     "model_summary": "FCN_ResNet50 VOC weights", "optimizer": "", "lr": "", "epochs_trained": 0,
     "best_val_accuracy": "", "test_accuracy": "", "precision": px1, "recall": "", "mean_iou": m1, "notes": "argmax"},
    {"experiment_id": "V2", "task": "segmentation", "dataset": "VOC2012-seg", "seed": SEED,
     "model_summary": "FCN_ResNet50 VOC weights", "optimizer": "", "lr": "", "epochs_trained": 0,
     "best_val_accuracy": "", "test_accuracy": "", "precision": px2, "recall": "", "mean_iou": m2, "notes": "median filter 5x5"},
]
with open(ART / "runs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("Done HW10-11")
'''
    ).strip()
    cells = [
        new_markdown_cell(
            "# HW10-11: CNN, transfer learning, сегментация (VOC)\n\n"
            "Часть A: STL10 — C1…C4. Часть B: Pascal VOC **segmentation**, FCN-ResNet50 (V1 argmax, V2 median post).\n"
            "Запуск из каталога `homeworks/HW10-11/`."
        ),
        new_code_cell(code),
    ]
    save_nb("HW10-11/HW10-11.ipynb", cells)


def hw12():
    code = textwrap.dedent(
        r'''
import os, json, csv, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd().resolve()
if ROOT.name != "HW12":
    cand = ROOT / "homeworks" / "HW12"
    if cand.is_dir():
        os.chdir(cand)
ART = Path("artifacts")
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = Path("S12-hw-dataset.csv")
df = pd.read_csv(csv_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
print(df.head(), len(df))

# temporal split 70/15/15
n = len(df)
n_test = int(n * 0.15)
n_val = int(n * 0.15)
n_train = n - n_val - n_test
tr = df.iloc[:n_train].copy()
va = df.iloc[n_train : n_train + n_val].copy()
te = df.iloc[n_train + n_val :].copy()
print("split", n_train, len(va), len(te))

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(tr["date"], tr["target"], label="train")
ax.plot(va["date"], va["target"], label="val")
ax.plot(te["date"], te["target"], label="test")
ax.legend()
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(FIG / "series_split.png", dpi=120)
plt.close()

# features without leakage
def add_features(d):
    d = d.copy()
    d["lag_1"] = d["target"].shift(1)
    d["lag_7"] = d["target"].shift(7)
    d["lag_14"] = d["target"].shift(14)
    d["rolling_mean_7"] = d["target"].shift(1).rolling(7).mean()
    d["rolling_std_7"] = d["target"].shift(1).rolling(7).std()
    d["dow"] = d["date"].dt.dayofweek
    return d

full = add_features(df)
tr_f = full.iloc[:n_train].dropna()
va_f = full.iloc[n_train : n_train + n_val].dropna()
te_f = full.iloc[n_train + n_val :].dropna()
feat_cols = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "dow"]
Xtr, ytr = tr_f[feat_cols].values, tr_f["target"].values
Xva, yva = va_f[feat_cols].values, va_f["target"].values
Xte, yte = te_f[feat_cols].values, te_f["target"].values

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xva_s = scaler.transform(Xva)
Xte_s = scaler.transform(Xte)

def metrics(y, yhat):
    mae = np.mean(np.abs(y - yhat))
    rmse = math.sqrt(np.mean((y - yhat) ** 2))
    mape = np.mean(np.abs((y - yhat) / (np.abs(y) + 1e-8))) * 100
    return mae, rmse, mape

# B1 naive last
y_va_hat_b1 = va_f["lag_1"].values
y_te_hat_b1 = te_f["lag_1"].values
b1_va = metrics(yva, y_va_hat_b1)
b1_te = metrics(yte, y_te_hat_b1)

# B2 moving average window 7 on shifted series
def ma_predict(series_vals, window):
    out = []
    for i in range(len(series_vals)):
        if i < window:
            out.append(series_vals[max(0, i - 1)] if i > 0 else series_vals[0])
        else:
            out.append(np.mean(series_vals[i - window : i]))
    return np.array(out)

# align: use train statistics for val/test — apply on full target series positions
full_target = df["target"].values
# simpler: val predictions = lag_1 as proxy OR compute rolling mean from past only in loop
window = 24
y_va_b2 = np.array([full_target[n_train + i - 1] if i == 0 else np.mean(full_target[n_train + i - window : n_train + i]) for i in range(len(va))])
# fix: use explicit loop from history
hist = full_target[: n_train]
va_preds = []
for i in range(len(va)):
    idx = n_train + i
    w = full_target[max(0, idx - window) : idx]
    va_preds.append(np.mean(w) if len(w) else full_target[idx - 1])
y_va_b2 = np.array(va_preds)
y_te_b2 = []
for i in range(len(te)):
    idx = n_train + n_val + i
    w = full_target[max(0, idx - window) : idx]
    y_te_b2.append(np.mean(w) if len(w) else full_target[idx - 1])
y_te_b2 = np.array(y_te_b2)
b2_va = metrics(yva, y_va_b2)
b2_te = metrics(yte, y_te_b2)

# B3 Ridge
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(Xtr_s, ytr)
y_va_b3 = ridge.predict(Xva_s)
y_te_b3 = ridge.predict(Xte_s)
b3_va = metrics(yva, y_va_b3)
b3_te = metrics(yte, y_te_b3)

# R1 GRU window
WINDOW = 48
HIDDEN = 64
series = df["target"].values.astype(np.float32)

class SeqDS(Dataset):
    def __init__(self, arr, start, end):
        self.data = arr[start:end]
    def __len__(self):
        return max(0, len(self.data) - WINDOW - 1)
    def __getitem__(self, i):
        x = self.data[i : i + WINDOW]
        y = self.data[i + WINDOW]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y)

# scale series for GRU
mu, sig = series[:n_train].mean(), series[:n_train].std() + 1e-8
series_n = (series - mu) / sig

tr_ds = SeqDS(series_n, 0, n_train)
va_ds = SeqDS(series_n, 0, n_train + n_val)
# validation windows that fall inside val region only
class ValDS(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return n_val - WINDOW - 1
    def __getitem__(self, i):
        idx = n_train + i
        x = series_n[idx - WINDOW : idx]
        y = series_n[idx]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y)

va_gru = ValDS()
tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
va_loader = DataLoader(va_gru, batch_size=128, shuffle=False)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1, :]).squeeze(-1)

def train_gru():
    m = GRUModel().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    best = 1e9
    hist = {"tl": [], "vl": []}
    for ep in range(40):
        m.train()
        tl = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = m(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(tr_loader.dataset)
        m.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = m(xb)
                vl += crit(pred, yb).item() * xb.size(0)
        vl /= max(1, len(va_loader.dataset))
        hist["tl"].append(tl)
        hist["vl"].append(vl)
        if vl < best:
            best = vl
            torch.save(m.state_dict(), ART / "best_gru.pt")
        if ep % 10 == 0:
            print("ep", ep, "val mse", vl)
    return hist

h_gru = train_gru()
m = GRUModel().to(device)
m.load_state_dict(torch.load(ART / "best_gru.pt", map_location=device))

def gru_predict_on(start, end):
    preds = []
    for idx in range(start + WINDOW, end):
        x = torch.from_numpy(series_n[idx - WINDOW : idx]).float().unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            p = m(x).item()
        preds.append(p * sig + mu)
    return np.array(preds)

y_va_r = gru_predict_on(n_train, n_train + n_val)
y_va_true = series[n_train + WINDOW : n_train + n_val]
b_r_va = metrics(y_va_true, y_va_r)

y_te_r = gru_predict_on(n_train + n_val, n)
y_te_true = series[n_train + n_val + WINDOW :]
b_r_te = metrics(y_te_true, y_te_r)

# choose best by val MAE among B1,B2,B3,R1
cands = {
    "B1": (b1_va[0], "naive-last"),
    "B2": (b2_va[0], "ma"),
    "B3": (b3_va[0], "ridge"),
    "R1": (b_r_va[0], "gru"),
}
best_id = min(cands, key=lambda k: cands[k][0])
print("best by val MAE", best_id, cands[best_id])

fig, ax = plt.subplots()
ax.plot(["B1", "B2", "B3", "R1"], [b1_va[0], b2_va[0], b3_va[0], b_r_va[0]], marker="o")
ax.set_ylabel("Val MAE")
plt.title("Baselines vs GRU")
plt.tight_layout()
plt.savefig(FIG / "baselines_compare.png", dpi=120)
plt.close()

fig, ax = plt.subplots()
ax.plot(h_gru["tl"], label="train")
ax.plot(h_gru["vl"], label="val")
ax.legend()
plt.savefig(FIG / "gru_learning_curves.png", dpi=120)
plt.close()

# best forecast on test — use best model logic
if best_id == "R1":
    y_hat_plot = y_te_r
    y_true_plot = y_te_true
else:
    y_hat_plot = {"B1": y_te_hat_b1, "B2": y_te_b2, "B3": y_te_b3}[best_id]
    y_true_plot = yte[-len(y_hat_plot):] if len(y_hat_plot) != len(yte) else yte

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(y_true_plot[:500], label="true")
ax.plot(y_hat_plot[:500], label="pred")
ax.legend()
plt.savefig(FIG / "best_forecast_test.png", dpi=120)
plt.close()

rows = [
    {"experiment_id": "B1", "task": "forecasting", "dataset": "S12-hw-dataset.csv", "seed": SEED,
     "split_summary": "70/15/15 time", "window_size": "", "horizon": 1,
     "model_summary": "naive last", "features_summary": "lag_1", "scaler": "",
     "optimizer": "", "lr": "", "epochs_trained": "", "best_val_mae": b1_va[0], "best_val_rmse": b1_va[1], "best_val_mape": b1_va[2],
     "test_mae": b1_te[0], "test_rmse": b1_te[1], "test_mape": b1_te[2], "notes": ""},
    {"experiment_id": "B2", "task": "forecasting", "dataset": "S12-hw-dataset.csv", "seed": SEED,
     "split_summary": "70/15/15 time", "window_size": window, "horizon": 1,
     "model_summary": f"MA({window})", "features_summary": "past window mean", "scaler": "",
     "optimizer": "", "lr": "", "epochs_trained": "", "best_val_mae": b2_va[0], "best_val_rmse": b2_va[1], "best_val_mape": b2_va[2],
     "test_mae": b2_te[0], "test_rmse": b2_te[1], "test_mape": b2_te[2], "notes": ""},
    {"experiment_id": "B3", "task": "forecasting", "dataset": "S12-hw-dataset.csv", "seed": SEED,
     "split_summary": "70/15/15 time", "window_size": "", "horizon": 1,
     "model_summary": "Ridge", "features_summary": ",".join(feat_cols), "scaler": "StandardScaler train",
     "optimizer": "", "lr": "", "epochs_trained": "", "best_val_mae": b3_va[0], "best_val_rmse": b3_va[1], "best_val_mape": b3_va[2],
     "test_mae": b3_te[0], "test_rmse": b3_te[1], "test_mape": b3_te[2], "notes": ""},
    {"experiment_id": "R1", "task": "forecasting", "dataset": "S12-hw-dataset.csv", "seed": SEED,
     "split_summary": "70/15/15 time", "window_size": WINDOW, "horizon": 1,
     "model_summary": f"GRU hidden={HIDDEN}", "features_summary": "scaled target window", "scaler": "z-score train",
     "optimizer": "Adam", "lr": 1e-3, "epochs_trained": 40, "best_val_mae": b_r_va[0], "best_val_rmse": b_r_va[1], "best_val_mape": b_r_va[2],
     "test_mae": b_r_te[0], "test_rmse": b_r_te[1], "test_mape": b_r_te[2], "notes": ""},
]
with open(ART / "runs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with open(ART / "best_gru_config.json", "w", encoding="utf-8") as f:
    json.dump({"window_size": WINDOW, "hidden": HIDDEN, "seed": SEED, "epochs": 40}, f, indent=2)

print("HW12 done")
'''
    ).strip()
    cells = [
        new_markdown_cell("# HW12: временные ряды, temporal split, GRU\n\nФайл `S12-hw-dataset.csv` в этой папке."),
        new_code_cell(code),
    ]
    save_nb("HW12/HW12.ipynb", cells)


def hw13():
    code = textwrap.dedent(
        r'''
import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

ROOT = Path.cwd().resolve()
if ROOT.name != "HW13":
    cand = ROOT / "homeworks" / "HW13"
    if cand.is_dir():
        os.chdir(cand)
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = load_dataset("emotion")
label_names = ds["train"].features["label"].names
num_labels = len(label_names)

def stratified_split(examples, val_ratio=0.1):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(examples["text"]))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_ratio)
    val_idx = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]
    return ds["train"].select(train_idx), ds["train"].select(list(val_idx))

train_ds, val_ds = stratified_split(ds["train"])
test_ds = ds["test"]
print(len(train_ds), len(val_ds), len(test_ds))

model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)

def tok_fn(batch):
    return tok(batch["text"], padding=True, truncation=True, max_length=128)

train_t = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
val_t = val_ds.map(tok_fn, batched=True, remove_columns=["text"])
test_t = test_ds.map(tok_fn, batched=True, remove_columns=["text"])
train_t.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_t.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_t.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
# quick pretrained inference demo (before fine-tune)
sample_texts = test_ds["text"][:5]
inputs = tok(sample_texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    logits = base_model(**inputs).logits
print("pretrained logits shape", logits.shape)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}

args = TrainingArguments(
    output_dir="emotion_out",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    seed=SEED,
    load_best_model_at_end=False,
)

trainer = Trainer(model=model, args=args, train_dataset=train_t, eval_dataset=val_t, compute_metrics=compute_metrics)
trainer.train()
preds = trainer.predict(test_t)
y_hat = np.argmax(preds.predictions, axis=-1)
y_true = np.array(test_ds["label"])
acc = accuracy_score(y_true, y_hat)
f1 = f1_score(y_true, y_hat, average="macro")
print("test acc", acc, "f1", f1)

cm = confusion_matrix(y_true, y_hat)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("pred")
plt.ylabel("true")
plt.tight_layout()
plt.savefig(ART / "confusion_matrix.png", dpi=120)
plt.close()

rows = []
for i in range(min(30, len(test_ds))):
    rows.append({
        "text": test_ds["text"][i][:500],
        "true_label": label_names[y_true[i]],
        "pred_label": label_names[y_hat[i]],
        "confidence": float(np.max(torch.softmax(torch.tensor(preds.predictions[i]), dim=-1))),
    })
pd.DataFrame(rows).to_csv(ART / "sample_predictions.csv", index=False)
print("saved artifacts")
'''
    ).strip()
    cells = [
        new_markdown_cell("# HW13: emotion classification, DistilBERT\n\nДатасет `emotion` из HuggingFace."),
        new_code_cell(code),
    ]
    save_nb("HW13/HW13.ipynb", cells)


def hw14():
    code = textwrap.dedent(
        r'''
import os, re, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path.cwd().resolve()
if ROOT.name != "HW14":
    cand = ROOT / "homeworks" / "HW14"
    if cand.is_dir():
        os.chdir(cand)
ART = Path("artifacts")
ART.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# small synthetic knowledge base (10-30 docs)
docs = [
    {"id": "d1", "text": "Курс AIE: модуль PyTorch вводит тензоры, autograd и nn.Module."},
    {"id": "d2", "text": "Регуляризация: dropout и weight decay уменьшают переобучение."},
    {"id": "d3", "text": "Оптимизатор Adam адаптивно масштабирует шаги по параметрам."},
    {"id": "d4", "text": "Сверточные сети используют локальные фильтры на изображениях."},
    {"id": "d5", "text": "ResNet использует остаточные связи для обучения глубоких сетей."},
    {"id": "d6", "text": "FAISS строит индекс для быстрого поиска ближайших векторов."},
    {"id": "d7", "text": "Mini-RAG: retrieval контекста затем генерация ответа."},
    {"id": "d8", "text": "Временные ряды требуют split по времени, не случайного перемешивания."},
    {"id": "d9", "text": "BERT использует self-attention для контекстных эмбеддингов токенов."},
    {"id": "d10", "text": "Оценка retrieval: hit@k и recall@k по заранее заданным релевантным документам."},
]

def chunk_text(text, size=80, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i : i + size])
        chunks.append(piece)
        i += max(1, size - overlap)
    return chunks

records = []
for d in docs:
    for j, ch in enumerate(chunk_text(d["text"])):
        records.append({"source": d["id"], "chunk_id": f"{d['id']}_c{j}", "text": ch})

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [r["text"] for r in records]
emb = model.encode(texts, normalize_embeddings=True).astype("float32")
dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)

queries = [
    ("Что такое dropout?", "d2"),
    ("Как ускорить поиск по векторам?", "d6"),
    ("Что такое ResNet?", "d5"),
    ("Как валидировать временной ряд?", "d8"),
    ("Что делает BERT?", "d9"),
    ("Как оценить retrieval?", "d10"),
    ("Что такое PyTorch nn?", "d1"),
    ("Что такое Adam?", "d3"),
]

def search(q, k=3):
    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, k)
    return [(records[i]["source"], records[i]["text"][:80], float(scores[0][j])) for j, i in enumerate(idxs[0])]

k_eval = 3
hits = []
rows_eval = []
for q, exp in queries:
    res = search(q, k=k_eval)
    retrieved_sources = [r[0] for r in res]
    hit = int(exp in retrieved_sources)
    hits.append(hit)
    rows_eval.append({"query": q, "expected_source": exp, "retrieved_sources": ";".join(retrieved_sources), "hit_at_k": hit})

hit_rate = np.mean(hits)
recall_k = hit_rate  # one relevant doc per query
print("hit@k", hit_rate, "recall@k", recall_k)

pd.DataFrame(rows_eval).to_csv(ART / "retrieval_eval.csv", index=False)

# experiment: chunk_size 40 vs 80
def build_index_chunk_size(cs):
    recs = []
    for d in docs:
        for j, ch in enumerate(chunk_text(d["text"], size=cs, overlap=10)):
            recs.append({"source": d["id"], "text": ch})
    e = model.encode([r["text"] for r in recs], normalize_embeddings=True).astype("float32")
    ix = faiss.IndexFlatIP(e.shape[1])
    ix.add(e)
    return recs, ix

r40, i40 = build_index_chunk_size(40)
r80, i80 = build_index_chunk_size(80)

def eval_ix(recs, ix):
    hs = []
    for q, exp in queries:
        qv = model.encode([q], normalize_embeddings=True).astype("float32")
        _, idxs = ix.search(qv, 3)
        srcs = [recs[i]["source"] for i in idxs[0]]
        hs.append(int(exp in srcs))
    return np.mean(hs)

print("chunk40", eval_ix(r40, i40), "chunk80", eval_ix(r80, i80))

# update KB
docs2 = docs + [
    {"id": "d11", "text": "Новый документ: FAISS поддерживает IVF и HNSW для больших баз."},
    {"id": "d12", "text": "Обновление политики: всегда логировать источники в RAG."},
]
records2 = []
for d in docs2:
    for j, ch in enumerate(chunk_text(d["text"])):
        records2.append({"source": d["id"], "text": ch})
emb2 = model.encode([r["text"] for r in records2], normalize_embeddings=True).astype("float32")
index2 = faiss.IndexFlatIP(emb2.shape[1])
index2.add(emb2)

q_spec = "Что такое IVF в FAISS?"
before = search(q_spec)  # old index still about general FAISS
qv = model.encode([q_spec], normalize_embeddings=True).astype("float32")
_, idxs = index2.search(qv, 3)
after = [(records2[i]["source"], records2[i]["text"][:80]) for i in idxs[0]]

pd.DataFrame([{
    "query": q_spec,
    "before_retrieved_sources": ";".join([x[0] for x in search(q_spec)]),
    "after_retrieved_sources": ";".join([a[0] for a in after]),
    "changed": str([x[0] for x in search(q_spec)] != [a[0] for a in after]),
}]).to_csv(ART / "retrieval_before_after_update.csv", index=False)

# mini-RAG
def rag_answer(question):
    ctx = search(question, k=2)
    context = " ".join([c[1] for c in ctx])
    answer = "Кратко: " + context[:200].replace("\n", " ") + "."
    return answer, [c[0] for c in ctx]

rag_rows = []
for q, _ in queries[:5]:
    ans, src = rag_answer(q)
    rag_rows.append({"question": q, "answer": ans, "retrieved_sources": ";".join(src)})
pd.DataFrame(rag_rows).to_csv(ART / "rag_examples.csv", index=False)
print("HW14 done")
'''
    ).strip()
    cells = [
        new_markdown_cell("# HW14: FAISS + sentence-transformers + mini-RAG\n\nУчебная база знаний в коде."),
        new_code_cell(code),
    ]
    save_nb("HW14/HW14.ipynb", cells)


if __name__ == "__main__":
    hw1011()
    hw12()
    hw13()
    hw14()
