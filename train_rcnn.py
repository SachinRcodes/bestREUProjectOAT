"""
Faster R-CNN on a 2-class door / door-frame dataset (train/valid/test already split).

Reads YOLO-format labels directly, trains on train/, early-stops on valid/, then
runs a final test-split evaluation with PROPER per-class metrics:
  • per-class AP@0.5 (door, door frame) + combined mAP@0.5
  • per-class Precision-Recall curves
  • micro-averaged P/R/F1-confidence curves
  • (classes + background) confusion matrix, raw and normalized
  • metrics.txt summary
Run folder name encodes image size, epochs and batch.
"""

import os
import csv
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ── CONFIG ───────────────────────────────────────────────
DATASET_ROOT = r"/home/nsr59/Door_detection/Dataset_All/door_doorframe_dataset"  # EDIT to your dataset

EPOCHS       = 20
BATCH        = 4
WORKERS      = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
LR           = 0.005
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP      = 12
LR_GAMMA     = 0.1
PATIENCE     = 8

IMG_SIZE     = 640          # FRCNN internal resize (min=max); 224 to match older runs
IOU_MATCH    = 0.50
CONF_FOR_CM  = 0.50
PRED_CONF    = 0.05         # low, so curves see the full score range

PROJECT  = r"/home/nsr59/Door_detection/runs"
RUN_NAME = f"RCNN_doorframe_{IMG_SIZE}px_{EPOCHS}e_{BATCH}b"
OUT_DIR  = Path(PROJECT) / RUN_NAME

DEFAULT_NAMES = ["door", "door frame"]   # used only if dataset has no data.yaml
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────


def read_classes(root):
    src = Path(root) / "data.yaml"
    if src.exists():
        with open(src) as f:
            d = yaml.safe_load(f) or {}
        names = d.get("names", DEFAULT_NAMES)
        return list(names)
    return DEFAULT_NAMES


CLASS_NAMES = read_classes(DATASET_ROOT)        # display names, index 0..nc-1
NC          = len(CLASS_NAMES)                  # real classes
NUM_CLASSES = NC + 1                            # +1 background (torchvision)


def yolo_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def split_images(root, split):
    d = Path(root) / split / "images"
    if not d.exists() and split == "valid":
        d = Path(root) / "val" / "images"
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)


# ── DATASET ──────────────────────────────────────────────
class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, train=False):
        self.image_paths = [Path(p) for p in image_paths]
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import random
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        lp = yolo_label_path(img_path)
        if lp.exists():
            for line in open(lp):
                p = line.split()
                if len(p) != 5:
                    continue
                cls, cx, cy, bw, bh = map(float, p)
                cx, cy, bw, bh = cx*W, cy*H, bw*W, bh*H
                x1, y1, x2, y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2]); labels.append(int(cls)+1)  # +1: bg=0

        if self.train and boxes and random.random() < 0.5:
            img = F.hflip(img)
            boxes = [[W-x2, y1, W-x1, y2] for (x1, y1, x2, y2) in boxes]

        if boxes:
            bt = torch.as_tensor(boxes, dtype=torch.float32)
            lt = torch.as_tensor(labels, dtype=torch.int64)
        else:
            bt = torch.zeros((0, 4), dtype=torch.float32)
            lt = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": bt, "labels": lt, "image_id": torch.tensor([idx]),
                  "area": (bt[:, 3]-bt[:, 1])*(bt[:, 2]-bt[:, 0]),
                  "iscrowd": torch.zeros((lt.shape[0],), dtype=torch.int64)}
        return F.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model():
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                        min_size=IMG_SIZE, max_size=IMG_SIZE)
    except ImportError:
        model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=IMG_SIZE, max_size=IMG_SIZE)
    inf = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inf, NUM_CLASSES)
    return model


# ── TRAIN / VAL ──────────────────────────────────────────
def train_one_epoch(model, loader, opt, epoch):
    model.train(); running = 0.0
    for i, (imgs, tgts) in enumerate(loader):
        imgs = [x.to(DEVICE) for x in imgs]
        tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
        loss = sum(model(imgs, tgts).values())
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
        if (i+1) % 20 == 0:
            print(f"  [epoch {epoch}] step {i+1}/{len(loader)} loss={loss.item():.4f}")
    avg = running / max(1, len(loader))
    print(f"[epoch {epoch}] mean train loss: {avg:.4f}")
    return avg


@torch.no_grad()
def quick_val_loss(model, loader):
    model.train(); running = 0.0
    for imgs, tgts in loader:
        imgs = [x.to(DEVICE) for x in imgs]
        tgts = [{k: v.to(DEVICE) for k, v in t.items()} for t in tgts]
        running += sum(model(imgs, tgts).values()).item()
    avg = running / max(1, len(loader))
    print(f"[val] mean val loss: {avg:.4f}")
    return avg


# ── METRICS HELPERS ──────────────────────────────────────
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb-lt, 0, None); inter = wh[..., 0]*wh[..., 1]
    return inter / np.clip(aa[:, None]+ab[None, :]-inter, 1e-9, None)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.])); mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))


@torch.no_grad()
def collect(model, loader):
    """Per-class detection matching (for AP) + class-agnostic matching (for the
    confusion matrix, recording predicted-vs-true class pairs)."""
    model.eval()
    det = {c: {"scores": [], "tp": []} for c in range(1, NC+1)}   # 1..NC
    gt_count = {c: 0 for c in range(1, NC+1)}
    cm = np.zeros((NC+1, NC+1), dtype=int)        # rows=pred, cols=true; last index = background

    for imgs, tgts in loader:
        preds = model([x.to(DEVICE) for x in imgs])
        for pred, tgt in zip(preds, tgts):
            pb = pred["boxes"].cpu().numpy(); ps = pred["scores"].cpu().numpy(); pl = pred["labels"].cpu().numpy()
            gb = tgt["boxes"].cpu().numpy().reshape(-1, 4); gl = tgt["labels"].cpu().numpy()

            # ---- per-class AP matching ----
            for c in range(1, NC+1):
                pcb, pcs = pb[pl == c], ps[pl == c]
                gcb = gb[gl == c]
                gt_count[c] += len(gcb)
                order = np.argsort(-pcs); pcb, pcs = pcb[order], pcs[order]
                used = np.zeros(len(gcb), bool); ious = iou_matrix(pcb, gcb)
                for i in range(len(pcb)):
                    tp = False
                    if ious.shape[1]:
                        j = int(np.argmax(ious[i]))
                        if ious[i, j] >= IOU_MATCH and not used[j]:
                            used[j] = True; tp = True
                    det[c]["scores"].append(float(pcs[i])); det[c]["tp"].append(tp)

            # ---- confusion matrix (class-agnostic IoU match @ conf threshold) ----
            keep = ps >= CONF_FOR_CM
            kb, kl, kss = pb[keep], pl[keep], ps[keep]
            order = np.argsort(-kss); kb, kl = kb[order], kl[order]
            usedg = np.zeros(len(gb), bool); matched = np.zeros(len(kb), bool)
            ious = iou_matrix(kb, gb)
            for i in range(len(kb)):
                if ious.shape[1]:
                    for j in np.argsort(-ious[i]):
                        if ious[i, j] < IOU_MATCH:
                            break
                        if not usedg[j]:
                            usedg[j] = True; matched[i] = True
                            cm[kl[i]-1, gl[j]-1] += 1     # predicted vs true class
                            break
            for i in range(len(kb)):
                if not matched[i]:
                    cm[kl[i]-1, NC] += 1                  # false positive -> true=background
            for j in range(len(gb)):
                if not usedg[j]:
                    cm[NC, gl[j]-1] += 1                  # missed GT -> pred=background
    return det, gt_count, cm


def make_plots(det, gt_count, cm, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # per-class PR + AP
    aps = {}
    plt.figure(figsize=(8, 5))
    for c in range(1, NC+1):
        s = np.array(det[c]["scores"]); tp = np.array(det[c]["tp"], bool)
        if len(s) == 0:
            aps[c] = 0.0; continue
        o = np.argsort(-s); tpc = np.cumsum(tp[o]); fpc = np.cumsum(~tp[o])
        recall = tpc/max(gt_count[c], 1); precision = tpc/np.clip(tpc+fpc, 1e-9, None)
        aps[c] = compute_ap(recall, precision)
        plt.plot(recall, precision, lw=2.5, label=f"{CLASS_NAMES[c-1]} {aps[c]:.3f}")
    mAP = float(np.mean(list(aps.values()))) if aps else 0.0
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve  (mAP@0.5 = {mAP:.3f})")
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"PR_curve.png", dpi=150); plt.close()

    # micro-averaged P/R/F1 vs confidence (all classes pooled)
    all_s = np.concatenate([np.array(det[c]["scores"]) for c in range(1, NC+1)]) if NC else np.array([])
    all_tp = np.concatenate([np.array(det[c]["tp"], bool) for c in range(1, NC+1)]) if NC else np.array([], bool)
    total_gt = sum(gt_count.values())
    thr = np.linspace(0, 1, 1000); P, R, Fc = [], [], []
    for t in thr:
        sel = all_s >= t; tp = int(np.sum(all_tp[sel])); fp = int(np.sum(~all_tp[sel]))
        p = tp/(tp+fp) if (tp+fp) else 0.; r = tp/total_gt if total_gt else 0.
        P.append(p); R.append(r); Fc.append(2*p*r/(p+r) if (p+r) else 0.)
    P, R, Fc = np.array(P), np.array(R), np.array(Fc); bi = int(np.argmax(Fc))
    for y, yl, ti, fn, lb in [(P, "Precision", "Precision-Confidence Curve", "P_curve.png", "all classes"),
                              (R, "Recall", "Recall-Confidence Curve", "R_curve.png", "all classes"),
                              (Fc, "F1", "F1-Confidence Curve", "F1_curve.png",
                               f"all classes  best {Fc[bi]:.2f} @ {thr[bi]:.3f}")]:
        plt.figure(figsize=(8, 5)); plt.plot(thr, y, "b-", lw=3, label=lb)
        plt.xlabel("Confidence"); plt.ylabel(yl); plt.title(ti)
        plt.xlim(0, 1); plt.ylim(0, 1.01); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/fn, dpi=150); plt.close()

    # confusion matrix (classes + background)
    labels = CLASS_NAMES + ["background"]

    def draw_cm(norm):
        m = cm.astype(float); t = "Confusion Matrix"
        if norm:
            col = m.sum(0, keepdims=True)
            m = np.divide(m, col, out=np.zeros_like(m), where=col != 0); t = "Confusion Matrix Normalized"
        plt.figure(figsize=(6.5, 5.5)); plt.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if m.max() else 1)
        for i in range(NC+1):
            for j in range(NC+1):
                plt.text(j, i, f"{m[i,j]:.2f}" if norm else f"{int(m[i,j])}", ha="center", va="center",
                         color="white" if m[i, j] > (m.max()/2 if m.max() else 1) else "black")
        plt.xticks(range(NC+1), labels, rotation=45, ha="right"); plt.yticks(range(NC+1), labels)
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(t); plt.colorbar(); plt.tight_layout()
        plt.savefig(out_dir/("confusion_matrix_normalized.png" if norm else "confusion_matrix.png"), dpi=150)
        plt.close()
    draw_cm(False); draw_cm(True)

    # metrics.txt
    lines = ["── Faster R-CNN Validation (test split) ──", f"  run: {RUN_NAME}",
             f"  total GT boxes: {total_gt}"]
    for c in range(1, NC+1):
        lines.append(f"  AP@0.5 [{CLASS_NAMES[c-1]}]: {aps[c]:.4f}  (GT {gt_count[c]})")
    lines += [f"  mAP@0.5: {mAP:.4f}",
              f"  best F1 (all): {Fc[bi]:.4f} @ conf {thr[bi]:.3f}",
              "──────────────────────────────────────────"]
    summary = "\n".join(lines) + "\n"
    print("\n" + summary); (out_dir/"metrics.txt").write_text(summary)
    print(f"[OK] plots + metrics.txt -> {out_dir}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {DEVICE} | classes={CLASS_NAMES} | run={RUN_NAME}")
    if DEVICE.type == "cpu":
        print("[warn] No GPU — Faster R-CNN on CPU is slow; YOLO is the better CPU choice.")

    tr = split_images(DATASET_ROOT, "train")
    va = split_images(DATASET_ROOT, "valid")
    te = split_images(DATASET_ROOT, "test")
    print(f"[data] train {len(tr)} | valid {len(va)} | test {len(te)}")
    if not tr:
        raise SystemExit(f"No training images under {DATASET_ROOT}/train/images")

    pin = (DEVICE.type == "cuda")
    tl = torch.utils.data.DataLoader(YoloDetectionDataset(tr, True), batch_size=BATCH,
                                     shuffle=True, num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin)
    vl = torch.utils.data.DataLoader(YoloDetectionDataset(va, False), batch_size=BATCH,
                                     shuffle=False, num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin)
    el = torch.utils.data.DataLoader(YoloDetectionDataset(te, False), batch_size=BATCH,
                                     shuffle=False, num_workers=WORKERS, collate_fn=collate_fn, pin_memory=pin)

    model = build_model().to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=LR_GAMMA)

    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "best"])

    best = float("inf"); no_improve = 0
    for ep in range(1, EPOCHS+1):
        trl = train_one_epoch(model, tl, opt, ep); sched.step()
        vll = quick_val_loss(model, vl)
        torch.save(model.state_dict(), OUT_DIR/"last.pt")
        is_best = vll < best
        if is_best:
            best = vll; no_improve = 0; torch.save(model.state_dict(), OUT_DIR/"best.pt")
            print(f"[\u2713] new best val_loss {best:.4f}")
        else:
            no_improve += 1
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{trl:.5f}", f"{vll:.5f}", int(is_best)])
        if no_improve >= PATIENCE:
            print(f"[early-stop] no val improvement for {PATIENCE} epochs."); break

    print("\n[eval] final validation on test split using best.pt ...")
    if (OUT_DIR/"best.pt").exists():
        model.load_state_dict(torch.load(OUT_DIR/"best.pt", map_location=DEVICE))
    det, gt_count, cm = collect(model, el)
    make_plots(det, gt_count, cm, OUT_DIR)
    print(f"\n[done] -> {OUT_DIR}")


if __name__ == "__main__":
    main()
