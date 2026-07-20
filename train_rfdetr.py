"""
Train + validate RF-DETR (Roboflow) on the COCO door dataset from yolo_to_coco.py.

Repo / docs: https://github.com/roboflow/rf-detr

INSTALL (GPU strongly recommended):
    pip install rfdetr

⚠ RF-DETR is a transformer detector (DINOv2 backbone). It needs a GPU.
  On a CPU-only node it is NOT practical — use a GPU partition, Google Colab
  (free GPU), or Roboflow hosted training.

After training, this script runs a final validation pass on the 'valid' split
and writes the same plots you have for YOLO / Faster R-CNN:
    PR_curve.png, P_curve.png, R_curve.png, F1_curve.png,
    confusion_matrix.png, confusion_matrix_normalized.png, metrics.txt

NOTE: RF-DETR's training API may evolve; if an argument is rejected, check the
current repo README. The predict() call returns a supervision Detections object
with .xyxy and .confidence — adjust if the API differs in your version.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRBase     # alternatives: RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge


# ── CONFIG ───────────────────────────────────────────────
DATASET_DIR = "/home/nsr59/Door_detection/Dataset_All/rfdetr_dataset"   # train/ valid/ test/
OUTPUT_DIR  = "/home/nsr59/Door_detection/runs/rfdetr_door"

EPOCHS      = 20
BATCH       = 4            # lower if GPU OOM
GRAD_ACCUM  = 4            # effective batch = BATCH * GRAD_ACCUM
LR          = 1e-4
RESOLUTION  = 224          # divisible by 56; matches your native image size

# Final-validation settings (for the plots)
EVAL_SPLIT   = "valid"     # which split to evaluate + plot ("valid" or "test")
IOU_MATCH    = 0.50        # IoU for a TP (mAP@0.5)
CONF_FOR_CM  = 0.50        # confidence threshold for the confusion matrix
INFER_THRESH = 0.05        # low threshold so the full PR curve can be built
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# EVAL HELPERS  (same approach as the FRCNN eval)
# ─────────────────────────────────────────────

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))


def validate_and_plot(model, dataset_dir, split, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split_dir = Path(dataset_dir) / split
    coco = json.load(open(split_dir / "_annotations.coco.json"))

    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    gt_by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt_by_img[a["image_id"]].append([x, y, x + w, y + h])

    all_scores, all_tp, total_gt = [], [], 0
    seen = 0
    for img_id, fname in id2file.items():
        gt = np.array(gt_by_img[img_id], dtype=np.float32).reshape(-1, 4)
        total_gt += len(gt)

        image = Image.open(split_dir / fname).convert("RGB")
        det = model.predict(image, threshold=INFER_THRESH)   # supervision Detections
        pb = np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4)
        ps = np.asarray(det.confidence, dtype=np.float32).reshape(-1)

        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        used = np.zeros(len(gt), dtype=bool)
        ious = iou_matrix(pb, gt)
        for i in range(len(pb)):
            tp = False
            if ious.shape[1]:
                j = int(np.argmax(ious[i]))
                if ious[i, j] >= IOU_MATCH and not used[j]:
                    used[j] = True; tp = True
            all_scores.append(float(ps[i])); all_tp.append(tp)

        seen += 1
        if seen % 100 == 0:
            print(f"[val] {seen}/{len(id2file)} images")

    if not all_scores:
        print("[val] model produced no detections — skipping plots.")
        return

    scores = np.array(all_scores)
    is_tp  = np.array(all_tp, dtype=bool)
    out    = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # PR curve + mAP@0.5
    order = np.argsort(-scores)
    tp_cum = np.cumsum(is_tp[order]); fp_cum = np.cumsum(~is_tp[order])
    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    ap50 = compute_ap(recall, precision)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, "b-", linewidth=3, label=f"door {ap50:.3f}  mAP@0.5")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(out / "PR_curve.png", dpi=150); plt.close()

    # P / R / F1 vs confidence
    thr = np.linspace(0, 1, 1000)
    P, R, Fc = [], [], []
    for t in thr:
        sel = scores >= t
        tp = int(np.sum(is_tp[sel])); fp = int(np.sum(~is_tp[sel])); fn = total_gt - tp
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/total_gt if total_gt else 0.0
        P.append(p); R.append(r); Fc.append(2*p*r/(p+r) if (p+r) else 0.0)
    P, R, Fc = np.array(P), np.array(R), np.array(Fc)
    bi = int(np.argmax(Fc)); best_f1, best_t = float(Fc[bi]), float(thr[bi])

    for y, ylabel, title, fname, lbl in [
        (P,  "Precision", "Precision-Confidence Curve", "P_curve.png", "door"),
        (R,  "Recall",    "Recall-Confidence Curve",    "R_curve.png", "door"),
        (Fc, "F1",        "F1-Confidence Curve",        "F1_curve.png",
         f"door  best {best_f1:.2f} @ {best_t:.3f}"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(thr, y, "b-", linewidth=3, label=lbl)
        plt.xlabel("Confidence"); plt.ylabel(ylabel); plt.title(title)
        plt.xlim(0, 1); plt.ylim(0, 1.01); plt.legend(loc="best"); plt.tight_layout()
        plt.savefig(out / fname, dpi=150); plt.close()

    # Confusion matrix at CONF_FOR_CM
    sel = scores >= CONF_FOR_CM
    tp = int(np.sum(is_tp[sel])); fp = int(np.sum(~is_tp[sel])); fn = total_gt - tp

    def plot_cm(normalized):
        cm = np.array([[tp, fp], [fn, 0.0]], dtype=float)
        title = "Confusion Matrix"
        if normalized:
            col = cm.sum(axis=0, keepdims=True)
            cm = np.divide(cm, col, out=np.zeros_like(cm), where=col != 0)
            title = "Confusion Matrix Normalized"
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() else 1)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i,j]:.2f}" if normalized else f"{int(cm[i,j])}",
                         ha="center", va="center",
                         color="white" if cm[i, j] > (cm.max()/2 if cm.max() else 1) else "black")
        plt.xticks([0, 1], ["door", "background"]); plt.yticks([0, 1], ["door", "background"])
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title)
        plt.colorbar(); plt.tight_layout()
        plt.savefig(out / ("confusion_matrix_normalized.png" if normalized
                           else "confusion_matrix.png"), dpi=150)
        plt.close()

    plot_cm(False); plot_cm(True)

    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/total_gt if total_gt else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    summary = (
        "── RF-DETR Validation ───────────────────\n"
        f"  split        : {split}\n"
        f"  val GT boxes : {total_gt}\n"
        f"  mAP@0.5      : {ap50:.4f}\n"
        f"  best F1      : {best_f1:.4f}  (at conf {best_t:.3f})\n"
        f"  @conf={CONF_FOR_CM:.2f}: P {prec:.4f} | R {rec:.4f} | F1 {f1:.4f}\n"
        f"  TP {tp}  FP {fp}  FN {fn}\n"
        "─────────────────────────────────────────\n"
    )
    print("\n" + summary)
    (out / "metrics.txt").write_text(summary)
    print(f"[OK] plots + metrics.txt written to {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("[warn] No GPU detected — RF-DETR on CPU is impractically slow.")
        print("[warn] Consider a GPU partition, Google Colab, or Roboflow hosted training.")
    else:
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # num_classes is inferred from the dataset's _annotations.coco.json categories.
    model = RFDETRBase(resolution=RESOLUTION)

    # RF-DETR validates on the 'valid' split each epoch internally (COCO mAP).
    model.train(
        dataset_dir      = DATASET_DIR,
        epochs           = EPOCHS,
        batch_size       = BATCH,
        grad_accum_steps = GRAD_ACCUM,
        lr               = LR,
        output_dir       = OUTPUT_DIR,
    )
    print(f"\n[done] training complete — checkpoints in {OUTPUT_DIR}")

    # ── Final validation pass → YOLO-style plots for comparison ──
    print(f"\n[val] running final validation on '{EVAL_SPLIT}' split for plots ...")
    validate_and_plot(model, DATASET_DIR, EVAL_SPLIT, OUTPUT_DIR)
    print(f"\n[✓] All done — see {OUTPUT_DIR}")


if __name__ == "__main__":
    main()