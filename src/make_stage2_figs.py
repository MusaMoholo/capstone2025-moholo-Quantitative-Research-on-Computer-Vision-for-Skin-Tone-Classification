import json, sys, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DEF = np.array([295,480,330,278,153,64], int)

def load_counts():
    p = Path("manifests") / "split_counts.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            a = np.array(d.get("test", DEF), int)
            if a.size == 6:
                return a
        except Exception:
            pass
    return DEF

def load_preds(pred_path: str):
    ys_true, ys_pred = [], []
    with open(pred_path, newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        yt_k = "y_true" if "y_true" in cols else (cols[0] if cols else None)
        yp_k = "y_pred" if "y_pred" in cols else (cols[1] if len(cols) > 1 else None)
        if not yt_k or not yp_k:
            raise RuntimeError(f"Unexpected header in {pred_path}: {cols}")
        for row in r:
            try:
                ys_true.append(int(row[yt_k]))
                ys_pred.append(int(row[yp_k]))
            except Exception:
                continue
    return np.array(ys_true, int), np.array(ys_pred, int)

def confusion(yt, yp):
    cm = np.zeros((6,6), int)
    for t,p in zip(yt, yp):
        if 0 <= t < 6 and 0 <= p < 6:
            cm[t, p] += 1
    return cm

def bar_true_pred(truec, predc, out_path):
    x = np.arange(6); w = 0.4
    plt.figure(figsize=(6,4))
    plt.bar(x - w/2, truec, w, label="True")
    plt.bar(x + w/2, predc, w, label="Pred")
    plt.xlabel("Class index"); plt.ylabel("Count")
    plt.title("Stage-2 (N=96): Predicted vs True")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()

def heatmap(cm, out_path):
    plt.figure(figsize=(5.5,4.8))
    plt.imshow(cm, aspect="auto")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Stage-2 (N=96): Confusion Matrix")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(out_path, dpi=180); plt.close()

if __name__ == "__main__":
    preds = sys.argv[1]
    outdir = Path("Figures"); outdir.mkdir(parents=True, exist_ok=True)
    truec = load_counts()
    yt, yp = load_preds(preds)
    predc = np.bincount(yp, minlength=6)
    cm = confusion(yt, yp)
    bar_true_pred(truec, predc, outdir / "stage2_pred_vs_true_lastn96.png")
    heatmap(cm, outdir / "stage2_confusion_lastn96.png")
