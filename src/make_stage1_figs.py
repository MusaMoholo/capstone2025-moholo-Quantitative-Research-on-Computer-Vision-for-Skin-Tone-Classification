import json, sys
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt
DEF = np.array([295,480,330,278,153,64], int)
def load_counts():
    p=Path("manifests")/"split_counts.json"
    if p.exists():
        try:
            d=json.loads(p.read_text()); a=np.array(d.get("test", DEF), int)
            if a.size==6: return a
        except Exception: pass
    return DEF
def load_preds(p):
    yt, yp = [], []
    with open(p) as f:
        h=f.readline().strip().split(","); m={k:i for i,k in enumerate(h)}
        yt_k = "y_true" if "y_true" in m else h[0]
        yp_k = "y_pred" if "y_pred" in m else h[1]
        for line in f:
            s=line.strip().split(",")
            if len(s)>=2:
                yt.append(int(s[m[yt_k]])); yp.append(int(s[m[yp_k]]))
    return np.array(yt), np.array(yp)
def bar(truec,predc,out):
    x=np.arange(6); w=0.4
    plt.figure(figsize=(6,4))
    plt.bar(x-w/2,truec,w,label="True"); plt.bar(x+w/2,predc,w,label="Pred")
    plt.xlabel("Class index"); plt.ylabel("Count"); plt.title("Stage-1 (gamma): Predicted vs True")
    plt.legend(); plt.tight_layout(); plt.savefig(out,dpi=180); plt.close()
def heat(cm,out):
    plt.figure(figsize=(5.5,4.8))
    plt.imshow(cm,aspect="auto"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Stage-1 (gamma): Confusion Matrix"); plt.colorbar(); plt.tight_layout()
    plt.savefig(out,dpi=180); plt.close()
if __name__=="__main__":
    preds_path = sys.argv[1]
    truec = load_counts()
    yt,yp = load_preds(preds_path)
    predc = np.bincount(yp, minlength=6)
    cm = np.zeros((6,6), int)
    for t,p in zip(yt,yp):
        if 0<=t<6 and 0<=p<6: cm[t,p]+=1
    Path("Figures").mkdir(parents=True, exist_ok=True)
    bar(truec, predc, Path("Figures")/"stage1_pred_vs_true_gamma.png")
    heat(cm, Path("Figures")/"stage1_confusion_gamma.png")
