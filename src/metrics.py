import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def macro_f1(y, yhat, k=6): return float(f1_score(y,yhat, average="macro", labels=list(range(k)), zero_division=0))
def per_class_f1(y, yhat, k=6): return f1_score(y,yhat, average=None, labels=list(range(k)), zero_division=0).tolist()
def confmat(y,yhat,k=6): return confusion_matrix(y,yhat, labels=list(range(k))).tolist()

def dp_delta(y, yhat, k=6):
    true = np.bincount(y, minlength=k)/len(y)
    pred = np.bincount(yhat, minlength=k)/len(yhat)
    return float(np.abs(true - pred).mean())

def eo_gap(y, yhat, k=6):
    fprs, fnrs = [], []
    for c in range(k):
        pos = (y==c); neg = ~pos
        pred_pos = (yhat==c)
        tpr = pred_pos[pos].mean() if pos.sum() else 0.0
        fpr = pred_pos[neg].mean() if neg.sum() else 0.0
        fnr = 1.0 - tpr
        fprs.append(fpr); fnrs.append(fnr)
    return float((sum(fprs)/k) + (sum(fnrs)/k))
