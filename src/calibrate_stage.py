import numpy as np
from scipy.optimize import minimize

def logits_from_probs(probs):
    probs = np.clip(probs, 1e-8, 1.0)
    return np.log(probs)

def fit_temperature(val_probs, y_val):
    z = logits_from_probs(val_probs)
    def nll_t(t):
        T = np.exp(t[0])
        zz = z / T
        zz = zz - zz.max(axis=1, keepdims=True)
        p = np.exp(zz) / np.exp(zz).sum(axis=1, keepdims=True)
        return -np.log(p[np.arange(len(y_val)), y_val]).mean()
    res = minimize(nll_t, x0=[0.0], method="L-BFGS-B")
    return float(np.exp(res.x[0]))

def apply_temp(probs, T):
    z = logits_from_probs(probs) / T
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z)/np.exp(z).sum(axis=1, keepdims=True)
