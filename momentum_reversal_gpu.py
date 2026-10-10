import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import cupy as cp


def main(k_max,ret):
    ret_g = cp.asarray(ret.to_numpy(dtype="float32"))
    frames = []
    for k in range(1, k_max + 1):
        res = fm_lag_gpu(ret_g, k)
        df = pd.DataFrame(res, index=ret.index[k:],
                      columns=["alpha", "gamma", "t_stat", "n_obs"]).dropna()
        df["k"] = k
        frames.append(df.set_index("k", append=True))
    betas = pd.concat(frames)
    return betas

def fm_lag_gpu(ret_g, k, CH=20000):
    # Cross-sectional regressions r(i,t) = alpha + gamma * r(i,t-k) for every t >= k
    T = ret_g.shape[0]
    f64 = cp.float64
    out = []

    for s in range(k, T, CH):
        e = min(s + CH, T)
        y = ret_g[s:e]            # r(i,t),   chunk x S
        x = ret_g[s - k:e - k]    # r(i,t-k), a view, no copy

        m = cp.isfinite(x) & cp.isfinite(y) #get values where x and y are valid
        n = m.sum(axis=1,dtype=f64) #as m returns true and false, it sums the trues
        n1 = cp.maximum(n,1) #as i will be div by n1, i substitute 0's by 1

        xbar = cp.where(m,x,0).sum(axis=1,dtype=f64) / n1
        ybar = cp.where(m,y,0).sum(axis=1,dtype=f64) / n1

        xc = cp.where(m,x-xbar[:,None].astype(cp.float32), 0) #[:,None] adds a new axis so the shape is (T,1)
        yc = cp.where(m,y-ybar[:,None].astype(cp.float32), 0)

        sxx = (xc * xc).sum(axis=1, dtype=f64)
        sxy = (xc * yc).sum(axis=1, dtype=f64)
        syy = (yc * yc).sum(axis=1, dtype=f64)

        gamma = sxy/sxx
        alpha=ybar-gamma*xbar
        ssr = syy - gamma*sxy
        se = cp.sqrt(ssr / (n - 2) / sxx)
        t_stat = gamma / se

        res = cp.stack([alpha, gamma, t_stat, n], axis=1)
        res = cp.where((n >= 3)[:, None], res, cp.nan)           # skip if insufficient assets
        out.append(cp.asnumpy(res))
    return np.vstack(out)
