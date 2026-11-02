import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import cupy as cp
import os


def main(k_max, ret, out_dir="betas", overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ret_g = cp.asarray(ret.to_numpy(dtype="float32"))

    for k in range(1, k_max + 1):
        path = out_dir / f"k={k:05d}.parquet"
        if path.exists() and not overwrite:      # lets you resume after a crash
            continue

        res = fm_lag_gpu(ret_g, k)
        df = pd.DataFrame(res, index=ret.index[k:],
                          columns=["alpha", "gamma", "t_stat", "n_obs"]).dropna()

        # write to a temp file, then rename, so an interrupted run never leaves a corrupt file
        tmp = path.with_suffix(".tmp")
        df.astype("float32").to_parquet(tmp)
        os.replace(tmp, path)

        if k % 100 == 0:
            print(f"k={k}/{k_max}")
            cp.get_default_memory_pool().free_all_blocks()   # release cached GPU memory


def load_betas(out_dir="betas", ks=None, cols=None):
    """Long format like your old output: MultiIndex (k, time). Pass a subset of ks to stay light."""
    out_dir = Path(out_dir)
    if ks is None:
        ks = sorted(int(p.stem.split("=")[1]) for p in out_dir.glob("k=*.parquet"))
    frames = [pd.read_parquet(out_dir / f"k={k:05d}.parquet", columns=cols) for k in ks]
    return pd.concat(frames, keys=ks, names=["k"])


def load_wide(col="gamma", out_dir="betas", ks=None):
    """One column (e.g. gamma or t_stat) as a time x k matrix."""
    out_dir = Path(out_dir)
    if ks is None:
        ks = sorted(int(p.stem.split("=")[1]) for p in out_dir.glob("k=*.parquet"))
    cols = {k: pd.read_parquet(out_dir / f"k={k:05d}.parquet", columns=[col])[col] for k in ks}
    return pd.concat(cols, axis=1)

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
