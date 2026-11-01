import cupy as cp
import numpy as np
import pandas as pd

q = 0.01
CH = 20000            # rows per chunk; lower to 10000 if you run out of VRAM

def gpu_qbounds(a, q):
    """Per-row q and 1-q quantiles ignoring NaN/inf (linear interpolation, like pandas)."""
    T, N = a.shape
    fin = cp.isfinite(a)
    n = fin.sum(axis=1)
    s = cp.sort(cp.where(fin, a, cp.inf), axis=1)          # non-finite values go to the end
    r = cp.arange(T)
    def pick(p):
        pos = p * cp.maximum(n - 1, 0)
        lo = cp.floor(pos).astype(cp.int64)
        hi = cp.minimum(cp.ceil(pos).astype(cp.int64), N - 1)
        f = pos - lo
        v = s[r, lo] * (1 - f) + s[r, hi] * f
        return cp.where(n > 0, v, cp.nan)
    return pick(q), pick(1 - q)

def gpu_clip(a, q):
    lo, hi = gpu_qbounds(a, q)
    return cp.clip(a, lo[:, None], hi[:, None])

def gpu_z(a, q):
    a = gpu_clip(a, q)
    mu = cp.nanmean(a, axis=1, keepdims=True)
    sd = cp.nanstd(a, axis=1, ddof=1, keepdims=True)
    return (a - mu) / sd

def factor_regression(ret, factors, q=0.01, CH=20000, min_obs=30,
                      raw=(), return_z=False):
    names = list(factors)
    K = len(names) + 1
    T, C = ret.shape

    ret_n = ret.to_numpy(dtype="float32")
    fac_n = {n: factors[n].reindex_like(ret).to_numpy(dtype="float32") for n in names}

    out = np.full((T, K), np.nan)
    Z = {n: np.full((T, C), np.nan, dtype="float32") for n in names}

    for s in range(0, T, CH):
        sl = slice(s, s + CH)

        Y = gpu_clip(cp.asarray(ret_n[sl]), q)
        X = []
        for n in names:
            x = cp.asarray(fac_n[n][sl])
            X.append(x if n in raw else gpu_z(x, q))
        X = cp.stack(X, axis=2)                                   # chunk x N x F

        m = cp.isfinite(Y) & cp.isfinite(X).all(axis=2)           # same as dropna()
        A = cp.concatenate([cp.ones_like(Y)[:, :, None], X], axis=2)   # chunk x N x K
        A = cp.where(m[:, :, None], A, 0)
        yv = cp.where(m, Y, 0)

        XtX = cp.matmul(A.transpose(0, 2, 1), A).astype(cp.float64)
        Xty = cp.matmul(A.transpose(0, 2, 1), yv[:, :, None]).astype(cp.float64)

        ok = m.sum(axis=1) >= min_obs
        XtX = cp.where(ok[:, None, None], XtX, cp.eye(K))
        beta = cp.linalg.solve(XtX, Xty)[:, :, 0]
        out[sl] = cp.asnumpy(cp.where(ok[:, None], beta, cp.nan))

        if return_z:
            for i, n in enumerate(names):
                Z[n][sl] = cp.asnumpy(X[:, :, i])

        del A, X, Y, yv, XtX, Xty, beta
        cp.get_default_memory_pool().free_all_blocks()

    coefs = pd.DataFrame(out, index=ret.index, columns=["const"] + names).dropna()
    if return_z:
        Z = {n: pd.DataFrame(v, index=ret.index, columns=ret.columns) for n, v in Z.items()}
        return coefs, Z
    return coefs
