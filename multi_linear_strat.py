import numpy as np
import pandas as pd
import anual_factor

BARS_PER_YEAR = anual_factor.annualization_factor()

def weights(signal, q=0.1): #signal weighting
    rank = signal.rank(axis=1, pct=True)
    L = (rank >= 1 - q).astype(float); S = (rank <= q).astype(float)
    w = 0.5 * L.div(L.sum(axis=1), axis=0) - 0.5 * S.div(S.sum(axis=1), axis=0)
    return w.fillna(0.0)

def sleeve(w_target, ret, H=1, kappa=1.0, delay=1):
    """One factor book on its own NAV: rebalance every H bars, a fraction kappa toward target."""
    
    W = w_target.shift(delay).fillna(0.0).to_numpy()
    R = ret.reindex_like(w_target).fillna(0.0).to_numpy()
    T, N = W.shape
    
    cur = np.zeros(N)
    pos = np.empty((T, N))
    gross = np.empty(T)
    trade = np.zeros(T)
    
    for t in range(T):
        if t % H == 0: # when to do a transaction [1,8,48] - 30-minute intervals
            step = kappa * (W[t] - cur)  #como agora ja nao é transacoes imediantas, determinar o shift é diferente
            trade[t] = np.abs(step).sum()
            cur = cur + step
        pos[t] = cur
        g = cur @ R[t]; gross[t] = g
        cur = cur * (1 + R[t]) / (1 + g)
    return {"pos": pos, "gross": pd.Series(gross, index=w_target.index),
            "turnover": pd.Series(trade, index=w_target.index), "R": R}

def combine(sleeves, a):
    """Blend sleeves; trades are netted across sleeves and re-allocation trades are included."""
    keys = list(sleeves); A = a[keys].to_numpy(); T = A.shape[0]
    N = sleeves[keys[0]]["pos"].shape[1]
    
    gross = np.zeros(T)
    trade = np.zeros(T)
    unnetted = np.zeros(T)
    
    for t in range(T):
        tgt = np.zeros(N); pre = np.zeros(N)
        for j, k in enumerate(keys):
            s = sleeves[k]; gross[t] += A[t, j] * s["gross"].iat[t]
            tgt += A[t, j] * s["pos"][t]
            if t > 0:
                p, r = s["pos"][t - 1], s["R"][t - 1]
                pre += A[t - 1, j] * p * (1 + r) / (1 + s["gross"].iat[t - 1])
            unnetted[t] += A[t, j] * s["turnover"].iat[t]
        trade[t] = np.abs(tgt - pre).sum()
    return pd.DataFrame({"gross": gross, "turnover": trade, "turnover_unnetted": unnetted}, index=a.index)

def summarize(bt):
    g, tn = bt["gross"].mean(), bt["turnover"].mean()
    return {"gross_ann": g * BARS_PER_YEAR, "gross_sharpe": g / bt["gross"].std() * np.sqrt(BARS_PER_YEAR),
            "turnover/bar": tn, "breakeven_bp": g / tn * 1e4 if tn > 0 else np.nan}

def alloc(sleeves, method="rp", window=1344, every=48, cost_bp=0.0, delta=0.5, min_std=1e-12):
    """Relative allocations a_k (rows sum to 1, or 0 if no sleeve is worth running).
    Uses only sleeve data strictly before the bar where the allocation starts (no look-ahead).
      rp : a_k ~ 1/sigma_k                      (risk parity, no mean estimate)
      mv : a  ~ Sigma_s^{-1} mu_net, clipped >= 0, with Sigma_s = (1-delta)*Sigma + delta*diag(Sigma)
           mu_net = mean gross - cost * mean turnover  (own turnover, i.e. ignores netting -> conservative)"""
    keys = list(sleeves); K = len(keys)
    G  = pd.DataFrame({k: sleeves[k]["gross"] for k in keys})
    TN = pd.DataFrame({k: sleeves[k]["turnover"] for k in keys})
    T = len(G); A = np.full((T, K), np.nan); A[:window] = 1.0 / K
    for t in range(window, T, every):
        g = G.iloc[t - window:t]
        sd = g.std().to_numpy()
        live = np.isfinite(sd) & (sd > min_std)          # sleeves that actually traded in the window
        a = np.zeros(K)
        if live.any():
            gl = g.loc[:, live]; sdl = sd[live]
            cost = cost_bp * 1e-4 * TN.iloc[t - window:t].loc[:, live].mean().to_numpy()
            if method == "rp":
                al = 1.0 / sdl
            elif method == "rp_gate":
                al = np.where(gl.mean().to_numpy() > margin * cost, 1.0 / sdl, 0.0)
            else:  # "mv"
                mu = gl.mean().to_numpy() - cost
                C = gl.cov().to_numpy(); C = (1 - delta) * C + delta * np.diag(np.diag(C))
                try:
                    al = np.clip(np.linalg.solve(C, mu), 0, None)
                except np.linalg.LinAlgError:
                    al = np.clip(np.linalg.lstsq(C, mu, rcond=None)[0], 0, None)
            a[live] = al
        a = a / a.sum() if a.sum() > 0 else np.zeros(K)
        A[t:t + every] = a
    return pd.DataFrame(A, index=G.index, columns=keys)

def main(ret,sig):
    print("This is the new update")
    sched = {"rev": (8, 1), "mom": (16, 1), "size": (48, 0.05)} # (H, kappa) per sleeve; tune these
    sl = {k: sleeve(weights(sig[k], 0.1), ret, H=sched[k][0], kappa=sched[k][1]) for k in sig}
    cost_bp = 0
    a  = alloc(sl, method="mv", cost_bp=2)   # or "rp"; rows sum to 1, or 0 if no sleeve survives
    bt = combine(sl, a)  
    net = bt["gross"] - cost_bp * 1e-4 * bt["turnover"]
    return net,bt