"""
Avellaneda-Lee style multi-asset Ornstein-Uhlenbeck statistical arbitrage
applied to a cryptocurrency universe.

Pipeline
--------
1. Clean the wide price panel (drop stablecoins, leveraged tokens, junk tickers).
2. On each trading day, using a trailing PCA_WINDOW of standardized returns,
   run PCA over the subset of coins with a complete history -> eigenportfolios
   (systematic risk factors), analogous to sector/market factors in equities.
3. Over a shorter trailing REG_WINDOW, regress every coin's returns on the
   eigenportfolio returns (single matrix solve for all coins at once) to get
   factor loadings (betas) and idiosyncratic residuals.
4. Cumulative sum the residuals -> auxiliary process X_i,t, fit a discrete
   OU / AR(1) process to it, and turn the fit into an s-score
   s = (X_t - theta) / sigma_eq.
5. Trade the s-score with the classic Avellaneda-Lee thresholds, market-neutral
   (long/short the coin, implicitly hedged against the eigenportfolios since
   the P&L booked is the coin's realized *residual* return).
"""
import numpy as np
import pandas as pd

# ----------------------------- parameters -----------------------------
PCA_WINDOW   = 252   # trailing days used to estimate factors (eigenportfolios)
REG_WINDOW   = 60    # trailing days used to estimate betas / fit the OU process
VAR_EXPLAIN  = 0.55  # target cumulative variance explained by eigenportfolios
MAX_FACTORS  = 15    # cap on number of eigenportfolios (Avellaneda-Lee use ~15 for S&P500)
MIN_FACTORS  = 1

ENTRY_S      = 1.25  # |s| above this opens a position
EXIT_LONG_S  = -0.50 # close a long when s rises back above this
EXIT_SHORT_S = 0.75  # close a short when s falls back below this

ANNUALIZATION = 365  # crypto trades every day
MAX_HALF_LIFE_DAYS = 30   # only trade coins whose OU half-life is below this
MIN_KAPPA = ANNUALIZATION * np.log(2) / MAX_HALF_LIFE_DAYS

MIN_COINS_FOR_PCA = 20     # need at least this many coins with full history to run PCA


def load_and_clean(pickle_path: str) -> pd.DataFrame:
    """Load the wide price panel and drop stablecoins / leveraged tokens / junk."""
    df = pd.read_pickle(pickle_path)
    df = df.sort_index()

    counts = df.notna().sum()
    candidates = counts[counts >= 1000].index.tolist()

    rets = np.log(df[candidates]).diff()
    ann_vol = rets.std() * np.sqrt(ANNUALIZATION)

    stable_like = set(ann_vol[ann_vol < 0.25].index)  # stablecoins, EUR/GBP/AUD/PAXG pegs
    leveraged = {c for c in candidates if any(s in c for s in
                 ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"))}
    non_ascii = {c for c in candidates if not c.isascii()}

    universe = [c for c in candidates if c not in stable_like
                and c not in leveraged and c not in non_ascii]
    return df[universe]


def _fit_ar1_vectorized(X: np.ndarray):
    """
    Fit X_t = a + b*X_{t-1} + eps for every column of X (shape T x N) at once.
    Returns a, b, resid_std (each length N).
    """
    x0 = X[:-1, :]
    x1 = X[1:, :]
    n = x0.shape[0]
    xm = x0.mean(axis=0)
    ym = x1.mean(axis=0)
    xc = x0 - xm
    yc = x1 - ym
    b = (xc * yc).sum(axis=0) / (xc * xc).sum(axis=0)
    a = ym - b * xm
    resid = yc - b * xc
    resid_std = np.sqrt((resid ** 2).sum(axis=0) / (n - 2))
    return a, b, resid_std


def run_backtest(prices: pd.DataFrame, gross_leverage: float = 1.0):
    """
    Returns:
      equity          - pd.Series of cumulative equity (starting at 1.0)
      daily_pnl       - pd.Series of daily strategy return
      trade_log       - pd.DataFrame of per-day diagnostics (n_active, n_factors, etc.)
    """
    log_ret = np.log(prices).diff()
    dates = prices.index
    T = len(dates)

    n_factors_used = np.full(T, np.nan)
    n_coins_used = np.full(T, np.nan)
    n_active_positions = np.zeros(T)

    strat_ret = pd.Series(0.0, index=dates)
    turnover = pd.Series(0.0, index=dates)  # sum(|w_new - w_old|), one-way notional traded

    # stateful position per coin: 0 = flat, 1 = long residual, -1 = short residual
    pos_state = {c: 0 for c in prices.columns}
    prev_weights = {c: 0.0 for c in prices.columns}

    start = PCA_WINDOW + REG_WINDOW + 1
    for t in range(start, T):
        window_pca = log_ret.iloc[t - PCA_WINDOW:t]           # for factor construction
        window_reg = log_ret.iloc[t - REG_WINDOW:t]           # for beta / OU fit

        # coins with a fully populated PCA window
        full_cols = window_pca.columns[window_pca.notna().all()]
        if len(full_cols) < MIN_COINS_FOR_PCA:
            continue

        R = window_pca[full_cols].values  # (PCA_WINDOW, N)
        sigma = R.std(axis=0)
        sigma[sigma == 0] = np.nan
        Rstd = (R - R.mean(axis=0)) / sigma
        valid = ~np.isnan(Rstd).any(axis=0)
        cols = full_cols[valid]
        Rstd = Rstd[:, valid]
        sigma = sigma[valid]
        N = Rstd.shape[1]
        if N < MIN_COINS_FOR_PCA:
            continue

        # ---- PCA on the correlation matrix ----
        corr = np.corrcoef(Rstd, rowvar=False)
        eigval, eigvec = np.linalg.eigh(corr)          # ascending
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = eigval[order], eigvec[:, order]
        eigval = np.clip(eigval, 0, None)
        cum_var = np.cumsum(eigval) / eigval.sum()
        m = int(np.searchsorted(cum_var, VAR_EXPLAIN) + 1)
        m = int(np.clip(m, MIN_FACTORS, min(MAX_FACTORS, N - 1)))

        Q = eigvec[:, :m]                              # (N, m) eigenportfolio weights on std. returns

        # eigenportfolio returns over the (longer) PCA window: F = Rstd @ Q, but we
        # need factor returns aligned with REG_WINDOW for the beta regression, so
        # rebuild raw-return eigenportfolio weights: w_i = Q_ij / sigma_i
        W = Q / sigma[:, None]                          # (N, m)

        reg_cols = [c for c in cols if window_reg[c].notna().all()]
        if len(reg_cols) < MIN_COINS_FOR_PCA:
            continue
        idx_map = {c: i for i, c in enumerate(cols)}
        sel = [idx_map[c] for c in reg_cols]
        W_sel = W[sel, :]

        Rreg = window_reg[reg_cols].values               # (REG_WINDOW, n)
        F = Rreg @ W_sel                                  # (REG_WINDOW, m) eigenportfolio returns

        # ---- single matrix OLS: regress every coin's returns on F at once ----
        Xd = np.column_stack([np.ones(REG_WINDOW), F])    # (REG_WINDOW, m+1)
        beta, *_ = np.linalg.lstsq(Xd, Rreg, rcond=None)  # (m+1, n)
        fitted = Xd @ beta
        eps = Rreg - fitted                                # (REG_WINDOW, n) idiosyncratic residuals

        Xcum = np.cumsum(eps, axis=0)                      # auxiliary OU process per coin

        a, b, resid_std = _fit_ar1_vectorized(Xcum)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.clip(b, -0.999, 0.999)
            kappa = -ANNUALIZATION * np.log(b.clip(min=1e-6))
            theta = a / (1 - b)
            sigma_eq = resid_std * np.sqrt(1.0 / (1 - b ** 2))
            s_score = (Xcum[-1, :] - theta) / sigma_eq

        tradeable = (kappa > MIN_KAPPA) & np.isfinite(s_score) & (sigma_eq > 0)

        # signal is known as of the close of day (t-1); it is traded on the very
        # next available return, log_ret.iloc[t] (close t-1 -> close t).
        fwd_ret = log_ret[reg_cols].iloc[t].values

        fwd_ret_full_row = log_ret.iloc[t]
        base_cols_full = list(cols)  # coins used to build the factors
        fwd_vals = fwd_ret_full_row[base_cols_full].values
        if np.isnan(fwd_vals).any():
            fwd_vals = np.nan_to_num(fwd_vals, nan=0.0)
        fwd_factor_ret = fwd_vals @ W                     # (m,)

        alpha = beta[0, :]                                  # (n,) regression intercept
        beta_factors = beta[1:, :]                          # (m, n)
        idio_fwd = fwd_ret - alpha - beta_factors.T @ fwd_factor_ret  # (n,) residual return realized on day t

        # ---- stateful signal: hold a position until its exit threshold fires ----
        signal = np.zeros(len(reg_cols))
        for i, c in enumerate(reg_cols):
            cur = pos_state.get(c, 0)
            s = s_score[i]
            if not np.isfinite(s):
                signal[i] = 0.0
                pos_state[c] = 0
                continue
            if cur == 0:
                if tradeable[i] and s < -ENTRY_S:
                    cur = 1
                elif tradeable[i] and s > ENTRY_S:
                    cur = -1
            elif cur == 1:
                if s > EXIT_LONG_S:
                    cur = 0
            elif cur == -1:
                if s < EXIT_SHORT_S:
                    cur = 0
            pos_state[c] = cur
            signal[i] = cur

        active = signal != 0.0
        n_active = active.sum()
        n_factors_used[t] = m
        n_coins_used[t] = N
        n_active_positions[t] = n_active

        if n_active > 0:
            w = gross_leverage * signal[active] / n_active
            pnl_t = np.dot(w, idio_fwd[active])
            strat_ret.iloc[t] = pnl_t

            new_weights = {c: 0.0 for c in reg_cols}
            for c, wi in zip(np.array(reg_cols)[active], w):
                new_weights[c] = wi
            tno = 0.0
            for c, wi in new_weights.items():
                tno += abs(wi - prev_weights.get(c, 0.0))
                prev_weights[c] = wi
            for c in list(prev_weights.keys()):
                if c not in new_weights and prev_weights[c] != 0.0:
                    tno += abs(prev_weights[c])
                    prev_weights[c] = 0.0
            turnover.iloc[t] = tno

    equity = (1 + strat_ret).cumprod()
    trade_log = pd.DataFrame({
        "n_factors": n_factors_used,
        "n_coins": n_coins_used,
        "n_active": n_active_positions,
        "turnover": turnover.values,
    }, index=dates)
    return equity, strat_ret, trade_log


def apply_costs(strat_ret: pd.Series, turnover: pd.Series, cost_bps_oneway: float) -> pd.Series:
    """Subtract a simple linear transaction-cost model: cost_bps_oneway * turnover."""
    cost = turnover * (cost_bps_oneway / 1e4)
    return strat_ret - cost


def performance_summary(daily_ret: pd.Series) -> dict:
    r = daily_ret[daily_ret.index >= daily_ret.first_valid_index()]
    r = r.fillna(0.0)
    ann_ret = r.mean() * ANNUALIZATION
    ann_vol = r.std() * np.sqrt(ANNUALIZATION)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate = (r[r != 0] > 0).mean() if (r != 0).any() else np.nan
    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate_active_days": hit_rate,
        "total_return": equity.iloc[-1] - 1,
        "n_days": len(r),
    }
