# Crypto Cross-Sectional Research

Research notebooks exploring **cross-sectional return predictability, market activity and factor structure** in the crypto market, using Binance spot data (USDT pairs).

The work is inspired by the intraday cross-sectional approach in *"Are You Trading Predictably?"*, adapted to crypto: a market that trades 24/7, with hundreds of liquid assets and very different volatility regimes.

> ⚠️ **Status:** exploratory research code. Notebooks contain scratch sections ("Trash") and are not production-ready. Nothing here is financial advice.

---

## Repository contents

| Notebook | What it does |
|---|---|
| `reversal_copy.ipynb` | Data download pipeline, lagged cross-sectional regressions (momentum / reversal), and a volatility-band analysis |
| `alpha_beta.ipynb` | Splits each coin's return into market (beta) and idiosyncratic (alpha) parts; builds momentum / reversal / size factors with Fama-MacBeth-style tests |
| `activity.ipynb` | Builds a market-wide volume "activity" signal from rolling z-scores of volume |

---

## Data

All data comes from the public **Binance REST API** (`/api/v3/ticker/24hr` and `/api/v3/klines`), no API key required.

- **Universe:** the top N `*USDT` symbols ranked by 24h quote volume (up to 1000).
- **Fields:** close price and volume at intervals such as `30m`, `1h`, `8h`, `1d`.
- **Caching:** downloads are saved as pickles named `crypto_{N}_{interval}_{start}_{end}_px.pk` (and `_vol.pk`), and `find_existing_file` reuses any cached file that fully covers the requested period.
- Pagination handles Binance's 1000-candle limit, with a short sleep between requests.

---

## Notebook details

### 1. `reversal_copy.ipynb`: momentum & reversal

**Cross-sectional regression** at each time *t* and lag *k*:

```
r(i,t) = α(k,t) + γ(k,t) · r(i,t-k) + u(i,t)
```

Averaging `γ` over time for each lag `k` shows which past horizons predict current returns: positive values suggest momentum, negative values suggest reversal. The result is plotted as a normalized bar chart by lag.

**Volatility band analysis:** annualized volatility is computed per coin (rolling window), then for each day the notebook finds the *densest band* of the cross-sectional volatility distribution (the narrowest percentile interval holding a given mass). It tracks how often coins enter and leave this band, and how the band drifts across market regimes (2017 → 2026).

### 2. `alpha_beta.ipynb`: alpha/beta decomposition & factors

- **Market model:** regresses each coin on the equal-weighted market return, then separates cumulative return into a *beta contribution* and an *alpha contribution* (constant + residual).
- **Largest complete rectangle:** finds the biggest dates × coins block with no missing data, so regressions run on a clean panel.
- **Risk factors:** builds `MOM`, `REV` and `SIZE` characteristics, winsorized and z-scored cross-sectionally, with GPU-accelerated (CuPy) daily cross-sectional regressions.
- **Tests:** Fama-MacBeth-style average payoff with HAC (Newey-West) t-stats, plus factor persistence via an AR(1) regression.
- **Long-short factors:** top vs. bottom 1% portfolios for each characteristic, combined with the market factor.

### 3. `activity.ipynb`: volume activity signal

Computes a rolling volume z-score per coin (500-period window, lagged one period to avoid look-ahead), then builds four market-wide signals:

1. **Activity:** mean volume z-score across coins
2. **Returns:** mean cross-sectional return
3. **Sign activity:** activity × sign of market return
4. **Return activity:** activity × market return

---

## Requirements

- Python 3.10+
- `pandas`, `numpy`, `requests`, `matplotlib`, `statsmodels`
- `python-binance` (imported in the notebooks)
- *(optional, GPU factor regressions)* `cupy`

```bash
pip install pandas numpy requests matplotlib statsmodels python-binance
# optional, for GPU acceleration (match your CUDA version)
pip install cupy-cuda12x
```

### Local modules referenced

`alpha_beta.ipynb` imports two helper modules that are **not** included in the notebooks themselves:

- `get_klines.py`: data download / caching helpers
- `momentum_reversal_gpu.py`: GPU lag-regression routine

Add these to the repo (the functions are developed in `reversal_copy.ipynb` and `activity.ipynb`) or the notebook will not run as-is.

---

## Quick start

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter lab
```

1. Open `reversal_copy.ipynb` and run the data-download cells (the first run is slow because of API rate limits; later runs use the cached pickles).
2. Run `momentum_reversal(k_max, '30m', start, end, 1000)` to reproduce the lag-response plot.
3. Open `alpha_beta.ipynb` or `activity.ipynb` to continue with factors and activity signals.

---

## Notes & caveats

- The universe is built from **today's** top-volume symbols, so results carry **survivorship bias** (delisted coins are missing).
- Binance only lists coins from their listing date, so panels are unbalanced; the "largest complete rectangle" step trades sample breadth for completeness.
- Several cells have hard-coded dates and file paths (e.g. `datasets/...`); adjust them to your setup.
- Pickled datasets are large and should not be committed. Add `*.pk` to `.gitignore`.

---

## Roadmap

- [ ] Move shared helpers into a proper package (`get_klines.py`, etc.)
- [ ] Clean out the scratch ("Trash") sections
- [ ] Add transaction costs and a backtest of the signals
- [ ] Out-of-sample and survivorship-bias checks

---

## License

Add a license of your choice (e.g. MIT).
