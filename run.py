import sys, time
sys.path.insert(0, "/home/claude/ou_arb")
import numpy as np
import pandas as pd
from ou_strategy import load_and_clean, run_backtest, performance_summary, apply_costs

DATA_PATH = "crypto_1000_1d_2017-01-01_2026-09-09_px.pk"

print("Loading & cleaning data...")
prices = load_and_clean(DATA_PATH)
print("Universe size:", prices.shape[1], " | Date range:", prices.index[0], "->", prices.index[-1])

t0 = time.time()
equity, daily_ret, trade_log = run_backtest(prices, gross_leverage=1.0)
print(f"Backtest done in {time.time()-t0:.1f}s")

print("\n--- Gross (no transaction costs) ---")
summary = performance_summary(daily_ret)
for k, v in summary.items():
    print(f"{k:>24s}: {v}")

avg_daily_turnover = trade_log["turnover"].mean()
print(f"\nAvg daily one-way turnover (fraction of gross book): {avg_daily_turnover:.3f}")

results = {"gross": daily_ret}
for bps in [5, 10, 25, 50]:
    net_ret = apply_costs(daily_ret, trade_log["turnover"], bps)
    results[f"net_{bps}bps"] = net_ret
    s = performance_summary(net_ret)
    print(f"\n--- Net of {bps} bps one-way cost ---")
    for k, v in s.items():
        print(f"{k:>24s}: {v}")

pd.DataFrame(results).to_csv("/home/claude/ou_arb/daily_returns_by_cost.csv")
equity.to_csv("/home/claude/ou_arb/equity_curve.csv")
trade_log.to_csv("/home/claude/ou_arb/trade_log.csv")
print("\nSaved equity_curve.csv, daily_returns_by_cost.csv, trade_log.csv")
