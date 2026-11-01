import numpy as np
import pandas as pd

def annualization_factor(interval: str = "30m", asset_type: str = "crypto", trading_hours_per_day: float = 6.5):
    s = interval.strip().lower()
    if s.endswith("m") and not s.endswith("min"):
        s = s[:-1] + "min"

    delta = pd.Timedelta(s)
    bar_hours = delta.total_seconds() / 3600

    if asset_type == "crypto":
        days_per_year = 365
        hours_per_day = 24
    elif asset_type == "stock":
        days_per_year = 252
        hours_per_day = trading_hours_per_day
    else:
        raise ValueError("asset_type must be 'crypto' or 'stock'")

    # daily or longer bars
    if bar_hours >= 24:
        bar_days = bar_hours / 24
        periods_per_year = days_per_year / bar_days
    else:
        bars_per_day = hours_per_day / bar_hours
        periods_per_year = days_per_year * bars_per_day

    return periods_per_year