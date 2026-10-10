import pandas as pd
import requests as _requests
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------------
WORKERS = 6     # symbols downloaded at the same time (try 4-8)


# -------------------------------------------------------------------------
# SHARED RATE-LIMIT PAUSE (so your functions below stay as you wrote them)
#
# Binance's limit is per IP, so all workers share it. This sits in front of
# requests.get: if ANY worker gets a 429/418, ALL workers pause for the
# Retry-After time. Your own try/wait code below still runs exactly as before.
# -------------------------------------------------------------------------
_pause_until = 0.0
_pause_lock = threading.Lock()


def _shared_get(url, params=None, **kwargs):
    global _pause_until

    while True:
        wait = _pause_until - time.time()
        if wait <= 0:
            break
        time.sleep(min(wait, 5))

    response = _requests.get(url, params=params, **kwargs)

    if response.status_code in (418, 429):
        retry_after = int(response.headers.get("Retry-After", 60))
        with _pause_lock:
            _pause_until = max(_pause_until, time.time() + retry_after)

    return response


class _RequestsProxy:
    get = staticmethod(_shared_get)

    def __getattr__(self, name):
        return getattr(_requests, name)   # ConnectionError, Timeout, ...


requests = _RequestsProxy()


# -------------------------------------------------------------------------
# 3rd step: downloads ONE symbol (your function)
# -------------------------------------------------------------------------
def get_all_klines(symbol, interval, start_date, end_date):
    
    url = "https://api.binance.com/api/v3/klines"

    columns = [
        'open_time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_volume',
        'trades',
        'taker_buy_base',
        'taker_buy_quote',
        'ignore'
    ]

    start_time = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
    end_time = int(pd.Timestamp(end_date, tz='UTC').timestamp() * 1000)

    if interval.endswith("M")==False:
        interval_time = int(pd.Timedelta(interval).total_seconds() * 1000)

    all_data = []

    while start_time < end_time:
            # Convert dates to milliseconds
        if interval.endswith("M"):
            start_time = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        while True:
            try:
                response = requests.get(url, params=params, timeout=15)
                if response.status_code in (418,429):
                    wait = int(response.headers.get("Retry-After",60))
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if response.status_code >= 500:
                    print(f"Server error {response.status_code}. Waiting 30s...")
                    time.sleep(30)
                    continue

                response.raise_for_status()
                break

            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"Connection problem ({e}). Waiting 30s...")
                time.sleep(30)

        batch = response.json()

        # Stop when there is no more data
        if not batch:
            break

        # Add this batch to our collection
        all_data.extend(batch)

        # Last candle's opening timestamp
        last_open_time = batch[-1][0]
        if interval.endswith("M"):
            start_date = pd.Timestamp(start_date)
            months = int(interval[:-1]) 
            start_date += pd.DateOffset(months=months)
        # Move to the next millisecond to avoid duplicates
        else:
            start_time = last_open_time + interval_time

        print(f"{symbol}: downloaded {len(all_data)} candles")

        # If fewer than 1000 were returned, we reached the end
        if len(batch) < 1000:
            break

    # Combine everything into ONE DataFrame
    df = pd.DataFrame(all_data, columns=columns)

    # Remove any possible duplicate timestamps
    df = df.drop_duplicates(subset='open_time')

    # Convert timestamps
    df['open_time'] = pd.to_datetime(
        df['open_time'],
        unit='ms',
        utc=True
    )

    df['close_time'] = pd.to_datetime(
        df['close_time'],
        unit='ms',
        utc=True
    )

    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    return df


# -------------------------------------------------------------------------
# 2nd step: downloads all symbols (your function + workers)
# -------------------------------------------------------------------------
def get_prices(t_interval, start_date, end_date, until):
    
    symbols = oldest_symbols[:until]

    url = "https://api.binance.com/api/v3/klines"

    columns = [
        'open_time', 'open', 'high', 'low', 'close',
        'volume', 'close_time', 'quote_volume',
        'trades', 'taker_buy_base', 'taker_buy_quote',
        'ignore'
    ]

    # -------------------------------------------------
    # 1. Download prices (several symbols at once)
    # -------------------------------------------------

    data = {}

    # Every finished symbol is saved here, so if the script stops you can run
    # it again and the finished symbols are loaded instead of re-downloaded.
    cache_dir = Path(f"./cache_{t_interval}_{pd.Timestamp(start_date).date()}_{pd.Timestamp(end_date).date()}")
    cache_dir.mkdir(exist_ok=True)

    def download_one(symbol):
        cache_file = cache_dir / f"{symbol}.pk"

        if cache_file.exists():
            return symbol, pd.read_pickle(cache_file), "loaded from cache"

        df = get_all_klines(symbol, t_interval, start_date, end_date)

        # keep only what is used below (saves a lot of memory)
        df = df[['open_time', 'close', 'volume']]

        # write to a temp file first so a crash never leaves a half-written file
        tmp_file = cache_dir / f"{symbol}.tmp"
        df.to_pickle(tmp_file)
        tmp_file.replace(cache_file)

        return symbol, df, f"done ({len(df)} candles)"

    failed = []
    finished = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, s): s for s in symbols}

        for fut in as_completed(futures):
            symbol = futures[fut]
            finished += 1
            try:
                symbol, df, status = fut.result()
                data[symbol] = df
                print(f"[{finished}/{len(symbols)}] {symbol}: {status}")
            except Exception as e:
                print(f"[{finished}/{len(symbols)}] {symbol}: FAILED ({e.__class__.__name__}: {e})")
                failed.append(symbol)

    if failed:
        raise RuntimeError(
            f"{len(failed)} symbols failed: {failed}. "
            f"The finished ones are saved in {cache_dir}, run again to retry."
        )

    # -------------------------------------------------
    # 2. Construct price and return matrices
    # -------------------------------------------------

    # Use open_time as index so all assets align by time
    vol = pd.concat(
            {
                symbol: data[symbol].set_index('open_time')['volume']
                for symbol in symbols
            },
            axis=1
        )
    
    px = pd.concat(
            {
                symbol: data[symbol].set_index('open_time')['close']
                for symbol in symbols
            },
            axis=1
        )
    
    Path("./datasets").mkdir(exist_ok=True)
    vol.to_pickle(f'./datasets/crypto_{until}_{t_interval}_{pd.Timestamp(start_date).date()}_{pd.Timestamp(end_date).date()}_vol.pk')
    px.to_pickle(f'./datasets/crypto_{until}_{t_interval}_{pd.Timestamp(start_date).date()}_{pd.Timestamp(end_date).date()}_px.pk')
    
    return px,vol


def find_existing_file(t_interval, start_date, end_date,until,x):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    pattern = f"crypto_{until}_{t_interval}_*_{x}.pk"

    for path in Path("./datasets").glob(pattern):
        try:
            # crypto_{interval}_{start}_{end}_{until}_px.pk
            parts = path.stem.split("_")

            file_start = pd.Timestamp(parts[3])
            file_end = pd.Timestamp(parts[4])

            if file_start <= start_date and file_end >= end_date:
                print(f"Using existing data: {path}") 
                if x=="vol":
                    return find_existing_file(t_interval, start_date, end_date,until,"px"), pd.read_pickle(path)
                else:
                    return pd.read_pickle(path)

        except (ValueError, IndexError):
            continue

    
    print("Downloading data...")

    return get_prices(t_interval, start_date, end_date,until)