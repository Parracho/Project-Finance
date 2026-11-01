import numpy as np
def largest_complete_rectangle(df, chunk=20000):
    """Same result as the O(T^2 N) loop, in O(T N log N), fully vectorised.

    For every end row e and stock j, run_start[e, j] is the first row of the
    unbroken run of non-NaN values ending at e (or "infinity" if NaN at e).
    Stock j is complete on [s, e] iff run_start[e, j] <= s, so sorting the run
    starts at row e gives, for the k-th smallest start s, a window with k stocks.
    """
    avail = df.notna().to_numpy()
    T, N = avail.shape
    if T == 0 or N == 0 or not avail.any():
        raise ValueError("no non-NaN data")

    # last NaN position at or before each row -> run start = that + 1
    rows = np.arange(T, dtype=np.int32)[:, None]
    last_nan = np.maximum.accumulate(np.where(avail, -1, rows).astype(np.int32), axis=0)
    run_start = np.where(avail, last_nan + 1, T).astype(np.int32)   # T = "unavailable"
    del last_nan

    best = (0, None, None)       # area, start, end
    k = np.arange(1, N + 1, dtype=np.int64)[None, :]
    for c0 in range(0, T, chunk):                       # chunk to bound memory
        c1 = min(c0 + chunk, T)
        rs = np.sort(run_start[c0:c1], axis=1).astype(np.int64)
        e = np.arange(c0, c1, dtype=np.int64)[:, None]
        area = np.where(rs < T, (e - rs + 1) * k, 0)
        m = area.max()
        if m >= best[0] and m > 0:
            # tie-break like the original loop: smallest start, then smallest end
            ei, ki = np.nonzero(area == m)
            starts, ends = rs[ei, ki], ei + c0
            o = np.lexsort((ends, starts))[0]
            cand = (int(m), int(starts[o]), int(ends[o]))
            if m > best[0] or (cand[1], cand[2]) < (best[1], best[2]):
                best = cand

    area, s, e = best
    stocks = np.flatnonzero(avail[s:e + 1].all(axis=0))
    assert (e - s + 1) * len(stocks) == area
    return {
        "start_date": df.index[s],
        "end_date": df.index[e],
        "dates": df.index[s:e + 1],
        "stocks": df.columns[stocks],
        "n_dates": e - s + 1,
        "n_stocks": len(stocks),
        "area": area,
    }
def ret_treatment(df):
    df = df.dropna(axis=0, how="all") 
    df = df.dropna(axis=1, how="all") 
    rect=largest_complete_rectangle(df)
    df = df[rect["stocks"]].loc[rect["dates"]]
    return df