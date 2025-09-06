
import time
import math
import pandas as pd
import numpy as np

def to_ms(dt_like):
    if isinstance(dt_like, (int, float, np.integer, np.floating)):
        return int(dt_like)
    if isinstance(dt_like, str):
        ts = pd.Timestamp(dt_like, tz="UTC")
    elif isinstance(dt_like, pd.Timestamp):
        ts = dt_like.tz_localize("UTC") if dt_like.tz is None else dt_like.tz_convert("UTC")
    else:
        ts = pd.Timestamp(dt_like, tz="UTC")
    return int(ts.value // 1_000_000)

def bps_to_frac(bps):
    return bps / 10_000.0

def safe_shift(s, n):
    return pd.Series(s).shift(n)

def realized_vol(ret, window):
    return ret.rolling(window).std() * np.sqrt(60)
