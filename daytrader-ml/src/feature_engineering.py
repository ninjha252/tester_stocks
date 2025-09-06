
import numpy as np
import pandas as pd

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    pv = price * volume
    num = pv.rolling(window).sum()
    den = volume.rolling(window).sum()
    return num / (den + 1e-12)

def build_features(df: pd.DataFrame,
                   rsi_periods=(7,14),
                   macd=(12,26,9),
                   atr_period=14,
                   vol_windows=(5,15,60),
                   vwap_windows=(5,15,60)) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = np.log(out["close"]).diff()
    out["ret_5"] = np.log(out["close"]).diff(5)
    out["ret_15"] = np.log(out["close"]).diff(15)
    out["ret_60"] = np.log(out["close"]).diff(60)

    for w in vol_windows:
        out[f"rv_{w}"] = out["ret_1"].rolling(w).std()
        out[f"roll_mean_{w}"] = out["ret_1"].rolling(w).mean()
        out[f"roll_kurt_{w}"] = out["ret_1"].rolling(w).kurt()

    for p in rsi_periods:
        out[f"rsi_{p}"] = _rsi(out["close"], p)

    fast, slow, signal = macd
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    out["atr"] = _atr(out, atr_period)

    for w in vwap_windows:
        vw = vwap(out["close"], out["volume"], w)
        out[f"vwap_dev_{w}"] = (out["close"] - vw) / (vw + 1e-12)

    for w in vol_windows:
        out[f"vol_z_{w}"] = (out["volume"] - out["volume"].rolling(w).mean()) / (out["volume"].rolling(w).std() + 1e-9)

    out["hl_range"] = (out["high"] - out["low"]) / (out["close"].shift(1) + 1e-12)
    out["upper_shadow"] = (out["high"] - out[["close","open"]].max(axis=1)) / (out["atr"] + 1e-12)
    out["lower_shadow"] = (out[["close","open"]].min(axis=1) - out["low"]) / (out["atr"] + 1e-12)

    out["minute"] = out.index.tz_convert("UTC").minute
    out = pd.get_dummies(out, columns=["minute"], drop_first=True)
    return out

def make_labels(df: pd.DataFrame, horizon_minutes: int = 5, cost_frac: float = 0.0003) -> pd.DataFrame:
    out = df.copy()
    fwd_close = out["close"].shift(-horizon_minutes)
    out["fwd_ret"] = np.log((fwd_close + 1e-12) / (out["close"] + 1e-12))
    out["y"] = (out["fwd_ret"] > cost_frac).astype(int)
    return out
