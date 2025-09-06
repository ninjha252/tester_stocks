
import time
import requests
import pandas as pd
from .utils import to_ms

BASE = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str, start, end, max_retries=5, pause=0.5) -> pd.DataFrame:
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    frames = []

    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000
        }
        data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(BASE, params=params, timeout=15)
                if resp.status_code == 429:
                    time.sleep(2.0 + attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.0 + attempt)

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["number_of_trades"] = df["number_of_trades"].astype(int)

        frames.append(df[["open_time","open","high","low","close","volume","number_of_trades"]])
        last_open_time = int(data[-1][0])
        next_start = last_open_time + 1
        if next_start >= end_ms:
            break
        start_ms = next_start
        time.sleep(pause)

    if not frames:
        return pd.DataFrame(columns=["open_time","open","high","low","close","volume","number_of_trades"])

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")
    out = out.set_index("open_time")
    return out
