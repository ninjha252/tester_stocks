# app.py
import sys, os
sys.path.insert(0, os.path.abspath("."))     # repo root importable
sys.path.insert(0, os.path.abspath(".."))    # if running from notebooks/

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests import HTTPError

# project modules
import src.binance_downloader as bd
from src.binance_downloader import fetch_klines
from src.feature_engineering import build_features, make_labels
from src.walkforward import WalkForwardConfig, run_walkforward
from src.utils import bps_to_frac

st.set_page_config(page_title="Intraday ML Backtester", layout="wide")

# ---------- helpers ----------
def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

@st.cache_data(show_spinner=False)
def get_bars(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp):
    """Try Binance global then Binance US; return (df, host)."""
    hosts = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]
    last_err = None
    for host in hosts:
        try:
            bd.BASE = host
            df = fetch_klines(symbol, interval, start, end)
            if len(df) > 0:
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                return df, host
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("All endpoints returned empty data")

def backtest_fixed_horizon(df: pd.DataFrame,
                           proba_col: str,
                           long_threshold: float = 0.60,
                           short_threshold: float = 0.0,
                           cost_bps: int = 5,
                           slippage_bps: int = 3,
                           hold_minutes: int = 5):
    """Event-driven: enter on threshold; hold exactly k bars; exit; pay entry/exit costs."""
    close = df["close"].astype(float)
    ret1 = np.log(close).diff().fillna(0.0)

    n = len(df)
    pos = np.zeros(n, dtype=int)   # -1/0/+1
    holding = 0
    current_pos = 0
    one_way_cost = bps_to_frac(cost_bps) + bps_to_frac(slippage_bps)

    for i in range(n):
        p = df[proba_col].iloc[i]
        if holding == 0:
            if pd.notna(p):
                if p >= long_threshold:
                    current_pos = 1; holding = hold_minutes
                elif short_threshold > 0 and p <= short_threshold:
                    current_pos = -1; holding = hold_minutes
                else:
                    current_pos = 0
        else:
            holding -= 1
            if holding == 0:
                pass
        pos[i] = current_pos
        if holding == 0 and current_pos != 0:
            current_pos = 0

    pos = pd.Series(pos, index=df.index, name="pos")
    pnl = pos.shift(1).fillna(0) * ret1
    trade = pos.diff().abs().fillna(pos.abs())  # entries/exits (1 each), flips = 2
    cost = trade * one_way_cost
    pnl = pnl - cost

    out = pd.DataFrame({"ret1": ret1, "pos": pos, "pnl": pnl}, index=df.index)
    out["equity"] = (1 + out["pnl"]).cumprod()

    r = out["pnl"]
    ann_factor = np.sqrt(365*24*60)  # minute bars per year
    sharpe_like = float(r.mean() / (r.std() + 1e-12) * ann_factor)
    max_dd = float((out["equity"] / out["equity"].cummax() - 1).min())
    entries = int(((pos.shift(1).fillna(0) == 0) & (pos != 0)).sum())
    metrics = {
        "Sharpe_like": sharpe_like,
        "MaxDD": max_dd,
        "Entries": entries,
        "TotalBars": int(len(out)),
        "FinalEquity": float(out["equity"].iloc[-1]) if len(out) else 1.0,
    }
    return out, metrics

# ---------- UI ----------
st.title("Intraday ML Backtester (1-min)")
with st.sidebar:
    st.header("Data")
    symbol = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"], index=0)
    interval = st.selectbox("Interval", ["1m"], index=0)
    end = pd.Timestamp.utcnow().floor("min")
    start = st.date_input("Start date (UTC)", (end - pd.Timedelta(days=14)).date())
    end_date = st.date_input("End date (UTC)", end.date())
    start_ts = to_utc(pd.Timestamp(start))
    end_ts = to_utc(pd.Timestamp(end_date)) + pd.Timedelta(hours=23, minutes=59)

    st.header("Labels & Costs")
    horizon = st.number_input("Hold minutes (k)", 1, 60, 5)
    cost_bps = st.number_input("Fees (bps, one-way)", 0, 200, 3)
    slip_bps = st.number_input("Slippage (bps, one-way)", 0, 200, 2)

    st.header("Model/Signals")
    long_th = st.slider("Long threshold p*", 0.50, 0.80, 0.60, 0.01)
    allow_shorts = st.checkbox("Allow shorts", value=False)
    short_th = st.slider("Short threshold p*", 0.20, 0.50, 0.40, 0.01, disabled=not allow_shorts)

    st.header("Walk-Forward")
    min_train_days = st.number_input("Min train days", 1, 30, 3)
    test_window_days = st.number_input("Test window days", 1, 7, 1)
    embargo_min = st.number_input("Embargo minutes", 0, 120, 10)

    run = st.button("Run backtest", use_container_width=True)

# ---------- Pipeline ----------
if run:
    try:
        with st.spinner("Downloading bars..."):
            df, host = get_bars(symbol, interval, start_ts, end_ts)
        st.caption(f"Source: {host} | Bars: {len(df)}")

    except Exception as e:
        st.warning(f"Data download failed ({e}). Generating synthetic series so you can still test.")
        idx = pd.date_range(start_ts, end_ts, freq="1min", inclusive="left")
        price = 30000 + np.cumsum(np.random.normal(0, 10, len(idx)))
        df = pd.DataFrame({
            "open":  price + np.random.normal(0,1,len(idx)),
            "high":  price + np.random.uniform(0,5,len(idx)),
            "low":   price - np.random.uniform(0,5,len(idx)),
            "close": price + np.random.normal(0,1,len(idx)),
            "volume": np.abs(np.random.normal(100,20,len(idx))),
            "number_of_trades": np.random.randint(50,150,len(idx)),
        }, index=idx)
        df.index = df.index.tz_localize("UTC")

    with st.spinner("Feature engineering..."):
        fdf = build_features(
            df,
            rsi_periods=(7,14),
            macd=(12,26,9),
            atr_period=14,
            vol_windows=(5,15,60),
            vwap_windows=(5,15,60),
        )
        fdf = make_labels(fdf, horizon_minutes=int(horizon), cost_frac=bps_to_frac(cost_bps))
        fdf = fdf.dropna()

    excluded = {"y","fwd_ret"}
    base_exclude = {"open","high","low","close"}
    feat_cols = [c for c in fdf.columns if c not in excluded and c not in base_exclude]
    st.caption(f"Using {len(feat_cols)} features")

    with st.spinner("Training (walk-forward)..."):
        wf_cfg = WalkForwardConfig(
            test_window_days=int(test_window_days),
            min_train_days=int(min_train_days),
            embargo_minutes=int(embargo_min),
            features=feat_cols,
            target_col="y",
        )
        res = run_walkforward(fdf, wf_cfg)

    if len(res["models"]) == 0:
        st.info("No folds created with current dates/settings. Reduce 'Min train days' or widen the date range.")
    else:
        # attach OOF preds
        fdf = fdf.join(res["oof_pred"].rename("proba"))
        # backtest (fixed horizon)
        bt, metr = backtest_fixed_horizon(
            fdf.dropna(subset=["proba","close"]),
            proba_col="proba",
            long_threshold=float(long_th),
            short_threshold=float(short_th if allow_shorts else 0.0),
            cost_bps=int(cost_bps),
            slippage_bps=int(slip_bps),
            hold_minutes=int(horizon),
        )

        # metrics summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Equity", f"{metr['FinalEquity']:.3f}")
        col2.metric("Sharpe_like", f"{metr['Sharpe_like']:.2f}")
        col3.metric("Max Drawdown", f"{metr['MaxDD']:.2%}")
        col4.metric("Entries", f"{metr['Entries']}")

        # equity plot
        fig, ax = plt.subplots(figsize=(10,4))
        bt["equity"].plot(ax=ax)
        ax.set_title("Equity curve (fixed-horizon, no overlap)")
        ax.set_xlabel("Time"); ax.set_ylabel("Equity")
        st.pyplot(fig, clear_figure=True)

        # folds table
        st.subheader("Walk-forward fold metrics")
        st.dataframe(res["metrics"], use_container_width=True)

        # feature importance (last model)
        import pandas as pd
        fi = pd.DataFrame({
            "feature": feat_cols,
            "importance": res["models"][-1].feature_importances_
        }).sort_values("importance", ascending=False).head(25)
        st.subheader("Top features (last fold)")
        st.dataframe(fi, use_container_width=True)

        st.caption("Note: Results are backtested on past data with your selected costs; not predictive of future returns.")
else:
    st.info("Set your parameters in the sidebar and click **Run backtest**.")
