# app.py
import sys, os, time, base64
from pathlib import Path
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent

# Make repo importable whether run from root or /notebooks
sys.path.insert(0, os.path.abspath("."))      # repo root
sys.path.insert(0, os.path.abspath(".."))     # parent (for safety)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests import HTTPError

# ML
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Notebook execution
import papermill as pm
import nbformat

# Project modules
import src.binance_downloader as bd
from src.binance_downloader import fetch_klines
from src.feature_engineering import build_features, make_labels
from src.walkforward import WalkForwardConfig, run_walkforward
from src.utils import bps_to_frac

st.set_page_config(page_title="Intraday ML Backtester", layout="wide")

# ======================
# Helpers / Core logic
# ======================

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Return a UTC-aware Timestamp whether input is naive or tz-aware."""
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

@st.cache_data(show_spinner=False)
def get_bars(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp):
    """
    Try Binance global then Binance US; return (df, host). DataFrame indexed by UTC.
    """
    hosts = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]
    last_err = None
    for host in hosts:
        try:
            bd.BASE = host  # tell our downloader which host to hit
            df = fetch_klines(symbol, interval, start, end)
            if len(df) > 0:
                for c in ["open", "high", "low", "close", "volume"]:
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
    """
    Event-driven backtest that:
      - enters when proba crosses threshold (from flat)
      - holds exactly k bars then exits
      - accrues 1-min returns while holding
      - charges entry & exit costs
    """
    close = df["close"].astype(float)
    ret1 = np.log(close).diff().fillna(0.0)

    n = len(df)
    pos = np.zeros(n, dtype=int)        # -1/0/+1
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
    trade = pos.diff().abs().fillna(pos.abs())     # entries/exits; flips=2
    cost = trade * one_way_cost
    pnl = pnl - cost

    out = pd.DataFrame({"ret1": ret1, "pos": pos, "pnl": pnl}, index=df.index)
    out["equity"] = (1 + out["pnl"]).cumprod()

    r = out["pnl"]
    ann_factor = np.sqrt(365*24*60)   # minute bars per year
    metrics = {
        "Sharpe_like": float(r.mean() / (r.std() + 1e-12) * ann_factor),
        "MaxDD": float((out["equity"] / out["equity"].cummax() - 1).min()),
        "Entries": int(((pos.shift(1).fillna(0) == 0) & (pos != 0)).sum()),
        "TotalBars": int(len(out)),
        "FinalEquity": float(out["equity"].iloc[-1]) if len(out) else 1.0,
    }
    return out, metrics

# ---- progress-enabled walk-forward (fold ETA in UI) ----
def _time_splits(idx: pd.DatetimeIndex, min_train_days: int, test_window_days: int, embargo_minutes: int):
    start = idx.min().normalize()
    end = idx.max().normalize()
    current_train_end = start + pd.Timedelta(days=min_train_days)
    folds = []
    while current_train_end + pd.Timedelta(days=test_window_days) <= end + pd.Timedelta(days=1):
        train_end = current_train_end
        test_start = train_end + pd.Timedelta(minutes=embargo_minutes)
        test_end = test_start + pd.Timedelta(days=test_window_days)
        tr = (idx < train_end)
        te = (idx >= test_start) & (idx < test_end)
        if te.sum() >= 50:
            folds.append((tr, te))
        current_train_end += pd.Timedelta(days=test_window_days)
    return folds

def _fmt_seconds(s: float) -> str:
    s = int(max(0, s))
    h, r = divmod(s, 3600)
    m, r = divmod(r, 60)
    return f"{h:d}:{m:02d}:{r:02d}" if h else f"{m:d}:{r:02d}"

def run_walkforward_streamlit(df: pd.DataFrame,
                              feat_cols: list,
                              target_col: str = "y",
                              min_train_days: int = 3,
                              test_window_days: int = 1,
                              embargo_minutes: int = 10,
                              n_estimators: int = 400,
                              learning_rate: float = 0.03,
                              status_placeholder=None,
                              progress_bar=None):
    X_all = df[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y_all = df.loc[X_all.index, target_col]
    idx = X_all.index

    folds = _time_splits(idx, min_train_days, test_window_days, embargo_minutes)
    total = len(folds)
    if progress_bar is None: progress_bar = st.progress(0.0)
    if status_placeholder is None: status_placeholder = st.empty()

    oof_pred = pd.Series(index=idx, dtype=float)
    metrics, models, times = [], [], []
    t0 = time.time()

    for i, (tr_mask, te_mask) in enumerate(folds, 1):
        t_fold = time.time()
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        model = LGBMClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        oof_pred.loc[X_te.index] = p

        y_hat = (p > 0.5).astype(int)
        metrics.append({
            "fold": i,
            "AUC": roc_auc_score(y_te, p),
            "Acc": accuracy_score(y_te, y_hat),
            "F1": f1_score(y_te, y_hat),
            "Precision": precision_score(y_te, y_hat),
            "Recall": recall_score(y_te, y_hat),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })
        models.append(model)

        # update progress + ETA
        elapsed = time.time() - t_fold
        times.append(elapsed)
        avg = sum(times)/len(times)
        remaining = avg * (total - i)
        progress_bar.progress(i/total)
        status_placeholder.info(
            f"Training fold {i}/{total} — elapsed: {_fmt_seconds(time.time()-t0)}  |  ETA: {_fmt_seconds(remaining)}"
        )

    status_placeholder.success(f"Training complete in {_fmt_seconds(time.time()-t0)} (folds: {total})")
    return {"oof_pred": oof_pred, "metrics": pd.DataFrame(metrics), "models": models, "features": feat_cols}



# --- ensure a usable Jupyter kernel for Papermill ---
def get_or_create_kernel():
    """
    Return an existing kernel name (prefer 'python3'). If none are registered,
    install a user-scoped ipykernel named 'streamlit-py' and return that.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        ksm = KernelSpecManager()
        specs = ksm.find_kernel_specs()  # dict name -> path
        if "python3" in specs:
            return "python3"
    except Exception:
        pass

    # No kernels registered: create one
    try:
        from ipykernel.kernelspec import install as install_ipykernel
        install_ipykernel(user=True, name="streamlit-py", display_name="Python 3 (streamlit)")
        return "streamlit-py"
    except Exception as e:
        raise RuntimeError(f"Could not provision a Jupyter kernel: {e}")

# ---- run a .ipynb via Papermill and pull last plot ----
from pathlib import Path
import os, base64
APP_DIR = Path(__file__).resolve().parent

def run_ipynb_and_get_plot(nb_path: str, parameters: dict | None = None, out_dir: str = "runs"):
    # resolve notebook path
    nb_file = Path(nb_path)
    if not nb_file.is_absolute():
        nb_file = (APP_DIR / nb_file).resolve()
    if not nb_file.exists():
        notebooks_dir = (APP_DIR / "notebooks")
        available = [p.name for p in notebooks_dir.glob("*.ipynb")] if notebooks_dir.exists() else []
        raise FileNotFoundError(f"Notebook not found: {nb_file}\nAvailable under {notebooks_dir}: {available}")

    # ensure PYTHONPATH has repo root so `import src...` works inside the notebook kernel
    os.environ["PYTHONPATH"] = str(APP_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

    out_dir = (APP_DIR / out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    out_nb = out_dir / f"{nb_file.stem}__{ts}.ipynb"

    # ensure a kernel exists (from earlier helper)
    kernel = get_or_create_kernel()

    # run with repo root as working directory
    try:
        pm.execute_notebook(
            input_path=str(nb_file),
            output_path=str(out_nb),
            parameters=parameters or {},
            kernel_name=kernel,
            cwd=str(APP_DIR),          # <- key: run from repo root
            log_output=True,
        )
    except TypeError:
        # older papermill without cwd= support: temporary chdir
        cur = os.getcwd()
        os.chdir(APP_DIR)
        try:
            pm.execute_notebook(
                input_path=str(nb_file),
                output_path=str(out_nb),
                parameters=parameters or {},
                kernel_name=kernel,
                log_output=True,
            )
        finally:
            os.chdir(cur)

    # pull last inline image or fallback file
    nb = nbformat.read(out_nb, as_version=4)
    for cell in reversed(nb.cells):
        for out in reversed(cell.get("outputs", [])):
            data = out.get("data", {})
            if "image/png" in data:
                img_bytes = base64.b64decode(data["image/png"])
                return str(out_nb), img_bytes
    fallback = APP_DIR / "artifacts" / "equity.png"
    if fallback.exists():
        return str(out_nb), fallback.read_bytes()
    return str(out_nb), None


# ---- one-symbol pipeline ----
def pipeline_for_symbol(sym: str,
                        interval: str,
                        start_ts: pd.Timestamp,
                        end_ts: pd.Timestamp,
                        horizon: int,
                        cost_bps: int,
                        slip_bps: int,
                        long_th: float,
                        short_th: float,
                        min_train_days: int,
                        test_window_days: int,
                        embargo_min: int,
                        n_estimators: int,
                        learning_rate: float,
                        use_progress: bool):
    """Fetch, feature, train (with optional fold progress), then backtest one symbol.
       Returns (bt, metr, host, folds, res, feat_cols)."""

    # ----- data -----
    try:
        df, host = get_bars(sym, interval, start_ts, end_ts)
    except Exception:
        # synthetic fallback so the app still runs
        host = "synthetic"
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

    # ----- features + labels -----
    fdf = build_features(
        df,
        rsi_periods=(7,14), macd=(12,26,9), atr_period=14,
        vol_windows=(5,15,60), vwap_windows=(5,15,60),
    )
    fdf = make_labels(fdf, horizon_minutes=int(horizon), cost_frac=bps_to_frac(cost_bps))
    fdf = fdf.dropna()

    excluded = {"y","fwd_ret"}
    base_exclude = {"open","high","low","close"}  # keep raw OHLC out of model inputs
    feat_cols = [c for c in fdf.columns if c not in excluded and c not in base_exclude]

    # ----- training (with optional progress/ETA) -----
    if use_progress:
        status = st.empty()
        pbar = st.progress(0.0)
        res = run_walkforward_streamlit(
            fdf, feat_cols, target_col="y",
            min_train_days=int(min_train_days),
            test_window_days=int(test_window_days),
            embargo_minutes=int(embargo_min),
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            status_placeholder=status,
            progress_bar=pbar,
        )
    else:
        wf_cfg = WalkForwardConfig(
            test_window_days=int(test_window_days),
            min_train_days=int(min_train_days),
            embargo_minutes=int(embargo_min),
            features=feat_cols,
            target_col="y",
        )
        res = run_walkforward(fdf, wf_cfg)

    folds = len(res.get("models", []))
    if folds == 0:
        return None, None, host, 0, res, feat_cols

    # ----- predictions + backtest -----
    fdf = fdf.join(res["oof_pred"].rename("proba"))
    bt, metr = backtest_fixed_horizon(
        fdf.dropna(subset=["proba","close"]),
        proba_col="proba",
        long_threshold=float(long_th),
        short_threshold=float(short_th),
        cost_bps=int(cost_bps),
        slippage_bps=int(slip_bps),
        hold_minutes=int(horizon),
    )
    return bt, metr, host, folds, res, feat_cols

# ======================
# UI
# ======================

with st.sidebar:
    st.header("Data")
    default_syms = [
        "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT",
        "DOGEUSDT","LTCUSDT","LINKUSDT","AVAXUSDT","MATICUSDT","ATOMUSDT"
    ]
    compare_mode = st.toggle("Compare multiple symbols", value=False)
    if compare_mode:
        symbols = st.multiselect("Symbols (Binance format)", options=default_syms, default=["BTCUSDT","ETHUSDT"])
        custom = st.text_input("Add custom (comma-separated)", "")
        if custom:
            symbols += [s.strip().upper() for s in custom.split(",") if s.strip()]
    else:
        symbol = st.selectbox("Symbol", options=default_syms, index=0)

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
    short_th = short_th if allow_shorts else 0.0

    st.header("Walk-Forward")
    min_train_days = st.number_input("Min train days", 1, 30, 3)
    test_window_days = st.number_input("Test window days", 1, 7, 1)
    embargo_min = st.number_input("Embargo minutes", 0, 120, 10)

    st.header("Training Speed")
    n_estimators = st.number_input("Trees (n_estimators)", 50, 1500, 400, 50)
    learning_rate = st.slider("Learning rate", 0.005, 0.2, 0.03, 0.005)

    st.header("Notebook mode")
    nb_mode = st.toggle("Run .ipynb instead of in-app pipeline", value=False)
    nb_path = st.text_input("Notebook path", "notebooks/01_intraday_lightgbm.ipynb", disabled=not nb_mode)

    run = st.button("Run backtest", use_container_width=True)

st.title("Intraday ML Backtester (1-min)")

# ======================
# Main pipeline
# ======================

if run:
    # ---- Notebook mode: execute .ipynb and show last plot ----
    if nb_mode:
        st.subheader("Notebook mode")
        st.write(f"Executing: `{nb_path}`")

        nb_params = {
            "SYMBOL": (symbol if not compare_mode else (symbols[0] if symbols else "BTCUSDT")),
            "HORIZON_MIN": int(horizon),
            "COST_BPS": int(cost_bps),
            "SLIP_BPS": int(slip_bps),
            # Add more params here if your notebook defines them
        }

        with st.status("Running notebook…", expanded=True) as status:
            try:
                executed_path, img_bytes = run_ipynb_and_get_plot(nb_path, parameters=nb_params)
                status.update(label=f"Finished: {executed_path}", state="complete")
            except Exception as e:
                status.update(label="Notebook failed", state="error")
                st.error(f"{e}")
                st.stop()

        if img_bytes:
            st.image(img_bytes, caption="Final plot from notebook")
        else:
            st.warning(
                "Executed notebook, but no plot image was found. "
                "Display a plot in the last cell or save one to artifacts/equity.png."
            )
        st.stop()

    # ---- In-app pipeline (compare or single) ----
    if compare_mode:
        st.subheader("Comparing symbols")
        if not symbols:
            st.warning("Select at least one symbol.")
        else:
            rows = []
            fig, ax = plt.subplots(figsize=(10, 4))
            overall = st.progress(0.0)

            for i, sym in enumerate(symbols, start=1):
                st.markdown(f"### {sym}")
                with st.container():
                    try:
                        bt, metr, host, folds, res, feat_cols = pipeline_for_symbol(
                            sym, interval, start_ts, end_ts, horizon,
                            cost_bps, slip_bps, long_th, short_th,
                            min_train_days, test_window_days, embargo_min,
                            n_estimators, learning_rate,
                            use_progress=True  # per-symbol fold ETA/progress
                        )
                        if bt is None:
                            st.warning(f"{sym}: no folds with current dates/settings — skipped.")
                        else:
                            st.caption(f"Source: {host} | Folds: {folds}")
                            ax.plot(bt["equity"], label=sym)
                            rows.append({"Symbol": sym, **metr, "Folds": folds, "Source": host})
                    except Exception as e:
                        st.error(f"{sym}: {e}")

                overall.progress(i / len(symbols))

            if rows:
                ax.set_title("Equity curves (fixed-horizon, no overlap)")
                ax.set_xlabel("Time"); ax.set_ylabel("Equity")
                ax.legend(loc="best")
                st.pyplot(fig, clear_figure=True)

                st.subheader("Summary metrics")
                st.dataframe(pd.DataFrame(rows).set_index("Symbol"), use_container_width=True)

    else:
        st.subheader(f"Symbol: {symbol}")
        try:
            bt, metr, host, folds, res, feat_cols = pipeline_for_symbol(
                symbol, interval, start_ts, end_ts, horizon,
                cost_bps, slip_bps, long_th, short_th,
                min_train_days, test_window_days, embargo_min,
                n_estimators, learning_rate,
                use_progress=True
            )
            st.caption(f"Source: {host} | Folds: {folds}")

            if bt is None:
                st.info("No folds created. Reduce 'Min train days' or widen the date range.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Final Equity", f"{metr['FinalEquity']:.3f}")
                c2.metric("Sharpe_like", f"{metr['Sharpe_like']:.2f}")
                c3.metric("Max Drawdown", f"{metr['MaxDD']:.2%}")
                c4.metric("Entries", f"{metr['Entries']}")

                fig, ax = plt.subplots(figsize=(10, 4))
                bt["equity"].plot(ax=ax)
                ax.set_title("Equity curve (fixed-horizon, no overlap)")
                ax.set_xlabel("Time"); ax.set_ylabel("Equity")
                st.pyplot(fig, clear_figure=True)

                st.subheader("Walk-forward fold metrics")
                st.dataframe(res["metrics"], use_container_width=True)

                if len(res["models"]):
                    fi = pd.DataFrame({
                        "feature": feat_cols,
                        "importance": res["models"][-1].feature_importances_
                    }).sort_values("importance", ascending=False).head(25)
                    st.subheader("Top features (last fold)")
                    st.dataframe(fi, use_container_width=True)

                st.caption("Note: Backtest results are for research only and not predictive of future returns.")
        except Exception as e:
            st.error(f"Run failed: {e}")

else:
    st.info("Set your parameters in the sidebar and click **Run backtest**.")
