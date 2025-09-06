
import numpy as np
import pandas as pd
from .utils import bps_to_frac

def backtest_long_short(df: pd.DataFrame,
                        proba_col: str,
                        long_threshold: float = 0.55,
                        short_threshold: float = 0.45,
                        cost_bps: int = 3,
                        slippage_bps: int = 2,
                        horizon_minutes: int = 5):
    out = df.copy()
    cost = bps_to_frac(cost_bps)
    slip = bps_to_frac(slippage_bps)

    pos = pd.Series(0, index=out.index, dtype=int)
    pos[out[proba_col] >= long_threshold] = 1
    if short_threshold > 0:
        pos[out[proba_col] <= short_threshold] = -1
    pos = pos.ffill().fillna(0).astype(int)

    fwd_ret = out["fwd_ret"]

    trade = pos.diff().abs().fillna(0)
    trade_cost = trade * (cost + slip)

    pnl = pos * fwd_ret - trade_cost
    out["pos"] = pos
    out["pnl"] = pnl
    out["equity"] = (1 + out["pnl"]).cumprod()

    ret = out["pnl"].dropna()
    ann_factor = np.sqrt(252*6.5*60)
    sharpe = ret.mean() / (ret.std() + 1e-12) * ann_factor
    dd = (out["equity"] / out["equity"].cummax() - 1).min()
    turnover = trade.mean()

    metrics = {
        "Sharpe_like": float(sharpe),
        "MaxDD": float(dd),
        "Turnover_per_bar": float(turnover),
        "TotalBars": int(len(out)),
        "FinalEquity": float(out["equity"].iloc[-1]) if len(out) else 1.0
    }
    return out, metrics
