"""
backtesting.py
--------------
Module 5: Simulate trading on model predictions and compute financial metrics.

Why build our own instead of using backtrader/zipline?
    Third-party backtesting libraries hide assumptions. Building our own
    keeps every assumption explicit and visible — important for a portfolio
    project where you need to explain every decision.

Strategy:
    Signal-based long/short:
        prediction == +1  → BUY  (go long, expect price to rise)
        prediction == -1  → SELL (go short, expect price to fall)
        prediction ==  0  → HOLD (no position, stays flat)

    Each day:
        1. Signal is generated from yesterday's close + news
        2. Trade executes at today's OPEN (realistic — you can't trade
           at yesterday's close after seeing yesterday's news)
        3. Position closes at today's CLOSE
        4. Transaction costs deducted per trade

Metrics computed:
    total_return        overall strategy return %
    annual_return       annualised return %
    sharpe_ratio        risk-adjusted return (annualised)
    max_drawdown        largest peak-to-trough loss %
    win_rate            % of trades that were profitable
    n_trades            total number of trades taken
    vs_buy_and_hold     outperformance vs simply holding the stock

Usage:
    from src.backtesting import Backtester

    bt      = Backtester()
    report  = bt.run(features_df, model)
    bt.plot(report)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Transaction cost per trade (buy + sell round-trip).
# 0.1% is conservative for retail — realistic for liquid large-caps.
DEFAULT_COST_BPS = 10   # basis points (10bps = 0.1%)

TRADING_DAYS_PER_YEAR = 252


class Backtester:
    """
    Simulates a trading strategy driven by model predictions.

    Args:
        cost_bps      : Round-trip transaction cost in basis points (default: 10)
        initial_capital: Starting portfolio value in $ (default: 10,000)

    Example:
        bt     = Backtester()
        report = bt.run(features_df, model)
        bt.plot(report)
    """

    def __init__(
        self,
        cost_bps: int = DEFAULT_COST_BPS,
        initial_capital: float = 10_000,
    ):
        self.cost_bps        = cost_bps
        self.cost_pct        = cost_bps / 10_000
        self.initial_capital = initial_capital

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        features_df: pd.DataFrame,
        model,
        test_start_frac: float = 0.7,
    ) -> dict:
        """
        Run backtest on the test portion of the feature matrix.

        We only backtest on the TEST set (last 30% by default) — the model
        has never seen this data during training, so performance is honest.
        Backtesting on training data is a common mistake that produces
        wildly inflated results.

        Args:
            features_df    : Full feature matrix (output of FeatureEngineer.build)
            model          : Trained PredictionModel instance
            test_start_frac: Where test period begins (default: 0.7 = last 30%)

        Returns:
            dict with keys: metrics, daily, signals
        """
        # ── Split into test portion ────────────────────────────────────────────
        n          = len(features_df)
        test_start = int(n * test_start_frac)
        test_df    = features_df.iloc[test_start:].copy()

        logger.info(
            f"Backtesting on {len(test_df)} days "
            f"({test_df.index[0].date()} → {test_df.index[-1].date()})"
        )

        # ── Generate signals ───────────────────────────────────────────────────
        from src.feature_engineering import FeatureEngineer
        X_test, _ = FeatureEngineer().split_features_target(
            test_df, target="next_day_direction"
        )

        # Only predict on rows where we have a non-flat target available
        # (last row has no next_day data — drop it)
        X_test = X_test.iloc[:-1]
        test_df = test_df.iloc[:-1]

        signals = model.predict(X_test)

        # ── Simulate daily P&L ─────────────────────────────────────────────────
        daily = self._simulate(test_df, signals)

        # ── Compute metrics ────────────────────────────────────────────────────
        metrics = self._compute_metrics(daily)

        # ── Buy-and-hold benchmark ─────────────────────────────────────────────
        bnh = self._buy_and_hold(test_df)
        metrics["buy_and_hold_return"] = bnh["total_return"]
        metrics["vs_buy_and_hold"]     = (
            metrics["total_return"] - metrics["buy_and_hold_return"]
        )

        self._log_report(metrics)

        return {
            "metrics": metrics,
            "daily":   daily,
            "bnh":     bnh,
            "signals": pd.Series(signals, index=test_df.index, name="signal"),
        }

    # ── Simulation ─────────────────────────────────────────────────────────────

    def _simulate(
        self,
        test_df: pd.DataFrame,
        signals: np.ndarray,
    ) -> pd.DataFrame:
        """
        Step through each day and compute P&L.

        Position sizing: 100% of capital per trade (fully invested).
        Shorts are allowed (signal = -1 earns when price falls).
        """
        rows = []
        prev_signal = 0

        for i, (date, row) in enumerate(test_df.iterrows()):
            signal      = int(signals[i])
            daily_ret   = row.get("next_day_return", 0) / 100  # % → decimal
            traded      = signal != 0
            trade_cost  = self.cost_pct if (signal != prev_signal and traded) else 0

            # Strategy return:
            #   long  (+1): earn the daily return
            #   short (-1): earn the negative of the daily return
            #   flat  ( 0): earn nothing
            strat_ret = signal * daily_ret - trade_cost

            rows.append({
                "date":       date,
                "signal":     signal,
                "daily_ret":  daily_ret,
                "strat_ret":  strat_ret,
                "trade_cost": trade_cost,
                "traded":     traded and (signal != prev_signal),
            })

            prev_signal = signal

        daily = pd.DataFrame(rows).set_index("date")

        # Cumulative portfolio value
        daily["cum_strat"]  = (1 + daily["strat_ret"]).cumprod()
        daily["portfolio"]  = self.initial_capital * daily["cum_strat"]

        return daily

    def _buy_and_hold(self, test_df: pd.DataFrame) -> dict:
        """Benchmark: buy on day 1, hold until the end."""
        rets    = test_df["next_day_return"].fillna(0) / 100
        cum     = (1 + rets).cumprod()
        total   = (cum.iloc[-1] - 1) * 100

        bnh_daily = pd.DataFrame({
            "cum_bnh":   cum,
            "portfolio": self.initial_capital * cum,
        }, index=test_df.index)

        return {
            "total_return": total,
            "daily":        bnh_daily,
        }

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _compute_metrics(self, daily: pd.DataFrame) -> dict:
        rets   = daily["strat_ret"]
        cum    = daily["cum_strat"]
        n_days = len(daily)

        total_return  = (cum.iloc[-1] - 1) * 100
        annual_return = ((1 + total_return / 100) ** (TRADING_DAYS_PER_YEAR / n_days) - 1) * 100

        # Sharpe ratio (annualised, assuming risk-free rate ≈ 0 for simplicity)
        daily_std = rets.std()
        sharpe    = (rets.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
                     if daily_std > 0 else 0.0)

        # Max drawdown
        rolling_max  = cum.cummax()
        drawdown     = (cum - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # Win rate
        trades    = daily[daily["traded"]]
        n_trades  = len(trades)
        win_rate  = (trades["strat_ret"] > 0).mean() * 100 if n_trades > 0 else 0

        return {
            "total_return":  round(total_return,  2),
            "annual_return": round(annual_return, 2),
            "sharpe_ratio":  round(sharpe,        3),
            "max_drawdown":  round(max_drawdown,  2),
            "win_rate":      round(win_rate,       1),
            "n_trades":      n_trades,
            "n_days":        n_days,
        }

    # ── Reporting ──────────────────────────────────────────────────────────────

    @staticmethod
    def _log_report(metrics: dict):
        logger.info("\n── Backtest Results ─────────────────────────────────")
        logger.info(f"  Period              : {metrics['n_days']} trading days")
        logger.info(f"  Total return        : {metrics['total_return']:+.2f}%")
        logger.info(f"  Annual return       : {metrics['annual_return']:+.2f}%")
        logger.info(f"  Sharpe ratio        : {metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max drawdown        : {metrics['max_drawdown']:.2f}%")
        logger.info(f"  Win rate            : {metrics['win_rate']:.1f}%")
        logger.info(f"  Trades taken        : {metrics['n_trades']}")
        logger.info(f"  Buy-and-hold return : {metrics['buy_and_hold_return']:+.2f}%")
        logger.info(f"  vs Buy-and-hold     : {metrics['vs_buy_and_hold']:+.2f}%")

    def plot(self, report: dict):
        """
        Plot strategy vs buy-and-hold equity curves.
        Requires plotly (already in requirements.txt).
        Works in Jupyter notebooks inline.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("plotly not installed — skipping plot.")
            return

        daily   = report["daily"]
        bnh     = report["bnh"]["daily"]
        metrics = report["metrics"]
        signals = report["signals"]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=[
                "Portfolio value — strategy vs buy-and-hold",
                "Daily strategy return",
                "Trading signals",
            ],
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.08,
        )

        # ── Equity curves ──────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily["portfolio"],
            name="Strategy",
            line=dict(color="#2563EB", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=bnh.index, y=bnh["portfolio"],
            name="Buy & Hold",
            line=dict(color="#9CA3AF", width=1.5, dash="dash"),
        ), row=1, col=1)

        # ── Daily returns ──────────────────────────────────────────────────────
        colors = ["#16A34A" if r >= 0 else "#DC2626" for r in daily["strat_ret"]]
        fig.add_trace(go.Bar(
            x=daily.index, y=daily["strat_ret"] * 100,
            name="Daily return %",
            marker_color=colors,
            showlegend=False,
        ), row=2, col=1)

        # ── Signals ───────────────────────────────────────────────────────────
        signal_colors = {1: "#16A34A", -1: "#DC2626", 0: "#9CA3AF"}
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals.values,
            mode="markers",
            name="Signal",
            marker=dict(
                color=[signal_colors.get(s, "#9CA3AF") for s in signals],
                size=6,
                symbol="circle",
            ),
            showlegend=False,
        ), row=3, col=1)

        # ── Annotation box ────────────────────────────────────────────────────
        annotation = (
            f"Total: {metrics['total_return']:+.1f}%  |  "
            f"Annual: {metrics['annual_return']:+.1f}%  |  "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
            f"Max DD: {metrics['max_drawdown']:.1f}%  |  "
            f"Win rate: {metrics['win_rate']:.0f}%  |  "
            f"vs B&H: {metrics['vs_buy_and_hold']:+.1f}%"
        )

        fig.update_layout(
            title=dict(text=annotation, font=dict(size=12)),
            height=700,
            legend=dict(orientation="h", y=1.02),
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_yaxes(gridcolor="#F3F4F6")
        fig.update_xaxes(gridcolor="#F3F4F6")

        fig.show()
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (run: python -m src.backtesting)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from src.rss_ingester import fetch_rss_and_stock
    from src.sentiment_model import SentimentPipeline
    from src.feature_engineering import FeatureEngineer
    from src.prediction import PredictionModel

    print("── Loading data ──")
    news_df, stock_df = fetch_rss_and_stock("AAPL")

    print("── Scoring sentiment ──")
    pipe    = SentimentPipeline()
    scored  = pipe.score(news_df)
    daily   = pipe.aggregate_daily(scored)

    print("── Engineering features ──")
    fe       = FeatureEngineer()
    features = fe.build(daily, stock_df)
    X, y     = fe.split_features_target(features, target="next_day_direction")

    print("── Training model ──")
    model   = PredictionModel()
    model.train_evaluate(X, y)

    print("── Running backtest ──")
    bt     = Backtester()
    report = bt.run(features, model)

    print("\n── Metrics ──")
    for k, v in report["metrics"].items():
        print(f"  {k:25s}: {v}")
