"""
feature_engineering.py
-----------------------
Module 3: Transform daily sentiment scores into prediction-ready features.

Why this matters:
    A raw sentiment score on day T is weakly predictive on its own.
    What's more informative is *change* — is sentiment improving or
    deteriorating? Is there unusual disagreement between headlines?
    Is coverage volume spiking (which often precedes volatility)?

    This module builds those signals systematically.

Features produced (per trading day):
    ── Sentiment level ──────────────────────────────────────────────
    mean_score          raw daily average (from Module 2)
    net_sentiment       pct_positive - pct_negative

    ── Momentum (is sentiment trending up or down?) ─────────────────
    score_3d_avg        3-day rolling mean of mean_score
    score_7d_avg        7-day rolling mean of mean_score
    score_momentum      mean_score minus 7-day average (deviation)
    score_change_1d     day-over-day change in mean_score

    ── Volatility (are headlines agreeing or disagreeing?) ──────────
    score_std           intraday std across headlines (from Module 2)
    score_3d_std        3-day rolling std of daily mean_score
    sentiment_range     max_score - min_score on that day

    ── Coverage (how much news is there?) ───────────────────────────
    n_articles          raw article count
    article_volume_3d   3-day rolling mean of article count
    volume_spike        n_articles / article_volume_3d  (>1 = spike)

    ── Target variable ───────────────────────────────────────────────
    next_day_return     next trading day % return  (regression target)
    next_day_direction  +1 / 0 / -1                (classification target)

Usage:
    from src.feature_engineering import FeatureEngineer

    fe       = FeatureEngineer()
    features = fe.build(daily_sentiment_df, stock_df)
    X, y     = fe.split_features_target(features)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum rows needed before rolling features become meaningful.
# Rows before this threshold have NaN rolling values and are dropped.
MIN_ROWS_REQUIRED = 7


class FeatureEngineer:
    """
    Joins sentiment and stock data, then engineers predictive features.

    Args:
        roll_short (int): Short rolling window in days (default: 3)
        roll_long  (int): Long rolling window in days  (default: 7)

    Example:
        fe       = FeatureEngineer()
        features = fe.build(daily_df, stock_df)
        X, y     = fe.split_features_target(features)
    """

    def __init__(self, roll_short: int = 3, roll_long: int = 7):
        self.roll_short = roll_short
        self.roll_long  = roll_long

    # ── Main entry point ───────────────────────────────────────────────────────

    def build(
        self,
        daily_sentiment: pd.DataFrame,
        stock_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join sentiment + stock data and compute all features.

        Args:
            daily_sentiment : Output of SentimentPipeline.aggregate_daily()
                              Index = date, columns include mean_score etc.
            stock_df        : Output of StockIngester.fetch()
                              Must have: date, close, daily_return,
                              next_day_return, next_day_direction

        Returns:
            DataFrame with one row per trading day, all features + targets.
            Rows with NaN (early window, missing data) are dropped.
        """
        logger.info("Building features...")

        # ── Step 1: Join sentiment to stock prices ─────────────────────────────
        merged = self._join(daily_sentiment, stock_df)
        if merged.empty:
            raise ValueError(
                "Join produced an empty DataFrame. "
                "Check that sentiment and stock dates overlap."
            )

        # ── Step 2: Engineer features ──────────────────────────────────────────
        df = merged.copy()
        df = self._add_sentiment_momentum(df)
        df = self._add_sentiment_volatility(df)
        df = self._add_coverage_features(df)
        df = self._add_price_features(df)

        # ── Step 3: Drop rows with NaN (rolling windows need history) ──────────
        before = len(df)
        df.dropna(inplace=True)
        after  = len(df)
        dropped = before - after

        if dropped > 0:
            logger.info(
                f"Dropped {dropped} rows with NaN "
                f"(expected — rolling windows need {self.roll_long} days of history)."
            )

        if len(df) < MIN_ROWS_REQUIRED:
            logger.warning(
                f"Only {len(df)} rows after cleaning. "
                f"Need more historical data for reliable features. "
                f"Consider fetching more days in data_ingestion."
            )

        logger.info(
            f"Feature matrix ready: {len(df)} rows × {len(self._feature_cols(df))} features."
        )
        self._log_feature_summary(df)
        return df

    # ── Join ───────────────────────────────────────────────────────────────────

    def _join(
        self,
        daily_sentiment: pd.DataFrame,
        stock_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join sentiment onto stock prices by date.

        We use stock dates as the spine — every trading day is kept,
        even if there's no news that day (sentiment columns → NaN,
        then forward-filled with the previous day's sentiment).

        Forward-filling is appropriate here: if there's no news on
        Monday, the most recent sentiment (Friday) is still the best
        available signal going into Tuesday's open.
        """
        # Normalise date columns
        stock = stock_df.copy()
        stock["date"] = pd.to_datetime(stock["date"])
        stock = stock.set_index("date").sort_index()

        sentiment = daily_sentiment.copy()
        sentiment.index = pd.to_datetime(sentiment.index)
        sentiment = sentiment.sort_index()

        # Keep only the sentiment columns we need
        sentiment_cols = [
            c for c in sentiment.columns
            if c in {
                "mean_score", "median_score", "max_score", "min_score",
                "std_score", "n_articles", "pct_positive", "pct_negative",
                "pct_neutral", "net_sentiment",
            }
        ]
        sentiment = sentiment[sentiment_cols]

        merged = stock.join(sentiment, how="left")

        # Forward-fill up to 3 days (don't propagate stale sentiment too far)
        merged[sentiment_cols] = merged[sentiment_cols].ffill(limit=3)

        # If still NaN (no prior sentiment at all), fill with neutral 0
        merged[sentiment_cols] = merged[sentiment_cols].fillna(0)

        logger.info(
            f"Joined {len(merged)} trading days "
            f"({merged.index.min().date()} → {merged.index.max().date()})."
        )
        return merged

    # ── Feature groups ─────────────────────────────────────────────────────────

    def _add_sentiment_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum features: is sentiment trending up or down?

        score_momentum > 0 means today's sentiment is more positive
        than the recent average — a bullish signal.
        score_momentum < 0 means deteriorating sentiment.
        """
        s = df["mean_score"]

        df["score_3d_avg"]    = s.rolling(self.roll_short).mean()
        df["score_7d_avg"]    = s.rolling(self.roll_long).mean()
        df["score_momentum"]  = s - df["score_7d_avg"]   # deviation from trend
        df["score_change_1d"] = s.diff(1)                # day-over-day delta

        return df

    def _add_sentiment_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility features: are headlines agreeing or disagreeing?

        High std_score on a single day = conflicting signals (some headlines
        very positive, others very negative). This often precedes large moves.

        score_3d_std captures multi-day uncertainty.
        sentiment_range is the simplest volatility proxy: max - min.
        """
        df["score_3d_std"]     = df["mean_score"].rolling(self.roll_short).std()
        df["sentiment_range"]  = df["max_score"] - df["min_score"]

        return df

    def _add_coverage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coverage features: how much news is there?

        Volume spikes are informative — a sudden surge in article count
        often signals a significant event (earnings, product launch,
        lawsuit) before the price has fully reacted.

        volume_spike > 1.5 means 50% more articles than the 3-day average.
        """
        df["article_volume_3d"] = df["n_articles"].rolling(self.roll_short).mean()

        # Avoid division by zero on days with no prior coverage
        df["volume_spike"] = df["n_articles"] / df["article_volume_3d"].replace(0, 1)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-based context features.

        We're careful NOT to include any future price information here.
        These are all backward-looking, available at market open.

        price_momentum_3d: recent price trend — gives the model context
        about whether sentiment is confirming or contradicting the price trend.

        volatility_3d: recent price volatility — high vol days may make
        sentiment signals less reliable.
        """
        df["price_momentum_3d"] = df["close"].pct_change(self.roll_short) * 100
        df["volatility_3d"]     = df["daily_return"].rolling(self.roll_short).std()

        return df

    # ── Feature / target split ─────────────────────────────────────────────────

    def split_features_target(
        self,
        df: pd.DataFrame,
        target: str = "next_day_direction",
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the feature matrix into X (features) and y (target).

        Args:
            df     : Output of build()
            target : "next_day_direction" (classification, default)
                     "next_day_return"    (regression)

        Returns:
            (X, y) — both aligned by index (date)
        """
        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' not found. "
                f"Available: {list(df.columns)}"
            )

        feature_cols = self._feature_cols(df)
        X = df[feature_cols]
        y = df[target]

        logger.info(
            f"X: {X.shape}, y: {y.shape} "
            f"(target='{target}', "
            f"classes={sorted(y.unique().tolist())})"
        )
        return X, y

    @staticmethod
    def _feature_cols(df: pd.DataFrame) -> list[str]:
        """Return the list of feature columns (everything except targets/metadata)."""
        exclude = {
            "next_day_return", "next_day_direction",
            "open", "high", "low", "volume",       # raw OHLCV — too noisy raw
            "direction",                             # current-day direction (leakage risk)
            "ticker",
        }
        return [c for c in df.columns if c not in exclude]

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @staticmethod
    def _log_feature_summary(df: pd.DataFrame):
        """Log a quick statistical summary for sanity checking."""
        key_cols = [
            c for c in [
                "mean_score", "score_momentum", "score_3d_std",
                "volume_spike", "net_sentiment",
            ]
            if c in df.columns
        ]
        if not key_cols:
            return
        logger.info("Feature summary (mean ± std):")
        for col in key_cols:
            logger.info(
                f"  {col:22s}: {df[col].mean():+.4f} ± {df[col].std():.4f}"
                f"  [{df[col].min():+.4f}, {df[col].max():+.4f}]"
            )

    def correlation_report(
        self,
        df: pd.DataFrame,
        target: str = "next_day_return",
    ) -> pd.DataFrame:
        """
        Compute correlation of each feature with the target.

        This is your first sanity check before training any model:
        do the features have ANY relationship with future returns?
        Even weak correlations (|r| > 0.05) are meaningful in finance.

        Args:
            df     : Output of build()
            target : Column to correlate against

        Returns:
            DataFrame sorted by absolute correlation, descending.
        """
        feature_cols = self._feature_cols(df)
        cols = feature_cols + [target]
        corr = df[cols].corr()[target].drop(target)

        report = pd.DataFrame({
            "feature":     corr.index,
            "correlation": corr.values,
            "abs_corr":    corr.abs().values,
        }).sort_values("abs_corr", ascending=False).reset_index(drop=True)

        logger.info(f"Top features by |correlation| with {target}:")
        for _, row in report.head(8).iterrows():
            if pd.isna(row.abs_corr):
                continue
            bar = "█" * int(row.abs_corr * 30)
            logger.info(
                f"  {row.feature:25s}: {row.correlation:+.4f}  {bar}"
            )

        return report


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (run: python -m src.feature_engineering)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.load_data import load
    from src.sentiment_model import SentimentPipeline

    print("── Loading data ──")
    news_df, stock_df = load("AAPL")

    print("── Scoring sentiment ──")
    pipe      = SentimentPipeline()
    scored    = pipe.score(news_df)
    daily     = pipe.aggregate_daily(scored)

    print("── Engineering features ──")
    fe       = FeatureEngineer()
    features = fe.build(daily, stock_df)

    print("\nFeature matrix shape:", features.shape)
    print("\nColumns:", list(features.columns))
    print("\nSample rows:")
    print(features.tail(5).to_string())

    print("\n── Correlation with next_day_return ──")
    report = fe.correlation_report(features, target="next_day_return")
    print(report.to_string(index=False))

    print("\n── Feature/target split ──")
    X, y = fe.split_features_target(features, target="next_day_direction")
    print(f"X shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts()}")
