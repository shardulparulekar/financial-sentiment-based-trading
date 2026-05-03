"""
sentiment_model.py
------------------
Module 2: Score financial news headlines using FinBERT.

FinBERT is a BERT model fine-tuned on financial text (earnings calls,
analyst reports, financial news). It understands domain-specific language
that generic models get wrong — e.g. "missed expectations" is negative,
"beat estimates" is positive, "restructuring" is usually negative.

The model lives on HuggingFace and is downloaded automatically (~440MB,
one-time only). You need a HuggingFace account but NO paid API — the
weights download for free.

Output per headline:
    sentiment_label : "positive" | "negative" | "neutral"
    sentiment_score : float in [-1.0, +1.0]
                      positive → +score, negative → -score, neutral → 0

Usage:
    from src.sentiment_model import SentimentPipeline

    pipe = SentimentPipeline()
    df   = pipe.score(news_df)        # adds sentiment columns to news_df
    daily = pipe.aggregate_daily(df)  # one sentiment score per trading day
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Primary model — ProsusAI/finbert (~440MB, ~1.5GB RAM when loaded)
FINBERT_MODEL_PRIMARY  = "ProsusAI/finbert"

# Fallback model — yiyanghkust/finbert-tone (~110MB, ~400MB RAM)
# Used automatically if the primary model fails to load (e.g. OOM on Streamlit Cloud)
# Same financial domain training, slightly lower accuracy but far lighter.
FINBERT_MODEL_FALLBACK = "yiyanghkust/finbert-tone"

# Active model — set at runtime, can be overridden via env var
import os as _os
FINBERT_MODEL = _os.getenv("FINBERT_MODEL", FINBERT_MODEL_PRIMARY)

# FinBERT was trained on sentences up to 512 tokens.
MAX_TOKEN_LENGTH = 512

# How many headlines to send to the model at once.
# Smaller batch on fallback model to reduce peak RAM usage.
DEFAULT_BATCH_SIZE = 32


class SentimentPipeline:
    """
    Wraps FinBERT in a clean interface for batch headline scoring.

    The model is loaded lazily on first use — importing this module
    is fast even if transformers is slow to import.

    Args:
        model_name  : HuggingFace model ID (default: ProsusAI/finbert)
        batch_size  : Headlines per forward pass (default: 32)
        device      : "cpu", "cuda", or "mps" (Apple Silicon).
                      Auto-detected if not specified.

    Example:
        pipe    = SentimentPipeline()
        scored  = pipe.score(news_df)
        daily   = pipe.aggregate_daily(scored)
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device     = device or self._detect_device()
        self._pipeline  = None  # loaded lazily

    # ── Device detection ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        """
        Pick the best available compute device.
            - CUDA  → NVIDIA GPU (fastest)
            - MPS   → Apple Silicon GPU (fast on M1/M2/M3 Macs)
            - CPU   → fallback (fine for this project, ~2-5s per batch)
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Device: CUDA (NVIDIA GPU)")
                return "cuda"
            if torch.backends.mps.is_available():
                logger.info("Device: MPS (Apple Silicon)")
                return "mps"
        except ImportError:
            pass
        logger.info("Device: CPU")
        return "cpu"

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self):
        """
        Load FinBERT from HuggingFace on first use, with automatic fallback.

        Tries primary model (ProsusAI/finbert, ~440MB) first.
        If it fails with OOM or any other error, automatically falls back to
        the lighter model (yiyanghkust/finbert-tone, ~110MB).

        This makes the app resilient on memory-constrained environments like
        Streamlit Community Cloud free tier (~1GB RAM limit).
        """
        if self._pipeline is not None:
            return

        from transformers import (
            pipeline,
            BertForSequenceClassification,
            BertTokenizer,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            logging as hf_logging,
        )

        hf_logging.set_verbosity_error()
        device_index = -1 if self.device == "cpu" else 0

        models_to_try = [self.model_name]
        # Always append fallback if not already the primary
        if self.model_name != FINBERT_MODEL_FALLBACK:
            models_to_try.append(FINBERT_MODEL_FALLBACK)

        last_error = None
        for attempt, model_id in enumerate(models_to_try):
            try:
                if attempt == 0:
                    logger.info(f"Loading primary model: {model_id} (~440MB) ...")
                else:
                    logger.warning(
                        f"Primary model failed ({last_error}). "
                        f"Falling back to lighter model: {model_id} (~110MB)..."
                    )
                    self.batch_size = 16   # reduce batch size on fallback

                # use_safetensors=True bypasses torch.load, avoiding CVE-2025-32434
                try:
                    model     = BertForSequenceClassification.from_pretrained(
                        model_id, use_safetensors=True,
                    )
                    tokenizer = BertTokenizer.from_pretrained(model_id)
                except Exception:
                    # Fallback model may not have safetensors — try without
                    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
                    tokenizer = AutoTokenizer.from_pretrained(model_id)

                self._pipeline = pipeline(
                    task       = "text-classification",
                    model      = model,
                    tokenizer  = tokenizer,
                    device     = device_index,
                    truncation = True,
                    max_length = MAX_TOKEN_LENGTH,
                    top_k      = None,
                )

                self.model_name = model_id   # record which model actually loaded
                logger.info(f"Model loaded: {model_id} ({'primary' if attempt == 0 else 'FALLBACK'})")
                return   # success — stop trying

            except Exception as e:
                last_error = str(e)
                logger.error(f"Failed to load {model_id}: {e}")
                self._pipeline = None
                continue

        # Both models failed
        raise RuntimeError(
            f"Could not load any sentiment model. "
            f"Last error: {last_error}. "
            f"Check available RAM and HuggingFace connectivity."
        )

    def model_info(self) -> dict:
        """Return info about the currently loaded model."""
        return {
            "model_name":  self.model_name,
            "is_fallback": self.model_name == FINBERT_MODEL_FALLBACK,
            "device":      self.device,
            "batch_size":  self.batch_size,
        }

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every headline in news_df with FinBERT sentiment.

        Adds these columns to the DataFrame:
            sentiment_label  : "positive" | "negative" | "neutral"
            sentiment_score  : float in [-1.0, +1.0]
            prob_positive    : raw model probability (0-1)
            prob_negative    : raw model probability (0-1)
            prob_neutral     : raw model probability (0-1)

        Args:
            news_df : Output of NewsIngester.fetch() or load_data.load()
                      Must have a 'title' column.

        Returns:
            news_df with sentiment columns added.
        """
        if news_df.empty:
            logger.warning("Empty DataFrame passed to score() — returning as-is.")
            return news_df

        if "title" not in news_df.columns:
            raise ValueError("news_df must have a 'title' column.")

        self._load()

        headlines = news_df["title"].fillna("").tolist()
        n         = len(headlines)
        logger.info(f"Scoring {n} headlines in batches of {self.batch_size} ...")

        results = []
        start   = time.time()

        for i in range(0, n, self.batch_size):
            batch  = headlines[i : i + self.batch_size]
            output = self._pipeline(batch)
            results.extend(output)

            # Progress logging every 5 batches
            if (i // self.batch_size) % 5 == 0:
                pct = min(100, int((i + len(batch)) / n * 100))
                logger.info(f"  {pct}% ({i + len(batch)}/{n} headlines)")

        elapsed = time.time() - start
        logger.info(f"Scored {n} headlines in {elapsed:.1f}s "
                    f"({n/elapsed:.0f} headlines/sec)")

        # ── Parse model output ─────────────────────────────────────────────────
        # The pipeline returns a list of lists, one per headline.
        # Each inner list has 3 dicts: [{"label": "positive", "score": 0.92}, ...]
        rows = []
        for probs in results:
            prob_map = {p["label"].lower(): p["score"] for p in probs}
            label    = max(prob_map, key=prob_map.get)

            # Convert to signed scalar:
            # positive confidence → positive score
            # negative confidence → negative score
            # neutral → small score near 0 weighted by direction
            score = self._to_signed_score(
                prob_map.get("positive", 0),
                prob_map.get("negative", 0),
                prob_map.get("neutral",  0),
            )

            rows.append({
                "sentiment_label": label,
                "sentiment_score": score,
                "prob_positive":   prob_map.get("positive", 0),
                "prob_negative":   prob_map.get("negative", 0),
                "prob_neutral":    prob_map.get("neutral",  0),
            })

        sentiment_df = pd.DataFrame(rows, index=news_df.index)
        result       = pd.concat([news_df, sentiment_df], axis=1)

        self._log_distribution(result)
        return result

    @staticmethod
    def _to_signed_score(p_pos: float, p_neg: float, p_neu: float) -> float:
        """
        Collapse the three-class output into a single [-1, +1] score.

        Formula: score = p_positive - p_negative
            - Pure positive (1, 0, 0) → +1.0
            - Pure negative (0, 1, 0) → -1.0
            - Pure neutral  (0, 0, 1) →  0.0
            - Mixed (0.6, 0.3, 0.1)  → +0.3

        This is more informative than just taking the argmax label,
        because it preserves confidence — a weakly positive headline
        gets a score near 0, not a full +1.
        """
        return round(float(p_pos - p_neg), 4)

    @staticmethod
    def _log_distribution(df: pd.DataFrame):
        """Log a quick breakdown of sentiment labels for sanity checking."""
        counts = df["sentiment_label"].value_counts()
        total  = len(df)
        logger.info("Sentiment distribution:")
        for label, count in counts.items():
            logger.info(f"  {label:10s}: {count:4d} ({count/total*100:.1f}%)")

    # ── Daily aggregation ──────────────────────────────────────────────────────

    def aggregate_daily(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-headline scores into one row per trading day.

        This is what gets joined to stock price data in Module 3.
        Multiple headlines on the same day are combined using several
        aggregation strategies — each captures something different:

            mean_score      : average sentiment (smoothed signal)
            median_score    : robust to outlier headlines
            max_score       : most bullish headline of the day
            min_score       : most bearish headline of the day
            std_score       : sentiment disagreement / volatility
            n_articles      : how much news coverage there was
            pct_positive    : share of positive headlines
            pct_negative    : share of negative headlines
            pct_neutral     : share of neutral headlines
            net_sentiment   : pct_positive - pct_negative (simple ratio)

        Args:
            scored_df : Output of score() — must have 'market_date' and
                        'sentiment_score' columns.

        Returns:
            DataFrame with one row per trading day, indexed by market_date.
        """
        date_col = "market_date" if "market_date" in scored_df.columns else "date"

        if date_col not in scored_df.columns:
            raise ValueError(
                "scored_df must have a 'market_date' column. "
                "Run align_news_to_market() from data_ingestion first."
            )

        logger.info(f"Aggregating sentiment by {date_col} ...")

        grp = scored_df.groupby(date_col)

        daily = pd.DataFrame({
            "mean_score":   grp["sentiment_score"].mean(),
            "median_score": grp["sentiment_score"].median(),
            "max_score":    grp["sentiment_score"].max(),
            "min_score":    grp["sentiment_score"].min(),
            "std_score":    grp["sentiment_score"].std().fillna(0),
            "n_articles":   grp["sentiment_score"].count(),
        })

        # Label-based ratios
        label_counts = scored_df.groupby([date_col, "sentiment_label"]).size().unstack(fill_value=0)
        for col in ["positive", "negative", "neutral"]:
            if col not in label_counts.columns:
                label_counts[col] = 0

        daily["pct_positive"] = label_counts["positive"] / daily["n_articles"]
        daily["pct_negative"] = label_counts["negative"] / daily["n_articles"]
        daily["pct_neutral"]  = label_counts["neutral"]  / daily["n_articles"]
        daily["net_sentiment"] = daily["pct_positive"] - daily["pct_negative"]

        daily.index.name = "date"
        daily = daily.sort_index()

        logger.info(
            f"Aggregated {len(scored_df)} headlines → "
            f"{len(daily)} trading days."
        )
        logger.info(
            f"Daily mean_score range: "
            f"[{daily['mean_score'].min():.3f}, {daily['mean_score'].max():.3f}]"
        )

        return daily


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (run: python -m src.sentiment_model)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test with a small set of hand-crafted headlines so you can
    # verify the model is working correctly before running on real data.
    TEST_HEADLINES = [
        "Apple beats earnings expectations, stock surges 5%",
        "Microsoft faces antitrust investigation by EU regulators",
        "Google announces new AI model at annual developer conference",
        "Tesla misses delivery targets, shares fall sharply",
        "Fed signals potential rate cut as inflation cools",
        "Amazon Web Services reports record quarterly revenue",
        "Tech stocks tumble amid broader market selloff",
        "NVIDIA forecasts strong demand driven by AI chip orders",
    ]

    print("── Testing FinBERT sentiment pipeline ──\n")

    pipe = SentimentPipeline()

    # Wrap in a minimal DataFrame matching what Module 1 produces
    test_df = pd.DataFrame({
        "title":       TEST_HEADLINES,
        "market_date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
    })

    scored = pipe.score(test_df)

    print("\n── Per-headline scores ──")
    for _, row in scored.iterrows():
        bar = "+" * int(max(0,  row.sentiment_score) * 20)
        bar += "-" * int(max(0, -row.sentiment_score) * 20)
        print(f"  {row.sentiment_score:+.3f} [{bar:<20}] {row.title[:60]}")

    print("\n── Daily aggregates ──")
    daily = pipe.aggregate_daily(scored)
    print(daily[["mean_score", "net_sentiment", "n_articles"]].to_string())
