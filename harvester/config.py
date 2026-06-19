from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    out_dir: Path = Path("outputs")
    pdf_dir: Path = Path("outputs/pdfs")
    arxiv_max_per_query: int = 300
    arxiv_sleep: float = 3.0
    title_min_len: int = 12

    # Applied by Deduplicator to ALL sources (OR logic — broad safety net)
    keep_keywords: tuple = (
        "time series", "time-series", "timeseries",
        "temporal sequence", "multivariate time",
        "anomaly", "outlier", "change point", "changepoint",
        "forecasting", "time series forecast",
        "imputation", "spatiotemporal",
    )

    # Applied by ArxivHFScraper with AND logic: paper must match ≥1 ts_keyword AND ≥1 task_keyword
    ts_keywords: tuple = (
        "time series", "time-series", "timeseries",
        "temporal sequence", "multivariate time", "univariate time",
        "time-stamped", "time stamp", "streaming data", "data stream",
        "sequential time",
    )
    task_keywords: tuple = (
        "anomaly", "outlier", "change point", "changepoint",
        "forecast", "imputation",
    )

    arxiv_cats: frozenset = frozenset({"cs.LG", "stat.ML", "cs.AI", "eess.SP", "cs.DB"})

    awesome_lists: tuple = (
        # Core time-series AI (anomaly + forecasting + representation)
        ("qingsongedu/awesome-AI-for-time-series-papers", "main"),
        ("qingsongedu/time-series-transformers-review", "main"),
        ("qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM", "main"),
        ("qingsongedu/Awesome-SSL4TS", "main"),
        # Forecasting-focused
        ("TongjiFinLab/awesome-time-series-forecasting", "main"),
        # Anomaly detection TS-specific
        ("lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms", "main"),
        # Removed: 404 dead links (rob-med, dsaidgovsg, caiyiqing, GZHermit)
        # Removed: image/video AD mixed lists (hoya012, zhuyiche)
        # Removed: broad foundation-model AD with heavy vision scope (mala-lab)
    )

    arxiv_queries: tuple = (
        # ── Anomaly detection (time-series) ─────────────────────────────────
        "time series anomaly detection",
        "multivariate time series anomaly detection",
        "unsupervised time series anomaly detection",
        "time series anomaly detection transformer",
        "time series anomaly detection autoencoder",
        "time series outlier detection machine learning",
        "time series change point detection",
        "anomaly detection sensor IoT time series",
        "time series anomaly detection benchmark",
        "time series anomaly detection graph neural network",
        "log anomaly detection time series",
        "time series anomaly detection reconstruction",
        "anomaly detection multivariate forecasting",
        # ── Forecasting ──────────────────────────────────────────────────────
        "time series forecasting deep learning",
        "multivariate time series forecasting neural network",
        "long-term time series forecasting",
        "probabilistic time series forecasting",
        "transformer time series forecasting",
        "time series forecasting attention mechanism",
        "time series forecasting survey benchmark",
        "time series forecasting state space model",
        "time series forecasting Informer PatchTST iTransformer",
        "time series forecasting linear model",
        "time series forecasting hybrid model",
        # ── Foundation / LLM for TS ──────────────────────────────────────────
        "time series foundation model",
        "large language model time series",
        "time series pretrained model zero-shot",
        # Removed: "generative model time series synthesis" (data generation, off-scope)
        # Removed: "time series GPT language model adaptation" (redundant)
        # ── Representation / SSL (methods used in AD & forecasting) ──────────
        "time series representation learning",
        "time series self-supervised learning",
        "time series contrastive learning",
        "time series masked autoencoder pretraining",
        # ── Supporting methods ───────────────────────────────────────────────
        "temporal graph neural network forecasting",
        "time series diffusion generative model",
        "temporal convolutional network sequence modeling",
        "spatiotemporal prediction deep learning",
        "time series imputation missing data",
        "traffic flow prediction graph network",
        # Removed: "time series classification convolutional" (classification ≠ AD/forecast)
        # Removed: "anomaly detection root cause analysis microservice" (AIOps, not TS ML)
        # Removed: "time series augmentation data generation" (augmentation, off-scope)
    )
