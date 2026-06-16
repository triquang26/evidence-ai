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

    keep_keywords: tuple = (
        "anomaly", "outlier", "forecast", "predict",
        "time series", "time-series", "temporal",
        "detection", "representation", "imputation",
        "spatiotemporal", "change point", "changepoint",
        "sensor", "monitoring",
    )

    arxiv_cats: frozenset = frozenset({"cs.LG", "stat.ML", "cs.AI", "eess.SP", "cs.DB"})

    awesome_lists: tuple = (
        ("qingsongedu/awesome-AI-for-time-series-papers", "main"),
        ("qingsongedu/time-series-transformers-review", "main"),
        ("qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM", "main"),
        ("qingsongedu/Awesome-SSL4TS", "main"),
        ("TongjiFinLab/awesome-time-series-forecasting", "main"),
        ("lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms", "main"),
        ("mala-lab/Awesome-Anomaly-Detection-Foundation-Models", "main"),
        ("zhuyiche/awesome-anomaly-detection", "master"),
        ("hoya012/awesome-anomaly-detection", "master"),
        ("rob-med/awesome-AD", "master"),
        ("dsaidgovsg/awesome-anomaly-detection", "master"),
        ("caiyiqing/awesome-time-series-papers", "main"),
        ("GZHermit/awesome-time-series", "main"),
    )

    arxiv_queries: tuple = (
        # Anomaly detection
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
        # Forecasting
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
        # Foundation / LLM
        "time series foundation model",
        "large language model time series",
        "time series pretrained model zero-shot",
        "generative model time series synthesis",
        "time series GPT language model adaptation",
        # Representation / SSL
        "time series representation learning",
        "time series self-supervised learning",
        "time series contrastive learning",
        "time series masked autoencoder pretraining",
        # Methods / domains
        "temporal graph neural network forecasting",
        "time series diffusion generative model",
        "temporal convolutional network sequence modeling",
        "spatiotemporal prediction deep learning",
        "time series imputation missing data",
        "time series classification convolutional",
        "anomaly detection root cause analysis microservice",
        "traffic flow prediction graph network",
        "time series augmentation data generation",
    )
