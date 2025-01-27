import logging
from typing import List, Set, Dict
from pathlib import Path
import torch
import random
import numpy as np
from transformers import RobertaTokenizer


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a logger instance with consistent formatting."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, level.upper()),
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Get the appropriate device (CPU/GPU) for torch operations."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_path_exists(path: Path) -> Path:
    """Ensure a directory path exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def tokenize_texts(texts, max_length=512):
    """
    Tokenize a list of texts using RobertaTokenizer.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    return tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
