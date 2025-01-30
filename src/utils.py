import os
import logging
import yaml
from typing import Dict
from pathlib import Path
import torch
import random
import numpy as np
from transformers import RobertaTokenizer


def load_config(config_file: str = "configs.yaml") -> Dict:
    """Load configuration from yaml file."""
    project_root = str(Path(__file__).parent.parent)
    config_path = os.path.join(project_root, "src", config_file)
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")


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
