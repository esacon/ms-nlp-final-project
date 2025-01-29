import os
import argparse
import logging
from src.pipeline import Pipeline
from src.utils import get_logger
from src.data_loader import DataLoader
import yaml
import torch

logger = get_logger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Entity Framing Hierarchical Model")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "evaluate"],
        help="Task to perform: train or evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for debugging purposes)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="EN PT",
        help="Language to use for the entity role classifier",
    )
    return parser.parse_args()

def validate_config(config_path):
    """Validate the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def main():
    args = parse_args()

    # Validate the configuration file
    validate_config(args.config)

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["languages"]["default"] = args.language

    # Initialize the pipeline
    pipeline = Pipeline(config)

    if args.task == "train":
        logger.info("Starting training task...")

        # Load training and validation data
        data_loader = DataLoader(config, max_articles=args.max_articles)
        train_data = data_loader.load_data(split="train")
        logger.info(f"Loaded {len(train_data)} training articles")
        val_data = data_loader.load_data(split="dev")
        logger.info(f"Loaded {len(val_data)} validation articles")

        # Train the model
        pipeline.train(train_data, val_data)

        # Save the model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "hierarchical_model.pt")
        torch.save(pipeline.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    elif args.task == "evaluate":
        logger.info("Starting evaluation task...")

        # Load evaluation data
        data_loader = DataLoader(config, max_articles=args.max_articles)
        eval_data = data_loader.load_data(split="test")

        # Evaluate the model
        metrics = pipeline.evaluate(eval_data)
        logger.info(f"Evaluation results: {metrics}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
