import os
import argparse
from src.pipeline import Pipeline
from src.utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Entity Framing Hierarchical Model")
    parser.add_argument(
        "--task",
        choices=["train", "evaluate"],
        required=True,
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
        help="Directory to save the trained model or evaluation results",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for debugging purposes)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["EN", "PT"],
        help="Languages to process (e.g., EN PT)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation",
    )
    return parser.parse_args()


def validate_config(config_path):
    """Validate the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def main():
    args = parse_args()

    # Validate the configuration file
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    validate_config(config_path)

    # Initialize the pipeline
    pipeline = Pipeline(config_path)

    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        pipeline.load_checkpoint(args.checkpoint)

    if args.task == "train":
        logger.info("Starting training task...")
        pipeline.train(
            languages=args.languages,
            max_articles=args.max_articles
        )

    elif args.task == "evaluate":
        logger.info("Starting evaluation task...")
        pipeline.evaluate(
            languages=args.languages,
            output_dir=args.output_dir,
            split="dev",
            max_articles=args.max_articles
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
