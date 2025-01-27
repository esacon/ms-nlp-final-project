import argparse
import logging
from datetime import datetime
from src.pipeline import Pipeline, run_training, run_testing
from src.utils import get_logger
from src.evaluate import evaluate_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entity Framing in English and Portuguese"
    )
    parser.add_argument(
        "--task",
        choices=["train", "evaluate", "test"],
        required=True,
        help="Task to perform (train, evaluate, or test)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-f",
        type=str,
        default=f"models/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
        help="Path to saved model (required for evaluation)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["EN", "PT"],
        default=["EN", "PT"],
        help="Languages to process",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
        help="Path to model file (for testing)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    try:
        if args.task == "train":
            logger.info("Starting training mode")
            run_training(args.config, args.languages, args.model_name)
        elif args.task == "evaluate":
            if not args.model_path:
                raise ValueError("Model path is required for evaluation")

            logger.info("Starting evaluation...")
            pipeline = Pipeline(args.config)
            results = evaluate_model(
                processed_data=pipeline.prepare_evaluation_data(),
                model_path=args.model_path,
                config=pipeline.config,
            )

            logger.info("Evaluation Results:")
            logger.info(f"Exact Match Ratio: {results['exact_match_ratio']:.4f}")
            logger.info(f"Macro Precision: {results['macro_precision']:.4f}")
            logger.info(f"Macro Recall: {results['macro_recall']:.4f}")
            logger.info(f"Macro F1: {results['macro_f1']:.4f}")
        elif args.task == "test":
            logger.info("Starting testing mode")
            run_testing(args.config, args.model_path)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
