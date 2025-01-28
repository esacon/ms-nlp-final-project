import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from src.train_pipeline import run_training, run_evaluation
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate entity role classification model"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train", "evaluate"],
        help="Task to perform: train or evaluate"
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        default="EN,PT",
        help="Languages to process (comma-separated). Example: EN,PT"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models" if "--task" == "train" else "predictions",
        help="Directory to save outputs (models or predictions)"
    )
    
    parser.add_argument(
        "-f",
        "--model_path",
        type=str,
        help="Path to model checkpoint (required for evaluation)"
    )

    parser.add_argument(
        "--max_articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for testing)"
    )

    args = parser.parse_args()
    
    # Convert languages string to list
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    try:
        # Make sure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.task == "train":
            logger.info(f"Starting training for languages: {languages}")
            run_training(
                config_path=args.config,
                languages=languages,
                max_articles=args.max_articles
            )
            
        elif args.task == "evaluate":
            if not args.model_path:
                raise ValueError("Model path (-f) is required for evaluation")
                
            logger.info(f"Starting evaluation for languages: {languages}")
            run_evaluation(
                config_path=args.config,
                model_path=args.model_path,
                languages=languages,
                output_dir=args.output_dir,
                split="test",
                max_articles=args.max_articles
            )
            
    except Exception as e:
        logger.error(f"Error in {args.task}: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
