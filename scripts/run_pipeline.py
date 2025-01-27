import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from transformers import RobertaTokenizerFast, RobertaModel
import torch
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.utils import get_logger
import json

logger = get_logger(__name__)


def main():
    # Configuration
    config = {
        "preprocessing": {
            "remove_urls": True,
            "remove_emojis": True,
            "remove_social": True,
            "normalize_unicode": True,
            "min_words_per_line": 4,
            "remove_duplicate_lines": True,
            "remove_short_lines": True,
            "merge_paragraphs": True,
            "clean_text": True
        },
        "model": {
            "name": "roberta-base",
            "max_length": 512,
            "context_window": 128,
            "batch_size": 32
        }
    }

    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader(config)
        feature_extractor = FeatureExtractor(
            max_length=config["model"]["max_length"],
            context_window=config["model"]["context_window"],
            batch_size=config["model"]["batch_size"],
            preprocessing_config=config["preprocessing"]
        )

        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base")
        feature_extractor.set_tokenizer_and_model(tokenizer, model)

        # Process each language
        for language in ["EN", "PT"]:
            logger.info(f"Processing {language} articles...")

            # Load articles
            articles = data_loader.load_articles("data", language)
            logger.info(f"Loaded {len(articles)} articles for {language}")

            # Sample output of preprocessing
            if articles:
                sample_article = articles[0]
                logger.info("\nSample preprocessing result:")
                logger.info(
                    f"Original text snippet: {sample_article.text[:200]}...")
                logger.info(
                    f"Preprocessed text snippet: {sample_article.preprocessed_text[:200]}...")

                if sample_article.annotations:
                    sample_entity = sample_article.annotations[0]
                    logger.info("\nSample entity:")
                    logger.info(
                        f"Entity mention: {sample_entity.entity_mention}")
                    logger.info(
                        f"Offsets: {sample_entity.start_offset}, {sample_entity.end_offset}")
                    if sample_entity.main_role:
                        logger.info(f"Main role: {sample_entity.main_role}")
                        logger.info(
                            f"Fine-grained roles: {sample_entity.fine_grained_roles}")

            # Extract features
            logger.info("\nExtracting features...")
            features, labels = data_loader.prepare_features(
                articles, feature_extractor)
            logger.info(f"Extracted features for {len(features)} entities")

            # Create dataloader
            dataloader = data_loader.create_dataloader(
                features=features,
                labels=labels,
                batch_size=config["model"]["batch_size"]
            )
            logger.info(f"Created DataLoader with {len(dataloader)} batches")

            # Sample batch inspection
            sample_batch = next(iter(dataloader))
            logger.info("\nSample batch structure:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(
                        f"{key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    logger.info(f"{key}: {type(value)}")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
