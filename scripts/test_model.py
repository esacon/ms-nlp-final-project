import sys
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from pathlib import Path
from typing import Dict, Any

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.model import EntityRoleClassifier
from src.utils import get_logger, load_config


logger = get_logger(__name__)


def test_model_predictions(model: EntityRoleClassifier, batch: Dict[str, Any], tokenizer: XLMRobertaTokenizerFast, threshold: float = 0.7) -> None:
    """
    Test model predictions on a single batch.

    Args:
        model: The model to test.
        batch: The batch of data to test.
        tokenizer: The tokenizer to use for the model.
        threshold: The threshold to use for the fine-grained roles.
    """

    model.eval()
    with torch.no_grad():
        # Ensure all tensors are properly shaped
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        entity_positions = batch['entity_position']

        # Print shapes for debugging
        logger.debug(f"Input shapes:")
        logger.debug(f"input_ids: {input_ids.shape}")
        logger.debug(f"attention_mask: {attention_mask.shape}")
        logger.debug(f"entity_positions: {entity_positions.shape}")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_positions=entity_positions
        )

        # Get predictions from logits
        main_preds = torch.argmax(outputs.main_logits, dim=1)
        fine_probs = torch.sigmoid(outputs.fine_logits)
        fine_preds = (fine_probs > threshold).float()

        # Map predictions to labels
        main_role_map = {0: "Protagonist", 1: "Antagonist", 2: "Innocent"}
        fine_role_map = [
            "Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous",
            "Instigator", "Conspirator", "Tyrant", "Foreign Adversary", "Traitor",
            "Spy", "Saboteur", "Corrupt", "Incompetent", "Terrorist", "Deceiver",
            "Bigot", "Forgotten", "Exploited", "Victim", "Scapegoat"
        ]

        # Print predictions for each entity in batch
        for i in range(len(main_preds)):
            # Get original text
            input_tokens = tokenizer.convert_ids_to_tokens(
                input_ids[i, 0] if input_ids.dim() == 3 else input_ids[i])
            start = entity_positions[i, 0, 0].item() if entity_positions.dim(
            ) == 3 else entity_positions[i, 0].item()
            end = entity_positions[i, 0, 1].item() if entity_positions.dim(
            ) == 3 else entity_positions[i, 1].item()
            entity_text = tokenizer.convert_tokens_to_string(
                input_tokens[start:end+1])

            logger.info(f"\nEntity: {entity_text}")
            logger.info(
                f"Predicted main role: {main_role_map[main_preds[i].item()]}")

            # Get fine-grained roles
            predicted_fine_roles = [
                fine_role_map[j] for j, is_role in enumerate(fine_preds[i])
                if is_role.item() == 1
            ]
            logger.info(
                f"Predicted fine-grained roles: {predicted_fine_roles}")

            # If labels are available, show ground truth
            if 'main_labels' in batch and 'fine_labels' in batch:
                true_main_role = batch['main_labels'][i].item()
                true_fine_roles = batch['fine_labels'][i]

                logger.info(
                    f"True main role: {main_role_map[true_main_role]}")
                true_fine_roles = [
                    fine_role_map[j] for j, is_role in enumerate(true_fine_roles)
                    if is_role.item() == 1
                ]
                logger.info(f"True fine-grained roles: {true_fine_roles}")


def main():
    config = load_config()

    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader(
            config, max_articles=config["testing"]["max_articles"])

        # Initialize feature extractor with base model
        logger.info("Loading base model and tokenizer...")
        base_model_name = config["model"]["name"]
        feature_extractor = FeatureExtractor(
            special_tokens=config["model"]["special_tokens"],
            max_length=config["model"]["max_length"],
            context_window=config["model"]["context_window"],
            batch_size=config["model"]["batch_size"],
            preprocessing_config=config["preprocessing"]
        )

        # Load tokenizer and model for feature extraction
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(base_model_name)
        # Add special tokens for entity markers
        special_tokens = {
            'additional_special_tokens': config["model"]["special_tokens"]}
        tokenizer.add_special_tokens(special_tokens)

        model_base = XLMRobertaModel.from_pretrained(base_model_name)
        # Resize token embeddings to account for new special tokens
        model_base.resize_token_embeddings(len(tokenizer))
        feature_extractor.set_tokenizer_and_model(tokenizer, model_base)

        # Initialize our classifier model with the same base model name and tokenizer
        logger.info("Initializing classifier model...")
        model = EntityRoleClassifier(config=config)
        # Ensure the classifier's base model has the same vocabulary size
        model.base_model.resize_token_embeddings(len(tokenizer))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Process sample articles for each language
        for language in config["languages"]["supported"]:
            logger.info(f"\nTesting {language} articles...")

            # Load articles
            articles = data_loader.load_articles(
                config["paths"]["train_data_dir"], language)
            logger.info(
                f"Loaded {len(articles)} sample articles for {language}")

            # Sample output of preprocessing
            if articles:
                sample_article = articles[0]
                logger.debug("\nSample preprocessing result:")
                logger.debug(
                    f"Original text snippet: {sample_article.text[:200]}...")
                logger.debug(
                    f"Preprocessed text snippet: {sample_article.preprocessed_text[:200]}...")

                if sample_article.annotations:
                    sample_entity = sample_article.annotations[0]
                    logger.debug("\nSample entity:")
                    logger.debug(
                        f"Entity mention: {sample_entity.entity_mention}")
                    logger.debug(
                        f"Offsets: {sample_entity.start_offset}, {sample_entity.end_offset}")
                    if sample_entity.main_role:
                        logger.debug(f"Main role: {sample_entity.main_role}")
                        logger.debug(
                            f"Fine-grained roles: {sample_entity.fine_grained_roles}")

            # Extract features
            logger.debug("\nExtracting features...")
            features, labels = data_loader.prepare_features(
                articles, feature_extractor)
            logger.debug(f"Extracted features for {len(features)} entities")

            if not features:
                logger.warning(f"No features extracted for {language}")
                continue

            # Create dataloader
            dataloader = data_loader.create_dataloader(
                features=features,
                labels=labels,
                batch_size=config["model"]["batch_size"],
                shuffle=False
            )

            # Test model on one batch
            logger.debug("\nTesting model predictions...")
            sample_batch = next(iter(dataloader))
            sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample_batch.items()}

            test_model_predictions(model, sample_batch,
                                   feature_extractor.tokenizer, threshold=config["model"]["threshold"])

    except Exception as e:
        logger.error(f"Error in testing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
