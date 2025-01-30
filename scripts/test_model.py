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
from src.taxonomy import (
    get_main_roles, get_fine_roles,
    get_role_indices, get_fine_role_indices,
    get_main_role_indices, get_all_fine_roles
)


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
        article_ids = batch.get('article_ids', ['Unknown ID'] * len(input_ids))

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

        # Get role mappings from taxonomy
        main_role_indices = get_main_role_indices()
        main_role_map = {idx: role for role, idx in main_role_indices.items()}
        
        # Create valid role indices mapping like in model.py
        valid_role_indices = {}
        for main_role in get_main_roles():
            main_idx = main_role_indices[main_role]
            valid_fine_roles = get_fine_roles(main_role)
            valid_indices = []
            for role in valid_fine_roles:
                if role in model.fine_role_indices:
                    # Adjust index to be relative to fine-grained space
                    fine_idx = model.fine_role_indices[role] - len(main_role_indices)
                    valid_indices.append(fine_idx)
            valid_role_indices[main_idx] = sorted(valid_indices)

        # Print predictions for each entity in batch
        for i in range(len(main_preds)):
            # Get article ID and entity mention
            article_id = batch.get('article_id', ['Unknown ID'] * len(input_ids))[i]
            entity_mention = batch.get('entity_mention', [''] * len(input_ids))[i]
            
            logger.info(f"\nArticle ID: {article_id}")
            logger.info(f"Entity mention: {entity_mention}")
            
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
            
            # Get main role prediction
            main_pred_idx = main_preds[i].item()
            main_role = main_role_map[main_pred_idx]
            logger.info(f"Predicted main role: {main_role}")

            # Get fine-grained roles using valid_role_indices
            valid_indices = valid_role_indices[main_pred_idx]
            valid_probs = fine_probs[i, valid_indices]
            
            # Get top predictions above threshold
            k = min(2, len(valid_indices))
            top_values, top_local_indices = torch.topk(valid_probs, k=k)
            
            # Convert local indices to fine-grained space indices
            predicted_fine_roles = []
            for local_idx, value in enumerate(top_values):
                if value >= threshold:
                    fine_idx = valid_indices[top_local_indices[local_idx].item()]
                    # Convert back to role name using model's mapping
                    for role, idx in model.fine_role_indices.items():
                        if (idx - len(main_role_indices)) == fine_idx:
                            predicted_fine_roles.append(role)
                            break
            
            # Always take at least one role (highest probability)
            if not predicted_fine_roles:
                max_local_idx = valid_probs.argmax().item()
                fine_idx = valid_indices[max_local_idx]
                for role, idx in model.fine_role_indices.items():
                    if (idx - len(main_role_indices)) == fine_idx:
                        predicted_fine_roles.append(role)
                        break
            
            logger.info(f"Predicted fine-grained roles: {predicted_fine_roles}")

            # Print probabilities for debugging
            logger.debug("Fine-grained role probabilities:")
            for j, prob in enumerate(valid_probs):
                fine_idx = valid_indices[j]
                for role, idx in model.fine_role_indices.items():
                    if (idx - len(main_role_indices)) == fine_idx:
                        if prob > 0.3:  # Only show significant probabilities
                            logger.debug(f"{role}: {prob.item():.4f}")
                        break

            # If labels are available, show ground truth
            if 'main_labels' in batch and 'fine_labels' in batch:
                true_main_idx = batch['main_labels'][i].item()
                true_main_role = main_role_map[true_main_idx]
                logger.info(f"True main role: {true_main_role}")
                
                # Get true fine roles
                true_fine_roles = []
                # The labels are already in fine-grained space (0 to num_fine_roles-1)
                for j, is_role in enumerate(batch['fine_labels'][i]):
                    if is_role.item() == 1:
                        # Convert from fine-grained space index to role name
                        for role, idx in model.fine_role_indices.items():
                            # Convert global index to fine-grained space
                            fine_idx = idx - len(main_role_indices)
                            if fine_idx == j:
                                true_fine_roles.append(role)
                                break
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
