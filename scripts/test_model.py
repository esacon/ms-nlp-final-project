import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.model import EntityRoleClassifier
from src.utils import get_logger
import yaml
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast

logger = get_logger(__name__)

def load_config():
    """Load configuration from yaml file."""
    config_path = os.path.join(project_root, "src", "configs.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_model_predictions(model, batch, tokenizer):
    """Test model predictions on a single batch."""
    model.eval()
    with torch.no_grad():
        # Ensure all tensors are properly shaped
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        entity_positions = batch['entity_position']
        embeddings = batch.get('embeddings')

        # Print shapes for debugging
        logger.info(f"Input shapes:")
        logger.info(f"input_ids: {input_ids.shape}")
        logger.info(f"attention_mask: {attention_mask.shape}")
        logger.info(f"entity_positions: {entity_positions.shape}")
        if embeddings is not None:
            logger.info(f"embeddings: {embeddings.shape}")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_positions=entity_positions,
            embeddings=embeddings
        )
        
        # Get predictions
        main_preds = torch.argmax(outputs.main_logits, dim=1)
        fine_preds = (outputs.fine_logits > 0).float()
        
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
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[i, 0] if input_ids.dim() == 3 else input_ids[i])
            start = entity_positions[i, 0, 0].item() if entity_positions.dim() == 3 else entity_positions[i, 0].item()
            end = entity_positions[i, 0, 1].item() if entity_positions.dim() == 3 else entity_positions[i, 1].item()
            entity_text = tokenizer.convert_tokens_to_string(input_tokens[start:end+1])
            
            logger.info(f"\nEntity: {entity_text}")
            logger.info(f"Predicted main role: {main_role_map[main_preds[i].item()]}")
            
            # Get fine-grained roles
            predicted_fine_roles = [
                fine_role_map[j] for j, is_role in enumerate(fine_preds[i])
                if is_role.item() == 1
            ]
            logger.info(f"Predicted fine-grained roles: {predicted_fine_roles}")
            
            # If labels are available, show ground truth
            if 'main_labels' in batch and 'fine_labels' in batch:
                logger.info(f"True main role: {main_role_map[batch['main_labels'][i].item()]}")
                true_fine_roles = [
                    fine_role_map[j] for j, is_role in enumerate(batch['fine_labels'][i])
                    if is_role.item() == 1
                ]
                logger.info(f"True fine-grained roles: {true_fine_roles}")

def main():
    # Load configuration
    config = load_config()
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_loader = DataLoader(config)
        
        # Initialize feature extractor with base model
        logger.info("Loading base model and tokenizer...")
        base_model_name = "roberta-base"  # Make sure this matches what we use in the feature extractor
        feature_extractor = FeatureExtractor(
            max_length=config["model"]["max_length"],
            context_window=config["model"]["context_window"],
            batch_size=config["model"]["batch_size"],
            preprocessing_config=config["preprocessing"]
        )

        # Load tokenizer and model for feature extraction
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(base_model_name)
        model_base = XLMRobertaModel.from_pretrained(base_model_name)
        feature_extractor.set_tokenizer_and_model(tokenizer, model_base)

        # Initialize our classifier model with the same base model name
        logger.info("Initializing classifier model...")
        config["model"]["name"] = base_model_name  # Ensure we use the same model
        model = EntityRoleClassifier(num_fine_labels=22, config=config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Process sample articles for each language
        for language in ["EN", "PT"]:
            logger.info(f"\nTesting {language} articles...")

            # Load articles
            articles = data_loader.load_articles("data", language)[:3]  # Test with 3 articles
            logger.info(f"Loaded {len(articles)} sample articles for {language}")

            # Sample output of preprocessing
            if articles:
                sample_article = articles[0]
                logger.info("\nSample preprocessing result:")
                logger.info(f"Original text snippet: {sample_article.text[:200]}...")
                logger.info(f"Preprocessed text snippet: {sample_article.preprocessed_text[:200]}...")

                if sample_article.annotations:
                    sample_entity = sample_article.annotations[0]
                    logger.info("\nSample entity:")
                    logger.info(f"Entity mention: {sample_entity.entity_mention}")
                    logger.info(f"Offsets: {sample_entity.start_offset}, {sample_entity.end_offset}")
                    if sample_entity.main_role:
                        logger.info(f"Main role: {sample_entity.main_role}")
                        logger.info(f"Fine-grained roles: {sample_entity.fine_grained_roles}")

            # Extract features
            logger.info("\nExtracting features...")
            features, labels = data_loader.prepare_features(articles, feature_extractor)
            logger.info(f"Extracted features for {len(features)} entities")

            if not features:
                logger.warning(f"No features extracted for {language}")
                continue

            # Create dataloader
            dataloader = data_loader.create_dataloader(
                features=features,
                labels=labels,
                batch_size=config["model"]["batch_size"]
            )
            logger.info(f"Created DataLoader with {len(dataloader)} batches")

            # Test model on one batch
            logger.info("\nTesting model predictions...")
            sample_batch = next(iter(dataloader))
            sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in sample_batch.items()}
            
            test_model_predictions(model, sample_batch, feature_extractor.tokenizer)

    except Exception as e:
        logger.error(f"Error in testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 