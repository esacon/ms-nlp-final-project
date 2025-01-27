import csv
import torch
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Dict, List
from src.model import load_trained_model
from src.taxonomy import load_taxonomy
from src.data_loader import Article, DataLoader
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


def exact_match_ratio(y_true: List[List[str]], y_pred: List[List[str]]) -> float:
    """
    Calculate exact match ratio for multi-label classification.
    Returns proportion of samples where all predicted labels match ground truth.
    """
    matches = sum(1 for true, pred in zip(y_true, y_pred) if set(true) == set(pred))
    return matches / len(y_true) if y_true else 0


def evaluate_model(
    processed_data: List[Article], model_path: str, config: Dict
) -> Dict:
    """
    Evaluate model performance using exact match ratio and per-role metrics.

    Args:
        processed_data: List of processed articles with annotations
        model_path: Path to saved model
        config: Configuration dictionary

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model and taxonomy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path)
    model.to(device)
    model.eval()

    taxonomy = load_taxonomy(config["paths"]["taxonomy_path"])
    main_roles = list(taxonomy.keys())
    all_fine_roles = []
    for subroles in taxonomy.values():
        all_fine_roles.extend(subroles)
    all_fine_roles = sorted(set(all_fine_roles))

    # Collect true and predicted labels
    y_true_main = []
    y_pred_main = []
    y_true_fine = []
    y_pred_fine = []

    with torch.no_grad():
        for article in processed_data:
            for ann in article.annotations:
                if ann.main_role and ann.fine_grained_roles:  # Only evaluate annotations with gold labels
                    # Get model predictions
                    features = model.feature_extractor.extract_features(
                        article.content[
                            max(0, ann.start_offset - 100) : min(
                                len(article.content), ann.end_offset + 100
                            )
                        ]
                    )

                    main_logits, fine_logits = model(
                        features["input_ids"].to(device),
                        features["attention_mask"].to(device),
                        features["entity_position"].to(device)
                    )

                    # Main role prediction
                    main_pred = main_roles[torch.argmax(main_logits[0]).item()]
                    
                    # Fine-grained role predictions
                    fine_preds = []
                    for idx, logit in enumerate(fine_logits[0]):
                        if logit > 0:  # Using 0 as threshold for binary decisions
                            fine_preds.append(all_fine_roles[idx])

                    if not fine_preds:  # If no roles above threshold, take highest probability
                        top_idx = torch.argmax(fine_logits[0]).item()
                        fine_preds = [all_fine_roles[top_idx]]

                    y_true_main.append(ann.main_role)
                    y_pred_main.append(main_pred)
                    y_true_fine.append(ann.fine_grained_roles)
                    y_pred_fine.append(fine_preds)

    # Calculate main role accuracy
    main_accuracy = accuracy_score(y_true_main, y_pred_main)

    # Calculate fine-grained metrics
    # Convert to binary matrix format for sklearn metrics
    y_true_binary = [
        [1 if role in sample else 0 for role in all_fine_roles] for sample in y_true_fine
    ]
    y_pred_binary = [
        [1 if role in sample else 0 for role in all_fine_roles] for sample in y_pred_fine
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="macro"
    )

    # Calculate exact match ratio for fine-grained roles
    exact_match = exact_match_ratio(y_true_fine, y_pred_fine)

    metrics = {
        "main_role_accuracy": main_accuracy,
        "fine_grained_exact_match": exact_match,
        "fine_grained_precision": precision,
        "fine_grained_recall": recall,
        "fine_grained_f1": f1,
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def generate_predictions_file(model_path: str, data_dir: str, output_path: str, languages: List[str]):
    """
    Generate predictions file in the format required by the scorer.
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing the data to predict on
        output_path: Where to save the predictions
        languages: List of languages to process
    """
    logger.info(f"Loading model from {model_path}")
    # Load model and required components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path)
    model.to(device)
    model.eval()

    logger.info("Loading taxonomy and initializing components")
    taxonomy = load_taxonomy("src/taxonomy.json")
    preprocessor = Preprocessor()
    feature_extractor = FeatureExtractor()

    # Get all possible roles
    main_roles = list(taxonomy.keys())
    all_fine_roles = []
    for subroles in taxonomy.values():
        all_fine_roles.extend(subroles)
    all_fine_roles = sorted(set(all_fine_roles))

    # Load data
    logger.info(f"Loading data from {data_dir}")
    data_loader = DataLoader(data_dir, "dev")
    data_loader.load_data(languages)
    articles = data_loader.get_articles_by_split("dev")
    logger.info(f"Loaded {len(articles)} articles")

    # Generate predictions
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        for article in articles:
            logger.info(f"Processing article {article.article_id}")
            clean_text = preprocessor.preprocess_text(article.content)
            
            for ann in article.annotations:
                logger.debug(f"Processing mention: {ann.entity_mention}")
                # Get context around mention
                context_start = max(0, ann.start_offset - 100)
                context_end = min(len(clean_text), ann.end_offset + 100)
                mention_context = clean_text[context_start:context_end]
                
                # Extract features
                features = feature_extractor.extract_features(mention_context)
                
                # Get predictions
                with torch.no_grad():
                    main_logits, fine_logits = model(
                        input_ids=features["input_ids"].to(device),
                        attention_mask=features["attention_mask"].to(device),
                        entity_positions=features["entity_position"].to(device)
                    )
                    
                    # Main role prediction
                    main_pred = main_roles[torch.argmax(main_logits[0]).item()]
                    
                    # Fine-grained predictions
                    fine_probs = torch.sigmoid(fine_logits[0])
                    threshold = 0.3  # Lower threshold for fine-grained roles
                    fine_preds = [all_fine_roles[i] for i, prob in enumerate(fine_probs) if prob > threshold]
                    
                    if not fine_preds:  # If no roles above threshold, take highest probability
                        top_idx = torch.argmax(fine_probs).item()
                        fine_preds = [all_fine_roles[top_idx]]
                    
                    # Write prediction
                    writer.writerow([
                        article.article_id,
                        ann.entity_mention,
                        str(ann.start_offset),
                        str(ann.end_offset),
                        main_pred,
                        *fine_preds
                    ])
                    logger.debug(f"Predicted roles for {ann.entity_mention}: {main_pred}, {fine_preds}")

    logger.info(f"Predictions saved to {output_path}")
