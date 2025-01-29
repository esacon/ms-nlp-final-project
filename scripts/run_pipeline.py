from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.table import Table
from typing import Dict, Any

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.utils import get_logger
from src.feature_extraction import FeatureExtractor
from src.data_loader import DataLoader
import yaml

logger = get_logger(__name__)
console = Console()


def visualize_entity_analysis(features: Dict[str, Any], tokenizer, entity_text: str) -> None:
    """Analyze and display extracted features."""
    print("\nFeature Analysis:")

    # Get basic information
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    entity_position = features["entity_position"]

    # Create tokens table
    table = Table(title=f"Entity Analysis: {entity_text}")
    table.add_column("Position", style="cyan", justify="right")
    table.add_column("Token", style="green", justify="left")
    table.add_column("Attention", style="yellow", justify="center")
    table.add_column("Embedding Norm", style="magenta", justify="right")
    table.add_column("Context", style="blue", justify="left")

    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Calculate context window
    start_pos = max(0, entity_position[0] - 5)
    end_pos = min(len(tokens), entity_position[1] + 5)

    # Get embedding norms
    embedding_norm = torch.norm(torch.tensor(
        features["entity_embeddings"][0])).item()

    # Add rows for context window
    for i in range(start_pos, end_pos):
        token = tokens[i]
        # Escape special characters for Rich
        display_token = token.replace("[", "\\[").replace("]", "\\]")

        # Highlight entity tokens
        if entity_position[0] <= i <= entity_position[1]:
            token_style = "bold red"
        else:
            token_style = "green"

        attention = "●" if attention_mask[i] == 1 else "○"
        context = "..." if i in [start_pos, end_pos-1] else ""

        table.add_row(
            str(i),
            f"[{token_style}]{display_token}[/{token_style}]",
            attention,
            f"{embedding_norm:.2f}",
            context
        )

    console.print(table)

    # Print embedding statistics
    entity_embeddings = torch.tensor(features["entity_embeddings"])
    if len(entity_embeddings) > 0:
        print("\nEntity Embedding Statistics:")
        print(f"  Mean: {entity_embeddings.mean().item():.3f}")
        print(f"  Std: {entity_embeddings.std().item():.3f}")
        print(f"  Max: {entity_embeddings.max().item():.3f}")
        print(f"  Min: {entity_embeddings.min().item():.3f}")

    console.print()


def main():
    # Initialize components
    print("─" * 100)
    print("Initializing Pipeline Components")
    print("─" * 100)
    
    config_path = "src/configs.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Initialize feature extractor with model and tokenizer
    feature_extractor = FeatureExtractor()
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(config["model"]["name"])
    model = XLMRobertaModel.from_pretrained(config["model"]["name"])
    feature_extractor.set_tokenizer_and_model(tokenizer, model)

    data_loader = DataLoader(config, max_articles=3)

    # Process articles
    for lang in ["EN", "PT"]:
        print(f"\n{'─' * 100}")
        print(f"Processing {lang} Articles")
        print("─" * 100)

        # Load articles
        articles = data_loader.load_articles("data", lang)
        print(f"\nLoaded {len(articles)} articles for {lang}")

        # Process first few articles with annotations
        article_count = 0
        for article in articles:
            if not article.annotations:
                continue

            article_count += 1
            if article_count > 3:  # Process only first 3 articles with annotations
                break

            print(f"\n{'─' * 100}")
            print(f"Article {article_count}")
            print("─" * 100)

            # Show article preview
            preview = article.text[:200] + \
                "..." if len(article.text) > 200 else article.text
            print("Article Preview:")
            print(preview)
            print()

            # Process each entity
            for i, annotation in enumerate(article.annotations, 1):
                print(f"Entity {i}:")
                print(f"  Mention: {annotation.entity_mention}")
                print(
                    f"  Offset: {annotation.start_offset}, {annotation.end_offset}")
                print(f"  Main Role: {annotation.main_role}")
                print(f"  Fine-grained Roles: {annotation.fine_grained_roles}")

                try:
                    # Extract features
                    features = feature_extractor.extract_features(
                        text=article.text,
                        entity_text=annotation.entity_mention,
                        start_offset=annotation.start_offset,
                        end_offset=annotation.end_offset
                    )

                    # Analyze and visualize features
                    visualize_entity_analysis(
                        features, feature_extractor.tokenizer, annotation.entity_mention)
                except Exception as e:
                    logger.error(f"Error in pipeline: {str(e)}")
                    raise

            input("\nPress Enter to continue to next article...")


if __name__ == "__main__":
    main()
