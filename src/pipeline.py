import torch
from datetime import datetime
import yaml  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import Dict, List, Tuple
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import numpy as np
import subprocess
import unicodedata
import os
from torch.optim import AdamW
from transformers import get_scheduler

from src.utils import set_seed, get_logger, get_device
from src.data_loader import DataLoader, Article, EntityDataset
from src.preprocessing import Preprocessor
from src.feature_extraction import FeatureExtractor
from src.model import EntityRoleClassifier
from src.taxonomy import load_taxonomy, validate_role

logger = get_logger(__name__)


class Pipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Set random seeds
        set_seed(self.config["training"]["seed"])

        # Initialize components
        self.device = get_device()
        self.preprocessor = Preprocessor()
        self.feature_extractor = None  # Will be initialized after model creation
        self.taxonomy = load_taxonomy(self.config["paths"]["taxonomy_path"])
        self.model = None

        logger.info(f"Pipeline initialized with device: {self.device}")

    def initialize_model(self, num_fine_labels: int):
        """Initialize the model with configuration"""
        self.model = EntityRoleClassifier(
            num_fine_labels=num_fine_labels,
            config=self.config
        ).to(self.device)

    def prepare_data(self, articles: List[Article], split: str) -> tuple[TensorDataset, List]:
        """
        Prepare data for training or validation.

        Args:
            articles (List[Article]): List of articles to process
            split (str): Split type (e.g. 'train', 'val', 'test')

        Returns:
            tuple[TensorDataset, List]: Processed dataset and list of successfully processed annotations
        """
        processed_texts = []
        entity_spans = []
        labels = []
        processed_annotations = []  # Track which annotations we successfully process

        logger.info(f"Processing {len(articles)} articles for {split}")

        for article in tqdm(articles, desc=f"Processing {split} articles"):
            if article.language not in self.config["languages"]["supported"]:
                logger.warning(
                    f"Skipping article {article.article_id} - unsupported language {article.language}")
                continue

            if not article.annotations:
                logger.warning(
                    f"Skipping article {article.article_id} - no annotations")
                continue

            try:
                # Use minimal preprocessing for dev/test to maintain span accuracy
                clean_text = self.preprocessor.preprocess_text(
                    article.content,
                    remove_urls=False,
                    remove_emojis=False,
                    remove_social=False,
                    normalize_unicode=True,  # Keep this for consistency
                    min_words_per_line=1,
                    remove_duplicate_lines=False,
                    remove_short_lines=False,
                    merge_paragraphs=False,
                    clean_text=True  # Basic cleaning only
                )

                for ann in article.annotations:
                    try:
                        # For training data, validate roles
                        if split == "train":
                            if not ann.main_role or not ann.fine_grained_roles:
                                logger.warning(
                                    f"Skipping annotation in {article.article_id} - missing roles")
                                continue
                            validate_role(self.taxonomy, ann.main_role,
                                          ann.fine_grained_roles)

                        # Get the original entity mention
                        orig_mention = article.content[ann.start_offset:ann.end_offset].strip(
                        )

                        # Find the entity mention in the cleaned text
                        # First try exact match
                        clean_start = clean_text.find(orig_mention)

                        # If exact match fails, try case-insensitive match
                        if clean_start == -1:
                            clean_start = clean_text.lower().find(orig_mention.lower())
                            if clean_start != -1:
                                # Use the actual case from clean_text
                                orig_mention = clean_text[clean_start:clean_start + len(
                                    orig_mention)]

                        # If still no match, try with unicode normalization
                        if clean_start == -1:
                            # Try with NFKD normalization (decomposed form)
                            norm_mention = unicodedata.normalize(
                                'NFKD', orig_mention)
                            norm_text = unicodedata.normalize(
                                'NFKD', clean_text)
                            clean_start = norm_text.find(norm_mention)

                            if clean_start != -1:
                                # Found match with normalized text, get the original text slice
                                clean_end = clean_start + len(norm_mention)
                                # Convert position in normalized text back to position in original text
                                orig_pos = 0
                                norm_pos = 0
                                while norm_pos < clean_start:
                                    # Skip combining accents
                                    if norm_text[norm_pos] != '\u0301':
                                        orig_pos += 1
                                    norm_pos += 1
                                clean_start = orig_pos

                                # Find end position
                                while norm_pos < clean_end:
                                    # Skip combining accents
                                    if norm_text[norm_pos] != '\u0301':
                                        orig_pos += 1
                                    norm_pos += 1
                                clean_end = orig_pos
                                orig_mention = clean_text[clean_start:clean_end]

                        # If still no match, try removing accents entirely
                        if clean_start == -1:
                            def remove_accents(text):
                                nfkd_form = unicodedata.normalize('NFKD', text)
                                return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

                            no_accent_mention = remove_accents(orig_mention)
                            no_accent_text = remove_accents(clean_text)
                            clean_start = no_accent_text.find(
                                no_accent_mention)

                            if clean_start != -1:
                                # Found match without accents, find corresponding position in original text
                                clean_end = clean_start + \
                                    len(no_accent_mention)
                                # Map back to original text positions
                                orig_pos = 0
                                no_accent_pos = 0
                                while no_accent_pos < clean_start:
                                    if unicodedata.combining(clean_text[orig_pos]):
                                        orig_pos += 1
                                    if no_accent_text[no_accent_pos] == no_accent_mention[0]:
                                        break
                                    orig_pos += 1
                                    no_accent_pos += 1
                                clean_start = orig_pos

                                # Find end position
                                while no_accent_pos < clean_end:
                                    if unicodedata.combining(clean_text[orig_pos]):
                                        orig_pos += 1
                                    orig_pos += 1
                                    no_accent_pos += 1
                                clean_end = orig_pos
                                orig_mention = clean_text[clean_start:clean_end]

                        # If still no match, try with punctuation removed
                        if clean_start == -1:
                            # Remove any leading/trailing punctuation and try again
                            core_mention = orig_mention.strip(
                                '.,!?()[]{}"\':;')  # Added single quote
                            clean_start = clean_text.find(core_mention)
                            if clean_start != -1:
                                orig_mention = core_mention

                        if clean_start != -1:
                            clean_end = clean_start + len(orig_mention)

                            # Verify the span
                            clean_span = clean_text[clean_start:clean_end].strip(
                            )

                            # Store text and spans
                            processed_texts.append(clean_text)
                            entity_spans.append((clean_start, clean_end))
                            processed_annotations.append(
                                ann)  # Track this annotation

                            # For validation/test data, use empty labels
                            if split != "train":
                                labels.append([])
                            else:
                                labels.append(ann.fine_grained_roles)
                        else:
                            logger.warning(
                                f"Could not find entity '{orig_mention}' in cleaned text for {article.article_id}")
                            continue

                    except ValueError as e:
                        logger.warning(
                            f"Skipping invalid annotation in {article.article_id}: {e}")
                        continue
                    except Exception as e:
                        logger.error(
                            f"Error processing annotation in {article.article_id}: {e}")
                        continue
            except Exception as e:
                logger.error(
                    f"Error processing article {article.article_id}: {e}")
                continue

        if not processed_texts:
            raise ValueError(
                f"No valid data processed for {split} split. Check the logs for details.")

        logger.info(
            f"Extracting features for {len(processed_texts)} examples")
        processed_features = self.feature_extractor.process_batch(
            processed_texts, entity_spans)
        logger.info(
            f"Successfully processed {len(processed_features)} examples for {split}")
        return self._create_tensor_dataset(processed_features, labels), processed_annotations

    def _create_tensor_dataset(
        self, features: List[Dict], labels: List[List[str]]
    ) -> TensorDataset:
        """Convert processed features and labels to TensorDataset using multi-hot encoding."""
        input_ids = torch.cat([f["input_ids"] for f in features])
        attention_masks = torch.cat([f["attention_mask"] for f in features])
        entity_positions = torch.cat([f["entity_position"] for f in features])

        # Get all possible roles
        main_roles = list(self.taxonomy.keys())
        all_fine_roles = []
        for subroles in self.taxonomy.values():
            all_fine_roles.extend(subroles)
        all_fine_roles = sorted(set(all_fine_roles))

        # Create main role labels (one-hot)
        main_labels_tensor = torch.zeros((len(labels), len(main_roles)), dtype=torch.long)
        
        # Create fine-grained role labels (multi-hot)
        fine_labels_tensor = torch.zeros((len(labels), len(all_fine_roles)), dtype=torch.float)
        
        for i, sample_labels in enumerate(labels):
            # Set main role label
            if sample_labels:  # Only for training data
                for label in sample_labels:
                    # Find which main role this label belongs to
                    for main_idx, (main_role, subroles) in enumerate(self.taxonomy.items()):
                        if label in subroles:
                            main_labels_tensor[i] = main_idx
                            break
                
                # Set fine-grained role labels
                for label in sample_labels:
                    if label in all_fine_roles:
                        label_idx = all_fine_roles.index(label)
                        fine_labels_tensor[i, label_idx] = 1.0

        # Create dummy embeddings tensor since we don't need it anymore
        embeddings = torch.zeros((len(features), 1), dtype=torch.float)

        return TensorDataset(input_ids, attention_masks, embeddings, entity_positions, main_labels_tensor, fine_labels_tensor)

    def train(self, languages: List[str] = ["EN", "PT"], model_name: str = "model.pt"):
        """
        Run training pipeline.

        Args:
            languages (List[str]): List of languages to process
            model_name (str): Name of the model to save
        """
        logger.info("Starting training pipeline")

        # Load training data
        logger.info(f"Loading training data for languages: {languages}")
        data_loader = DataLoader(config=self.config)
        train_articles = []
        for lang in languages:
            articles = data_loader.load_articles(self.config["paths"]["train_data_dir"], lang)
            train_articles.extend(articles)
        logger.info(f"Loaded {len(train_articles)} training articles")

        # Load validation data
        logger.info("Loading validation data")
        val_articles = []
        for lang in languages:
            articles = data_loader.load_articles(self.config["paths"]["dev_data_dir"], lang)
            val_articles.extend(articles)
        logger.info(f"Loaded {len(val_articles)} validation articles")

        # Initialize model
        logger.info("Initializing model")
        num_fine_labels = sum(len(subroles) for subroles in self.taxonomy.values())
        self.initialize_model(num_fine_labels)

        # Initialize feature extractor with model's tokenizer and base model
        logger.info("Initializing feature extractor")
        self.feature_extractor = FeatureExtractor(
            max_length=self.config["model"]["max_length"],
            context_window=self.config["preprocessing"]["context_window"]
        )
        self.feature_extractor.set_tokenizer_and_model(
            self.model.tokenizer, self.model.roberta)

        # Prepare datasets
        logger.info("Preparing datasets for training and validation")
        train_dataset, train_annotations = self.prepare_data(
            train_articles, "train")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        val_dataset, val_annotations = self.prepare_data(
            val_articles, "validation")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.config["model"]["batch_size"],
            shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.config["model"]["batch_size"]
        )

        # Train
        logger.info("Training model")
        self.train_model(train_loader, val_loader)

        # Save model
        logger.info("Saving model")
        output_path = Path(self.config["paths"]
                           ["model_output_dir"]) / model_name
        self.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")

    def prepare_evaluation_data(self, languages: List[str] = ["EN", "PT"]):
        """
        Prepare data for evaluation.

        Args:
            languages (List[str]): List of languages to process
        """
        data_loader = DataLoader(config=self.config)
        test_articles = []
        for lang in languages:
            articles = data_loader.load_articles(self.config["paths"]["test_data_dir"], lang)
            test_articles.extend(articles)
        return self.prepare_data(test_articles, "test")

    def test(self, languages: List[str] = ["EN", "PT"], model_path: str = "models/model.pt", eval_mode: str = "dev"):
        """
        Test the trained model and evaluate using the official scorer.

        Args:
            languages (List[str]): List of languages to process
            model_path (str): Path to the trained model
            eval_mode (str): Evaluation mode, either "dev" or "test"
        """
        logger.info(f"Starting testing pipeline in {eval_mode} mode")

        # Process each language separately
        for lang in languages:
            logger.info(f"Processing language: {lang}")

            # Load evaluation data
            logger.info(f"Loading {eval_mode} data for language: {lang}")
            data_dir = self.config["paths"][f"{eval_mode}_data_dir"]
            data_loader = DataLoader(config=self.config)
            eval_articles = data_loader.load_articles(data_dir, lang)

            if not eval_articles:
                logger.warning(
                    f"No {eval_mode} articles found for language {lang}, skipping...")
                continue

            # Load trained model (if not already loaded)
            if not hasattr(self, '_loaded_model_path') or self._loaded_model_path != model_path:
                logger.info(f"Loading model from {model_path}")
                self.load_model(model_path, sum(len(subroles) for subroles in self.taxonomy.values()))
                self.model.eval()
                self._loaded_model = self.model
                self._loaded_model_path = model_path
            else:
                self.model = self._loaded_model

            # Initialize feature extractor with model's tokenizer and base model (if needed)
            if self.feature_extractor is None:
                logger.info("Initializing feature extractor")
                self.feature_extractor = FeatureExtractor(
                    max_length=self.config["model"]["max_length"],
                    context_window=self.config["preprocessing"]["context_window"]
                )
                self.feature_extractor.set_tokenizer_and_model(
                    self.model.tokenizer, self.model.roberta)

            # First collect all annotations for the current language
            all_annotations = []
            for article in eval_articles:
                if article.language == lang:
                    all_annotations.extend(article.annotations)

            if not all_annotations:
                logger.warning(
                    f"No annotations found for language {lang}, skipping...")
                continue

            logger.info(f"Found {len(all_annotations)} annotations for {lang}")

            # Prepare evaluation dataset from articles containing annotations
            logger.info(f"Preparing {eval_mode} dataset for {lang}")
            eval_dataset, processed_annotations = self.prepare_data(
                eval_articles, eval_mode)
            eval_loader = TorchDataLoader(
                eval_dataset, batch_size=self.config["model"]["batch_size"])

            # Make predictions
            logger.info("Making predictions")
            predictions = []
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Testing {lang}"):
                    input_ids, attention_mask, embeddings, entity_positions, main_labels, fine_labels = [
                        b.to(self.device) for b in batch
                    ]

                    main_logits, fine_logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        entity_positions=entity_positions
                    )

                    # Get main role predictions
                    main_preds = torch.argmax(main_logits, dim=1).cpu().numpy()
                    
                    # Get fine-grained predictions
                    fine_probs = torch.sigmoid(fine_logits).cpu().numpy()
                    batch_predictions = (fine_probs > 0.5)

                    # Convert predictions to role labels
                    main_roles = list(self.taxonomy.keys())
                    all_fine_roles = []
                    for subroles in self.taxonomy.values():
                        all_fine_roles.extend(subroles)
                    all_fine_roles = sorted(set(all_fine_roles))

                    for main_pred, fine_pred, fine_prob in zip(main_preds, batch_predictions, fine_probs):
                        # Get main role
                        main_role = main_roles[main_pred]
                        
                        # Get fine-grained roles
                        pred_roles = [all_fine_roles[i] for i, is_role in enumerate(fine_pred) if is_role]
                        if not pred_roles:  # If no roles above threshold, take highest probability
                            highest_prob_idx = int(np.argmax(fine_prob))
                            pred_roles = [all_fine_roles[highest_prob_idx]]

                        predictions.append({
                            "main_role": main_role,
                            "fine_grained_roles": pred_roles
                        })

            # Verify we have the same number of predictions as processed annotations
            if len(predictions) != len(processed_annotations):
                raise ValueError(
                    f"Mismatch between number of predictions ({len(predictions)}) and processed annotations ({len(processed_annotations)})")

            # Save predictions for this language
            output_dir = Path("predictions")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"model_predictions_{lang}.tsv"

            logger.info(
                f"Saving {len(predictions)} predictions for {lang} to {output_path}")
            with open(output_path, "w") as f:
                # Write predictions in the same order as processed annotations
                for ann, pred in zip(processed_annotations, predictions):
                    line = "\t".join([
                        ann.article_id,
                        ann.entity_mention,
                        str(ann.start_offset),
                        str(ann.end_offset),
                        pred["main_role"],
                        "\t".join(pred["fine_grained_roles"])
                    ])
                    f.write(line + "\n")

            # Run official scorer for this language (only in dev mode)
            if eval_mode == "dev":
                logger.info(f"Running official scorer for {lang}")

                # Get gold standard file for current language
                gold_file = Path(data_dir) / lang / "subtask-1-annotations.txt"

                # Ensure paths are absolute
                script_path = Path("data/scorers/subtask1_scorer.py").resolve()
                gold_file = gold_file.resolve()
                output_path = output_path.resolve()

                logger.info(f"Using scorer: {script_path}")
                logger.info(f"Gold standard file: {gold_file}")
                logger.info(f"Predictions file: {output_path}")

                if not script_path.exists():
                    raise FileNotFoundError(
                        f"Scorer script not found at {script_path}")
                if not gold_file.exists():
                    raise FileNotFoundError(
                        f"Gold standard file not found at {gold_file}")
                if not output_path.exists():
                    raise FileNotFoundError(
                        f"Predictions file not found at {output_path}")

                cmd = [
                    "python",
                    str(script_path),
                    "-g", str(gold_file),
                    "-p", str(output_path),
                    "-l"  # Add language-specific evaluation
                ]

                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True)
                    logger.info(f"Scorer output for {lang}:")
                    logger.info(result.stdout)
                    if result.stderr:
                        logger.warning(f"Scorer warnings for {lang}:")
                        logger.warning(result.stderr)
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Scorer failed for {lang} with exit code {e.returncode}")
                    if e.output:
                        logger.error(f"Scorer output: {e.output}")
                    if e.stderr:
                        logger.error(f"Scorer stderr: {e.stderr}")
                    raise
            else:
                logger.info(f"Skipping scoring in test mode for {lang}")

        logger.info(f"Testing completed successfully in {eval_mode} mode")

    def train_model(self, train_loader: TorchDataLoader, val_loader: TorchDataLoader) -> None:
        """Train the model with the given data loaders"""
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["model"]["learning_rate"],
            weight_decay=self.config["model"]["weight_decay"]
        )
        
        num_warmup_steps = int(self.config["model"]["num_training_steps"] * self.config["model"]["warmup_ratio"])
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.config["model"]["num_training_steps"]
        )
        
        # Loss functions
        main_loss_fn = torch.nn.CrossEntropyLoss()
        fine_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.config["model"]["num_epochs"]):
            self.model.train()
            total_loss = 0
            
            # Training step
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                entity_positions = batch["entity_position"].to(self.device)
                main_labels = batch["main_labels"].to(self.device)
                fine_labels = batch["fine_labels"].to(self.device)
                
                # Forward pass
                main_logits, fine_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions
                )
                
                # Calculate losses
                main_loss = main_loss_fn(main_logits, main_labels)
                fine_loss = fine_loss_fn(fine_logits, fine_labels)
                loss = main_loss + fine_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            
            # Validation step
            val_loss = self.evaluate(val_loader)
            
            self.logger.info(
                f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}, "
                f"Validation loss: {val_loss:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")
    
    def evaluate(self, val_loader: TorchDataLoader) -> float:
        """Evaluate the model on validation data"""
        self.model.eval()
        total_loss = 0
        main_loss_fn = torch.nn.CrossEntropyLoss()
        fine_loss_fn = torch.nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                entity_positions = batch["entity_position"].to(self.device)
                main_labels = batch["main_labels"].to(self.device)
                fine_labels = batch["fine_labels"].to(self.device)
                
                main_logits, fine_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions
                )
                
                main_loss = main_loss_fn(main_logits, main_labels)
                fine_loss = fine_loss_fn(fine_logits, fine_labels)
                loss = main_loss + fine_loss
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, test_loader: TorchDataLoader) -> List[Dict]:
        """Generate predictions for test data"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                entity_positions = batch["entity_position"].to(self.device)
                
                main_logits, fine_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions
                )
                
                # Get predictions
                main_preds = torch.argmax(main_logits, dim=1)
                fine_preds = (torch.sigmoid(fine_logits) > 0.5).float()
                
                # Convert to CPU and numpy
                main_preds = main_preds.cpu().numpy()
                fine_preds = fine_preds.cpu().numpy()
                
                # Store predictions
                for i in range(len(main_preds)):
                    predictions.append({
                        "main_role": main_preds[i],
                        "fine_roles": fine_preds[i]
                    })
        
        return predictions
    
    def save_model(self, path: str) -> None:
        """Save the model to the specified path"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        
    def load_model(self, path: str, num_fine_labels: int) -> None:
        """Load the model from the specified path"""
        self.model = EntityRoleClassifier.from_pretrained(
            path=path,
            config=self.config,
            num_fine_labels=num_fine_labels
        ).to(self.device)


def run_training(config_path: str = "configs.yaml", languages: List[str] = ["EN"], model_name: str = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"):
    """Entry point for training pipeline."""
    try:
        pipeline = Pipeline(config_path)
        pipeline.train(languages=languages, model_name=model_name)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_testing(config_path: str = "configs.yaml", model_path: str = "models/model.pt"):
    """Entry point for testing pipeline."""
    try:
        pipeline = Pipeline(config_path)
        pipeline.test(model_path=model_path)
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise
