import os
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

from src.utils import set_seed, get_logger, get_device, load_config
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.model import EntityRoleClassifier
from src.taxonomy import (
    get_main_roles,
    get_fine_roles,
    get_main_role_indices,
)

logger = get_logger(__name__)


@dataclass
class PipelineOutput:
    """Container for pipeline outputs"""
    loss: float
    main_acc: float
    fine_acc: float
    predictions: Optional[List[Dict[str, Any]]] = None


class Pipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = load_config(config_path)
        self.device = get_device()

        # Set random seeds
        set_seed(self.config["training"]["seed"])

        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.base_model = None

        # Initialize components
        self._initialize_components()

        logger.info(f"Pipeline initialized with device: {self.device}")

    def _initialize_components(self) -> None:
        """Initialize all model components."""
        logger.info("Initializing components...")

        # Initialize feature extractor
        logger.info("Loading base model and tokenizer...")
        self.feature_extractor = FeatureExtractor(
            special_tokens=self.config["model"]["special_tokens"],
            max_length=self.config["model"]["max_length"],
            context_window=self.config["model"]["context_window"],
            batch_size=self.config["model"]["batch_size"],
            preprocessing_config=self.config["preprocessing"]
        )

        # Load tokenizer and base model
        base_model_name = self.config["model"]["name"]
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            base_model_name)

        # Add special tokens for entity markers
        special_tokens = {
            'additional_special_tokens': self.config["model"]["special_tokens"]}
        self.tokenizer.add_special_tokens(special_tokens)

        self.base_model = XLMRobertaModel.from_pretrained(base_model_name)
        # Resize token embeddings to account for new special tokens
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.feature_extractor.set_tokenizer_and_model(
            self.tokenizer, self.base_model)

        # Initialize classifier model
        logger.info("Initializing classifier model...")
        self.model = EntityRoleClassifier(config=self.config)
        # Ensure the classifier's base model has the same vocabulary size
        self.model.base_model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)

    def prepare_data(
        self,
        languages: List[str],
        split: str,
        max_articles: Optional[int] = None
    ) -> Tuple[TorchDataLoader, List]:
        """
        Prepare data for training or evaluation.

        Args:
            languages: List of language codes to process
            split: Data split to use ('train', 'dev', or 'test')
            max_articles: Maximum number of articles to load (for testing)

        Returns:
            Tuple of (DataLoader, processed articles)
        """
        logger.info(f"Preparing {split} data for languages: {languages}")

        all_articles = []
        for lang in languages:
            articles = self.data_loader.load_articles(
                self.config["paths"][f"{split}_data_dir"],
                split=split,
                language=lang
            )
            if max_articles:
                articles = articles[:max_articles]
                logger.info(f"Limited to {max_articles} articles for testing")
            all_articles.extend(articles)
            logger.info(f"Loaded {len(articles)} {split} articles for {lang}")

        # Extract features
        logger.info("\nExtracting features...")
        features, labels = self.data_loader.prepare_features(
            all_articles,
            self.feature_extractor
        )
        logger.info(f"Extracted features for {len(features)} entities")

        # Create dataloader
        dataloader = self.data_loader.create_dataloader(
            features=features,
            labels=labels,
            batch_size=self.config["model"]["batch_size"],
            shuffle=(split == "train")
        )

        return dataloader, all_articles

    def _train_epoch(
        self,
        train_loader: TorchDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int
    ) -> PipelineOutput:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_main_acc = 0
        total_fine_acc = 0
        steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_positions=batch["entity_position"],
                labels=torch.cat(
                    [batch["main_labels"].unsqueeze(-1), batch["fine_labels"]], dim=-1)
            )

            # Backward pass
            optimizer.zero_grad()
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate metrics
            with torch.no_grad():
                main_preds = torch.argmax(outputs.main_logits, dim=1)
                main_acc = (main_preds == batch["main_labels"]).float().mean()

                fine_preds = (torch.sigmoid(outputs.fine_logits)
                              > self.config["model"]["threshold"]).float()
                fine_acc = (fine_preds == batch["fine_labels"]).float().mean()

            # Update metrics
            total_loss += outputs.loss.item()
            total_main_acc += main_acc.item()
            total_fine_acc += fine_acc.item()
            steps += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{outputs.loss.item():.4f}',
                'main_acc': f'{main_acc.item():.4f}',
                'fine_acc': f'{fine_acc.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        return PipelineOutput(
            loss=total_loss / steps,
            main_acc=total_main_acc / steps,
            fine_acc=total_fine_acc / steps
        )

    def _evaluate(
        self,
        eval_loader: TorchDataLoader,
        return_predictions: bool = False
    ) -> PipelineOutput:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_main_acc = 0
        total_fine_acc = 0
        steps = 0
        predictions = [] if return_predictions else None

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    entity_positions=batch["entity_position"],
                    labels=torch.cat(
                        [batch["main_labels"], batch["fine_labels"]], dim=1)
                    if "main_labels" in batch else None
                )

                # Calculate metrics if labels available
                if "main_labels" in batch:
                    total_loss += outputs.loss.item()

                    main_preds = torch.argmax(outputs.main_logits, dim=1)
                    main_acc = (
                        main_preds == batch["main_labels"]).float().mean()
                    total_main_acc += main_acc.item()

                    fine_preds = (torch.sigmoid(outputs.fine_logits)
                                  > self.config["model"]["threshold"]).float()
                    fine_acc = (
                        fine_preds == batch["fine_labels"]).float().mean()
                    total_fine_acc += fine_acc.item()

                    steps += 1

                # Collect predictions if requested
                if return_predictions:
                    batch_predictions = self._get_batch_predictions(
                        outputs,
                        batch,
                        self.config["model"]["threshold"]
                    )
                    predictions.extend(batch_predictions)

        if steps == 0:
            steps = 1  # Avoid division by zero when no labels available

        return PipelineOutput(
            loss=total_loss / steps,
            main_acc=total_main_acc / steps,
            fine_acc=total_fine_acc / steps,
            predictions=predictions
        )

    def _get_batch_predictions(
        self,
        outputs: Any,
        batch: Dict[str, Any],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Get predictions for a batch."""
        predictions = []

        # Get predictions from logits
        main_preds = torch.argmax(outputs.main_logits, dim=1)
        fine_probs = torch.sigmoid(outputs.fine_logits)

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
                if role in self.model.fine_role_indices:
                    # Adjust index to be relative to fine-grained space
                    fine_idx = self.model.fine_role_indices[role] - len(
                        main_role_indices)
                    valid_indices.append(fine_idx)
            valid_role_indices[main_idx] = sorted(valid_indices)

        # Print predictions for each entity in batch
        for i in range(len(main_preds)):
            # Get main role prediction
            main_pred_idx = main_preds[i].item()
            main_role = main_role_map[main_pred_idx]

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
                    fine_idx = valid_indices[top_local_indices[local_idx].item(
                    )]
                    # Convert back to role name using model's mapping
                    for role, idx in self.model.fine_role_indices.items():
                        if (idx - len(main_role_indices)) == fine_idx:
                            predicted_fine_roles.append(role)
                            break

            # Always take at least one role (highest probability)
            if not predicted_fine_roles:
                max_local_idx = valid_probs.argmax().item()
                fine_idx = valid_indices[max_local_idx]
                for role, idx in self.model.fine_role_indices.items():
                    if (idx - len(main_role_indices)) == fine_idx:
                        predicted_fine_roles.append(role)
                        break

            # Create prediction entry
            prediction = {
                "article_id": batch.get("article_ids", [""])[i],
                "entity_mention": batch.get("entity_mentions", [""])[i],
                "start_offset": batch.get("start_offsets", [0])[i],
                "end_offset": batch.get("end_offsets", [0])[i],
                "main_role": main_role,
                "fine_grained_roles": predicted_fine_roles
            }

            # Log predictions for debugging
            logger.debug(
                f"\nPrediction for {prediction['article_id']} - {prediction['entity_mention']}:")
            logger.debug(f"Main role: {main_role}")
            logger.debug(f"Fine roles: {predicted_fine_roles}")
            logger.debug("Fine-grained role probabilities:")
            for j, prob in enumerate(valid_probs):
                fine_idx = valid_indices[j]
                for role, idx in self.model.fine_role_indices.items():
                    if (idx - len(main_role_indices)) == fine_idx:
                        if prob > 0.3:  # Only show significant probabilities
                            logger.debug(f"{role}: {prob.item():.4f}")
                        break

            predictions.append(prediction)

        return predictions

    def train(
        self,
        languages: List[str] = ["EN", "PT"],
        max_articles: Optional[int] = None
    ) -> None:
        """Run training pipeline."""
        logger.info("Starting training pipeline")

        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{self.config['model']['name'].split('/')[-1]}_{timestamp}"
        logger.info(f"Starting training run: {run_id}")

        # Prepare datasets
        train_loader, _ = self.prepare_data(languages, "train", max_articles)
        val_loader, _ = self.prepare_data(languages, "dev", max_articles)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["model"]["learning_rate"]),
            weight_decay=float(self.config["model"]["weight_decay"])
        )

        num_training_steps = len(train_loader) * \
            self.config["model"]["num_epochs"]
        num_warmup_steps = int(num_training_steps *
                               self.config["model"]["warmup_ratio"])

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # Training loop
        best_val_metrics = {"loss": float('inf')}

        for epoch in range(self.config["model"]["num_epochs"]):
            # Training
            train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler, epoch)

            # Validation
            val_metrics = self._evaluate(val_loader)

            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Save best model
            if val_metrics.loss < best_val_metrics["loss"]:
                best_val_metrics = val_metrics
                self._save_checkpoint(
                    path=self._get_model_path(run_id, "best"),
                    epoch=epoch,
                    metrics=val_metrics,
                    run_id=run_id
                )

            # Regular checkpoint
            if (epoch + 1) % self.config["training"]["save_epochs"] == 0:
                self._save_checkpoint(
                    path=self._get_model_path(run_id, f"epoch_{epoch+1}"),
                    epoch=epoch,
                    metrics=val_metrics,
                    run_id=run_id
                )

        # Save final model
        self._save_checkpoint(
            path=self._get_model_path(run_id, "final"),
            epoch=self.config["model"]["num_epochs"] - 1,
            metrics=val_metrics,
            run_id=run_id
        )

        logger.info("Training completed!")
        logger.info(f"Best validation metrics: {best_val_metrics}")
        logger.info(f"Models saved with run ID: {run_id}")

    def evaluate(
        self,
        languages: List[str],
        output_dir: str,
        split: str = "test",
        max_articles: Optional[int] = None
    ) -> None:
        """
        Evaluate model and generate predictions.

        Args:
            languages: List of language codes to process
            output_dir: Directory to save predictions
            split: Data split to evaluate on ('dev' or 'test')
            max_articles: Maximum number of articles to evaluate on
        """
        logger.info(f"Starting evaluation on {split} set")

        # Generate unique run ID for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"eval_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        # Process each language separately
        for lang in languages:
            # Prepare data
            eval_loader, _ = self.prepare_data([lang], split, max_articles)

            # Get predictions
            results = self._evaluate(eval_loader, return_predictions=True)

            # Save predictions
            output_path = os.path.join(
                output_dir, f"predictions_{lang}_{run_id}.txt")
            self._save_predictions(results.predictions, output_path)

            logger.info(
                f"Saved {len(results.predictions)} predictions for {lang}")

            # Run scorer if in dev mode
            if split == "dev":
                self._run_scorer(
                    gold_file=os.path.join(
                        self.config["paths"]["dev_data_dir"],
                        lang,
                        "subtask-1-annotations.txt"
                    ),
                    pred_file=output_path,
                    lang=lang
                )

    def _get_model_path(self, run_id: str, checkpoint_type: str = "best") -> str:
        """Get path for model checkpoint."""
        model_dir = Path(self.config["paths"]["model_output_dir"])
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir / f"{checkpoint_type}_model_{run_id}.pt")

    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: PipelineOutput,
        run_id: str
    ) -> None:
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': {
                'loss': metrics.loss,
                'main_acc': metrics.main_acc,
                'fine_acc': metrics.fine_acc
            },
            'config': self.config,
            'run_id': run_id
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Initialize components if needed
        if self.model is None:
            self._initialize_components()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(
            f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: PipelineOutput,
        val_metrics: PipelineOutput
    ) -> None:
        """Log metrics for an epoch."""
        logger.info(
            f"Epoch {epoch+1}/{self.config['model']['num_epochs']} - "
            f"Train loss: {train_metrics.loss:.4f} - "
            f"Train main acc: {train_metrics.main_acc:.4f} - "
            f"Train fine acc: {train_metrics.fine_acc:.4f} - "
            f"Val loss: {val_metrics.loss:.4f} - "
            f"Val main acc: {val_metrics.main_acc:.4f} - "
            f"Val fine acc: {val_metrics.fine_acc:.4f}"
        )

    def _save_predictions(self, predictions: List[Dict], output_path: str) -> None:
        """Save predictions in the scorer-compatible format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(
                "article_id\tentity_mention\tstart_offset\tend_offset\tmain_role\tfine_grained_roles\n")

            # Write predictions
            for pred in predictions:
                line_parts = [
                    pred["article_id"],
                    pred["entity_mention"],
                    str(pred["start_offset"]),
                    str(pred["end_offset"]),
                    pred["main_role"],
                ]
                line_parts.extend(pred["fine_grained_roles"])

                line = "\t".join(line_parts)
                f.write(line + "\n")

    def _run_scorer(self, gold_file: str, pred_file: str, lang: str) -> None:
        """Run the official scorer on predictions."""
        try:
            from data.scorers.subtask1_scorer import (
                read_file,
                check_file_format,
                exact_match_ratio,
                evaluate_fine_grained_metrics,
                evaluate_main_role_accuracy
            )

            # Read files
            gold_dict = read_file(gold_file)
            pred_dict = read_file(pred_file)

            # Check format
            format_errors = check_file_format(gold_dict, pred_dict)
            if format_errors:
                logger.error(f"Format errors in {lang} predictions:")
                for error in format_errors:
                    logger.error(error)
                return

            # Calculate metrics
            emr = exact_match_ratio(gold_dict, pred_dict)
            micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall = evaluate_fine_grained_metrics(
                gold_dict, pred_dict)
            main_role_accuracy = evaluate_main_role_accuracy(
                gold_dict, pred_dict)

            # Log results
            logger.info(f"\nScorer results for {lang}:")
            logger.info(
                f"EMR\tMicro_Precision\tMicro_Recall\tMicro_F1\tMain_Role_Accuracy")
            logger.info(
                f"{emr:.4f}\t{micro_precision:.4f}\t{micro_recall:.4f}\t{micro_f1:.4f}\t{main_role_accuracy:.4f}")

            logger.info(f"\nAdditional metrics for {lang}:")
            logger.info(f"Macro F1: {macro_f1:.4f}")
            logger.info(f"Macro Precision: {macro_precision:.4f}")
            logger.info(f"Macro Recall: {macro_recall:.4f}")

        except Exception as e:
            logger.error(f"Error running scorer for {lang}: {str(e)}")
            raise
