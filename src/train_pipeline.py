import os
import sys
from pathlib import Path
import torch
from datetime import datetime
import yaml
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from torch.optim import AdamW
from transformers import get_scheduler, XLMRobertaTokenizerFast, XLMRobertaModel

from src.utils import set_seed, get_logger, get_device
from src.data_loader import DataLoader
from src.feature_extraction import FeatureExtractor
from src.model import EntityRoleClassifier

logger = get_logger(__name__)

# Add these constants at the top after imports
MAIN_ROLES = ['Protagonist', 'Antagonist', 'Innocent']
FINE_GRAINED_ROLES = [
    # Protagonist roles
    'Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
    # Antagonist roles
    'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor',
    'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
    # Innocent roles
    'Forgotten', 'Exploited', 'Victim', 'Scapegoat'
]

class TrainingPipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        
        # Set random seeds
        set_seed(self.config["training"]["seed"])
        
        # Initialize components
        self.device = get_device()
        self.data_loader = DataLoader(self.config)
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        self.base_model = None
        
        # Role mappings
        self.main_role_map = {i: role for i, role in enumerate(MAIN_ROLES)}
        self.fine_role_map = {i: role for i, role in enumerate(FINE_GRAINED_ROLES)}
        
        logger.info(f"Pipeline initialized with device: {self.device}")
    
    def load_config(self) -> Dict:
        """Load configuration from yaml file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_components(self):
        """Initialize all model components."""
        logger.info("Initializing components...")
        
        # Initialize feature extractor
        logger.info("Loading base model and tokenizer...")
        self.feature_extractor = FeatureExtractor(
            max_length=self.config["model"]["max_length"],
            context_window=self.config["model"]["context_window"],
            batch_size=self.config["model"]["batch_size"],
            preprocessing_config=self.config["preprocessing"]
        )
        
        # Load tokenizer and base model
        base_model_name = self.config["model"]["name"]
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(base_model_name)
        self.base_model = XLMRobertaModel.from_pretrained(base_model_name)
        self.feature_extractor.set_tokenizer_and_model(self.tokenizer, self.base_model)
        
        # Initialize classifier model
        logger.info("Initializing classifier model...")
        self.model = EntityRoleClassifier(
            num_fine_labels=22,  # Hardcoded as per test_model.py
            config=self.config
        ).to(self.device)
    
    def prepare_data(self, languages: List[str], split: str, max_articles: int = None) -> Tuple[TorchDataLoader, List]:
        """
        Prepare data for training or evaluation.
        
        Args:
            languages: List of language codes to process
            split: Data split to use ('train', 'dev', or 'test')
            max_articles: Maximum number of articles to load (for testing)
        
        Returns:
            Tuple of (DataLoader, processed annotations)
        """
        logger.info(f"Preparing {split} data for languages: {languages}")
        
        all_articles = []
        for lang in languages:
            articles = self.data_loader.load_articles("data", lang, split)
            if max_articles:
                articles = articles[:max_articles]
                logger.info(f"Limited to {max_articles} articles for testing")
            all_articles.extend(articles)
            logger.info(f"Loaded {len(articles)} {split} articles for {lang}")
        
        # Extract features
        logger.info("\nExtracting features...")
        features, _ = self.data_loader.prepare_features(all_articles, self.feature_extractor)
        logger.info(f"Extracted features for {len(features)} entities")
        
        # Create labels
        main_labels = []
        fine_labels = []
        
        for article in all_articles:
            for ann in article.annotations:
                if ann.main_role and ann.fine_grained_roles:
                    # Convert main role to index
                    main_role_idx = MAIN_ROLES.index(ann.main_role)
                    main_labels.append(main_role_idx)
                    
                    # Convert fine-grained roles to multi-hot
                    fine_label = torch.zeros(len(FINE_GRAINED_ROLES))
                    for role in ann.fine_grained_roles:
                        if role in FINE_GRAINED_ROLES:
                            role_idx = FINE_GRAINED_ROLES.index(role)
                            fine_label[role_idx] = 1
                    fine_labels.append(fine_label)
        
        # Convert to tensors
        if main_labels and fine_labels:
            main_labels = torch.tensor(main_labels, dtype=torch.long)
            fine_labels = torch.stack(fine_labels)
            labels = {"main_labels": main_labels, "fine_labels": fine_labels}
        else:
            # For evaluation, create dummy labels
            labels = {
                "main_labels": torch.zeros(len(features), dtype=torch.long),
                "fine_labels": torch.zeros(len(features), len(FINE_GRAINED_ROLES))
            }
        
        # Create dataloader
        dataloader = self.data_loader.create_dataloader(
            features=features,
            labels=labels,
            batch_size=self.config["model"]["batch_size"],
            shuffle=(split == "train")
        )
        logger.info(f"Created DataLoader with {len(dataloader)} batches")
        
        return dataloader, all_articles
    
    def _generate_run_id(self) -> str:
        """Generate a unique run identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config["model"]["name"].split("/")[-1]  # Get base name of model
        langs = "_".join(self.config.get("languages", {}).get("supported", ["EN"]))
        return f"{model_name}_{langs}_{timestamp}"

    def _get_model_path(self, run_id: str, checkpoint_type: str = "best") -> str:
        """Get path for model checkpoint."""
        model_dir = Path(self.config["paths"]["model_output_dir"])
        model_dir.mkdir(exist_ok=True, parents=True)
        return str(model_dir / f"{checkpoint_type}_model_{run_id}.pt")

    def _get_predictions_path(self, output_dir: str, lang: str, run_id: str) -> str:
        """Get path for predictions file."""
        pred_dir = Path(output_dir)
        pred_dir.mkdir(exist_ok=True, parents=True)
        return str(pred_dir / f"predictions_{lang}_{run_id}.txt")

    def train(self, languages: List[str] = ["EN", "PT"], max_articles: int = None):
        """Run training pipeline."""
        logger.info("Starting training pipeline")
        
        # Generate unique run ID
        run_id = self._generate_run_id()
        logger.info(f"Starting training run: {run_id}")
        
        # Initialize components if not done
        if self.model is None:
            self.initialize_components()
        
        # Prepare datasets
        train_loader, train_articles = self.prepare_data(languages, "train", max_articles)
        val_loader, val_articles = self.prepare_data(languages, "dev", max_articles)
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["model"]["learning_rate"]),
            weight_decay=float(self.config["model"]["weight_decay"])
        )
        
        total_steps = len(train_loader) * self.config["model"]["num_epochs"]
        warmup_steps = int(total_steps * self.config["model"]["warmup_ratio"])
        
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_metrics = {"loss": float('inf')}
        scaler = torch.cuda.amp.GradScaler() if self.config["training"]["mixed_precision"] else None
        
        for epoch in range(self.config["model"]["num_epochs"]):
            # Training
            train_metrics = self._train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch
            )
            
            # Validation
            val_metrics = self._evaluate(val_loader)
            
            # Logging
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics["loss"] < best_val_metrics["loss"]:
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
    
    def _train_epoch(
        self, 
        train_loader: TorchDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_fine_loss = 0
        total_main_acc = 0
        total_fine_acc = 0
        steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            entity_positions = batch["entity_position"].to(self.device)
            main_labels = batch["main_labels"].to(self.device)
            fine_labels = batch["fine_labels"].to(self.device)
            embeddings = batch.get("embeddings", None)
            if embeddings is not None:
                embeddings = embeddings.to(self.device)
            
            # Forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        entity_positions=entity_positions,
                        embeddings=embeddings,
                        main_labels=main_labels,
                        fine_labels=fine_labels
                    )
                    loss = outputs.loss
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions,
                    embeddings=embeddings,
                    main_labels=main_labels,
                    fine_labels=fine_labels
                )
                loss = outputs.loss
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Calculate accuracies
            with torch.no_grad():
                main_preds = torch.argmax(outputs.main_logits, dim=1)
                main_acc = (main_preds == main_labels).float().mean().item()
                
                fine_preds = (torch.sigmoid(outputs.fine_logits) > 0.5).float()
                fine_acc = (fine_preds == fine_labels).float().mean().item()
            
            # Update metrics
            total_loss += loss.item()
            total_main_loss += outputs.main_loss.item()
            total_fine_loss += outputs.fine_loss.item()
            total_main_acc += main_acc
            total_fine_acc += fine_acc
            steps += 1
            
            # Update progress bar with detailed metrics
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'main_loss': f"{outputs.main_loss.item():.4f}",
                'fine_loss': f"{outputs.fine_loss.item():.4f}",
                'main_acc': f"{main_acc:.4f}",
                'fine_acc': f"{fine_acc:.4f}"
            })
            
            # Log detailed metrics every 100 steps
            if steps % 100 == 0:
                logger.info(
                    f"Epoch {epoch} - Step {steps}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Main Loss: {outputs.main_loss.item():.4f} | "
                    f"Fine Loss: {outputs.fine_loss.item():.4f} | "
                    f"Main Acc: {main_acc:.4f} | "
                    f"Fine Acc: {fine_acc:.4f}"
                )
        
        # Calculate average metrics
        avg_loss = total_loss / steps
        avg_main_loss = total_main_loss / steps
        avg_fine_loss = total_fine_loss / steps
        avg_main_acc = total_main_acc / steps
        avg_fine_acc = total_fine_acc / steps
        
        return {
            'loss': avg_loss,
            'main_loss': avg_main_loss,
            'fine_loss': avg_fine_loss,
            'main_acc': avg_main_acc,
            'fine_acc': avg_fine_acc
        }
    
    def _evaluate(self, val_loader: TorchDataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        total_main_acc = 0
        total_fine_acc = 0
        steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                embeddings = batch.get("embeddings", None)
                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                entity_positions = batch["entity_position"].to(self.device)
                main_labels = batch["main_labels"].to(self.device)
                fine_labels = batch["fine_labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions,
                    embeddings=embeddings,
                    main_labels=main_labels,
                    fine_labels=fine_labels
                )
                
                loss, main_acc, fine_acc = self._compute_loss_and_metrics(
                    outputs, main_labels, fine_labels
                )
                
                total_loss += loss.item()
                total_main_acc += main_acc
                total_fine_acc += fine_acc
                steps += 1
        
        return {
            "loss": total_loss / steps,
            "main_acc": total_main_acc / steps,
            "fine_acc": total_fine_acc / steps
        }
    
    def _compute_loss_and_metrics(
        self,
        outputs,
        main_labels: torch.Tensor,
        fine_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute loss and accuracy metrics."""
        main_logits = outputs.main_logits if hasattr(outputs, 'main_logits') else outputs[0]
        fine_logits = outputs.fine_logits if hasattr(outputs, 'fine_logits') else outputs[1]
        
        # Calculate losses
        main_loss = torch.nn.functional.cross_entropy(main_logits, main_labels)
        fine_loss = torch.nn.functional.binary_cross_entropy_with_logits(fine_logits, fine_labels)
        loss = main_loss + fine_loss
        
        # Calculate accuracies
        main_preds = torch.argmax(main_logits, dim=1)
        fine_preds = (torch.sigmoid(fine_logits) > 0.5).float()
        
        main_acc = (main_preds == main_labels).float().mean().item()
        fine_acc = (fine_preds == fine_labels).float().mean().item()
        
        return loss, main_acc, fine_acc
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics for an epoch."""
        logger.info(
            f"Epoch {epoch+1}/{self.config['model']['num_epochs']} - "
            f"Train loss: {train_metrics['loss']:.4f} - "
            f"Train main loss: {train_metrics['main_loss']:.4f} - "
            f"Train fine loss: {train_metrics['fine_loss']:.4f} - "
            f"Train main acc: {train_metrics['main_acc']:.4f} - "
            f"Train fine acc: {train_metrics['fine_acc']:.4f} - "
            f"Val loss: {val_metrics['loss']:.4f} - "
            f"Val main acc: {val_metrics['main_acc']:.4f} - "
            f"Val fine acc: {val_metrics['fine_acc']:.4f}"
        )
    
    def _save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
        run_id: str
    ) -> None:
        """Save a model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
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
            self.initialize_components()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    def evaluate(self, languages: List[str], output_dir: str, split: str = "test", max_articles: int = None) -> None:
        """Run evaluation on the specified languages and split."""
        logger.info(f"Starting evaluation on {split} split for languages: {languages}")
        
        # Load data
        eval_loader, eval_articles = self.prepare_data(languages, split, max_articles)
        
        # Set model to evaluation mode
        self.model.eval()
        predictions = []
        entity_idx = 0  # Keep track of which entity we're processing
        
        # Create a flat list of all annotations for easier indexing
        all_annotations = []
        for article in eval_articles:
            for ann in article.annotations:
                all_annotations.append((article, ann))
        
        # Class weights to balance predictions
        main_role_weights = torch.tensor([1.2, 1.2, 0.6]).to(self.device)  # Reduce weight for Innocent
        temperature = 1.5  # Temperature for softmax to make distributions sharper
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                entity_positions = batch["entity_position"].to(self.device)
                embeddings = batch.get("embeddings", None)
                if embeddings is not None:
                    embeddings = embeddings.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions,
                    embeddings=embeddings
                )
                
                # Apply class weights and temperature scaling
                weighted_logits = outputs.main_logits * main_role_weights.unsqueeze(0)
                main_probs = torch.softmax(weighted_logits / temperature, dim=1)
                fine_probs = torch.sigmoid(outputs.fine_logits / temperature)
                
                # Process each prediction in the batch
                for i in range(main_probs.size(0)):
                    if entity_idx >= len(all_annotations):
                        continue
                        
                    # Get article and entity information
                    article, entity = all_annotations[entity_idx]
                    
                    # Get main role prediction with dynamic thresholding
                    main_probs_i = main_probs[i]
                    max_prob, main_role_idx = torch.max(main_probs_i, dim=0)
                    
                    # If the confidence is too low, look at second highest probability
                    if max_prob < 0.5:  # Confidence threshold
                        sorted_probs, sorted_indices = torch.sort(main_probs_i, descending=True)
                        if sorted_probs[1] > 0.3:  # Check second highest probability
                            main_role_idx = sorted_indices[1]
                            max_prob = sorted_probs[1]
                    
                    main_role = MAIN_ROLES[main_role_idx]
                    main_confidence = max_prob.item()
                    
                    # Get valid fine-grained roles based on main role
                    if main_role == "Protagonist":
                        valid_roles = set(FINE_GRAINED_ROLES[:6])
                        max_roles = 3
                        threshold = 0.4  # Higher threshold for Protagonist
                    elif main_role == "Antagonist":
                        valid_roles = set(FINE_GRAINED_ROLES[6:18])
                        max_roles = 3
                        threshold = 0.35  # Medium threshold for Antagonist
                    else:  # Innocent
                        valid_roles = set(FINE_GRAINED_ROLES[18:22])
                        max_roles = 4
                        threshold = 0.45  # Higher threshold for Innocent to prevent default predictions
                    
                    # Get fine-grained role predictions with dynamic thresholding
                    fine_roles = []
                    role_confidences = []
                    
                    # Get all valid roles with their probabilities
                    valid_probs = []
                    for j, role in enumerate(FINE_GRAINED_ROLES):
                        if role in valid_roles:
                            prob = fine_probs[i, j].item()
                            valid_probs.append((prob, role))
                    
                    # Sort by probability
                    valid_probs.sort(reverse=True)
                    
                    # Dynamic thresholding based on probability distribution
                    if valid_probs:
                        max_prob = valid_probs[0][0]
                        dynamic_threshold = max(threshold, max_prob * 0.6)  # At least 60% of max probability
                        
                        # Take roles above dynamic threshold
                        for prob, role in valid_probs:
                            if prob > dynamic_threshold and len(fine_roles) < max_roles:
                                fine_roles.append(role)
                                role_confidences.append(prob)
                    
                    # Ensure at least one role is predicted, but with stricter criteria
                    if not fine_roles and valid_probs:
                        top_prob, top_role = valid_probs[0]
                        if top_prob > threshold * 0.8:  # Only take top role if it's reasonably confident
                            fine_roles = [top_role]
                            role_confidences = [top_prob]
                        else:
                            # If no confident predictions, take top 2 roles with highest relative probability
                            relative_probs = [(p/sum(p for p, _ in valid_probs[:2]), r) for p, r in valid_probs[:2]]
                            fine_roles = [r for _, r in relative_probs]
                            role_confidences = [p for p, _ in relative_probs]
                    
                    # Create prediction entry
                    prediction = {
                        "article_id": article.id,
                        "entity_mention": entity.entity_mention,
                        "start_offset": entity.start_offset,
                        "end_offset": entity.end_offset,
                        "main_role": main_role,
                        "main_confidence": main_confidence,
                        "fine_roles": fine_roles,
                        "fine_confidences": role_confidences
                    }
                    predictions.append(prediction)
                    entity_idx += 1
                    
                    # Log prediction details for debugging
                    logger.debug(
                        f"\nPrediction for {article.id} - {entity.entity_mention}:\n"
                        f"Main role probs: {dict(zip(MAIN_ROLES, main_probs_i.tolist()))}\n"
                        f"Selected main role: {main_role} ({main_confidence:.4f})\n"
                        f"Fine role probs: {valid_probs}\n"
                        f"Selected fine roles: {list(zip(fine_roles, role_confidences))}"
                    )
        
        # Save predictions for each language
        for lang in languages:
            output_path = self._get_predictions_path(output_dir, lang, self._generate_run_id())
            self._save_predictions(predictions, output_path)
            logger.info(f"Saved predictions for {lang} to {output_path}")
            
            # Run scorer if gold file exists
            gold_file = f"data/{split}/{lang}/subtask-1-annotations.txt"
            if os.path.exists(gold_file):
                self._run_scorer(gold_file, output_path, lang)
    
    def _save_predictions(self, predictions: List[Dict], output_path: str) -> None:
        """Save predictions to file in the required format."""
        logger.info(f"Saving {len(predictions)} predictions to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                # Format prediction line
                line_parts = [
                    pred["article_id"],
                    pred["entity_mention"],
                    str(pred["start_offset"]),
                    str(pred["end_offset"]),
                    pred["main_role"]
                ]
                
                # Add fine-grained roles
                line_parts.extend(pred["fine_roles"])
                
                # Write line
                f.write('\t'.join(line_parts) + '\n')
                
                # Log prediction details with confidence scores
                logger.debug(
                    f"\nPrediction for {pred['article_id']} - {pred['entity_mention']}:\n"
                    f"Main role: {pred['main_role']} (confidence: {pred['main_confidence']:.4f})\n"
                    f"Fine roles: {list(zip(pred['fine_roles'], pred['fine_confidences']))}"
                )
    
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
            micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall = evaluate_fine_grained_metrics(gold_dict, pred_dict)
            main_role_accuracy = evaluate_main_role_accuracy(gold_dict, pred_dict)
            
            # Log results in the same format as the scorer
            logger.info(f"\nScorer results for {lang}:")
            logger.info(f"EMR\tMicro_Precision\tMicro_Recall\tMicro_F1\tMain_Role_Accuracy")
            logger.info(f"{emr:.4f}\t{micro_precision:.4f}\t{micro_recall:.4f}\t{micro_f1:.4f}\t{main_role_accuracy:.4f}")
            
            # Also log additional metrics
            logger.info(f"\nAdditional metrics for {lang}:")
            logger.info(f"Macro F1: {macro_f1:.4f}")
            logger.info(f"Macro Precision: {macro_precision:.4f}")
            logger.info(f"Macro Recall: {macro_recall:.4f}")
            
        except Exception as e:
            logger.error(f"Error running scorer for {lang}: {str(e)}")
            raise

def run_training(
    config_path: str = "src/configs.yaml",
    languages: List[str] = ["EN", "PT"],
    max_articles: int = None
) -> None:
    """Entry point for training pipeline."""
    try:
        pipeline = TrainingPipeline(config_path)
        pipeline.train(languages=languages, max_articles=max_articles)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

def run_evaluation(
    config_path: str = "src/configs.yaml",
    model_path: str = "models/best_model.pt",
    languages: List[str] = ["EN", "PT"],
    output_dir: str = "predictions",
    split: str = "test",
    max_articles: int = None
) -> None:
    """Entry point for evaluation pipeline."""
    try:
        pipeline = TrainingPipeline(config_path)
        pipeline.load_checkpoint(model_path)
        pipeline.evaluate(languages=languages, output_dir=output_dir, split=split, max_articles=max_articles)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise 