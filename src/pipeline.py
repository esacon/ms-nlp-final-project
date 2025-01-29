import torch
from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import logging
from typing import Dict, List
from src.model import HierarchicalRoleClassifier
from src.data_loader import DataLoader
from src.taxonomy import load_taxonomy
from src.feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Dict):
        """
        Initialize the pipeline with the configuration dictionary.
        Args:
            config (Dict): Configuration dictionary containing paths, model settings, etc.
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and taxonomy
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            config["model"]["name"])
        self.taxonomy = load_taxonomy(config["paths"]["taxonomy_path"])

        # Initialize hierarchical model
        self.model = HierarchicalRoleClassifier(
            model_name=config["model"]["name"],
            main_role_classes=len(self.taxonomy.keys()),
            fine_role_classes=sum(len(subroles)
                                  for subroles in self.taxonomy.values())
        ).to(self.device)

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            max_length=config["model"]["max_length"],
            context_window=config["model"]["context_window"],
            batch_size=config["model"]["batch_size"]
        )
        self.feature_extractor.set_tokenizer_and_model(self.tokenizer, self.model.base_model)

        logger.info("Pipeline initialized.")

    def train(self, train_articles: List, val_articles: List):
        """
        Train the hierarchical model on training data.
        Args:
            train_articles (List): Training dataset articles.
            val_articles (List): Validation dataset articles.
        """
        # Create data loader instance for feature extraction
        data_loader = DataLoader(self.config)

        # Process articles into features and labels
        logger.info("Preparing training data...")
        if not train_articles:
            raise ValueError("No training articles provided")
        
        train_features, train_labels = data_loader.prepare_features(train_articles, self.feature_extractor)
        if not train_features:
            raise ValueError("No features extracted from training articles")
            
        train_dataset = data_loader.create_dataset(train_features, train_labels)
        logger.info(f"Created training dataset with {len(train_dataset)} examples")

        logger.info("Preparing validation data...")
        if not val_articles:
            raise ValueError("No validation articles provided")
            
        val_features, val_labels = data_loader.prepare_features(val_articles, self.feature_extractor)
        if not val_features:
            raise ValueError("No features extracted from validation articles")
            
        val_dataset = data_loader.create_dataset(val_features, val_labels)
        logger.info(f"Created validation dataset with {len(val_dataset)} examples")

        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.config["model"]["batch_size"],
            shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.config["model"]["batch_size"],
            shuffle=False
        )

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=float(
            self.config["model"]["learning_rate"]))
        num_training_steps = len(train_loader) * \
            self.config["model"]["num_epochs"]
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # Loss functions
        main_loss_fn = torch.nn.CrossEntropyLoss()
        fine_loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.config["model"]["num_epochs"]):
            self.model.train()
            total_main_loss, total_fine_loss = 0, 0

            for batch in train_loader:
                optimizer.zero_grad()

                # Prepare inputs
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                main_labels = batch["main_labels"].to(self.device)
                fine_labels = batch["fine_labels"].to(self.device)

                # Forward pass
                main_logits, fine_logits = self.model(
                    input_ids, attention_mask)

                # Compute losses
                main_loss = main_loss_fn(main_logits, main_labels)
                fine_loss = fine_loss_fn(fine_logits, fine_labels)
                total_loss = main_loss + fine_loss

                # Backward pass
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                total_main_loss += main_loss.item()
                total_fine_loss += fine_loss.item()

            logger.info(
                f"Epoch {epoch + 1}/{self.config['model']['num_epochs']}, Main Loss: {total_main_loss:.4f}, Fine Loss: {total_fine_loss:.4f}")

    def evaluate(self, eval_articles: List):
        """
        Evaluate the model on the validation/test dataset.
        Args:
            eval_articles (List): List of articles for evaluation.
        Returns:
            Metrics as a dictionary.
        """
        # Process articles into features
        data_loader = DataLoader(self.config)
        eval_features, eval_labels = data_loader.prepare_features(eval_articles, self.feature_extractor)
        eval_dataset = data_loader.create_dataset(eval_features, eval_labels)
        eval_loader = TorchDataLoader(
            eval_dataset,
            batch_size=self.config["model"]["batch_size"],
            shuffle=False
        )

        self.model.eval()
        y_true_main, y_pred_main = [], []
        y_true_fine, y_pred_fine = [], []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                main_labels = batch["main_labels"].to(self.device)
                fine_labels = batch["fine_labels"].to(self.device)

                # Forward pass
                main_logits, fine_logits = self.model(input_ids, attention_mask)

                # Predict main roles
                main_preds = torch.argmax(main_logits, dim=1).cpu().tolist()
                y_pred_main.extend(main_preds)
                y_true_main.extend(main_labels.cpu().tolist())

                # Predict fine roles with constraints
                fine_probs = torch.sigmoid(fine_logits).cpu().tolist()
                for i, main_pred in enumerate(main_preds):
                    allowed_indices = self.taxonomy[main_pred]
                    y_pred_fine.append([
                        1 if j in allowed_indices and fine_probs[i][j] > 0.5 else 0
                        for j in range(fine_logits.size(1))
                    ])
                    y_true_fine.append(fine_labels[i].cpu().tolist())

        # Calculate metrics (exact match, precision, recall, F1, etc.)
        # This can be implemented using sklearn or similar libraries.
        metrics = {
            "main_accuracy": (sum(y1 == y2 for y1, y2 in zip(y_true_main, y_pred_main)) / len(y_true_main)),
            # Additional metrics like precision/recall for fine roles can be added here.
        }

        logger.info(f"Evaluation Metrics: {metrics}")
        return metrics
