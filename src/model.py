import os
import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from torch.optim import AdamW
from transformers import get_scheduler
from typing import Tuple, Dict
from src.utils import get_device


class EntityRoleClassifier(torch.nn.Module):
    def __init__(self, num_fine_labels: int, config: Dict):
        super().__init__()
        self.config = config
        self.device = get_device()

        # Initialize XLM-RoBERTa model and tokenizer
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            config["model"]["name"])
        self.roberta = XLMRobertaModel.from_pretrained(
            config["model"]["name"],
            hidden_dropout_prob=config["model"]["dropout_prob"],
            attention_probs_dropout_prob=config["model"]["attention_dropout"]
        )

        # Add special tokens for entity markers
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<e>', '</e>']})
        self.roberta.resize_token_embeddings(len(self.tokenizer))

        hidden_size = self.roberta.config.hidden_size

        # Main role classifier (3 classes: Protagonist, Antagonist, Other)
        self.main_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 3)
        )

        # Fine-grained role classifier (multi-label)
        self.fine_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_fine_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                entity_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Tensor of shape (batch_size, seq_length)
            entity_positions: Tensor of shape (batch_size, 2) containing start/end positions

        Returns:
            Tuple of main role logits and fine-grained role logits
        """
        # Get contextualized embeddings
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        # (batch_size, seq_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Extract entity span representations
        batch_size = input_ids.size(0)
        entity_embeddings = []

        for i in range(batch_size):
            start, end = entity_positions[i]
            # Average pooling over entity tokens
            entity_span = sequence_output[i, start:end+1]
            entity_embedding = torch.mean(entity_span, dim=0)
            entity_embeddings.append(entity_embedding)

        entity_embeddings = torch.stack(entity_embeddings)

        # Get CLS token representation for context
        context_embeddings = sequence_output[:, 0]

        # Combine entity and context representations
        combined_embedding = torch.cat(
            [entity_embeddings, context_embeddings], dim=1)

        # Get predictions
        main_logits = self.main_classifier(combined_embedding)
        fine_logits = self.fine_classifier(combined_embedding)

        return main_logits, fine_logits

    def train_model(self, train_loader, val_loader):
        """
        Train the model using the provided data loaders.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        optimizer = AdamW(self.parameters(), lr=float(
            self.config["learning_rate"]))
        num_training_steps = len(train_loader) * self.config["num_epochs"]
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps,
        )

        main_loss_fn = torch.nn.CrossEntropyLoss()
        fine_loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(self.config["num_epochs"]):
            self.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids, attention_mask, embeddings, entity_positions, main_labels, fine_labels = [
                    b.to(self.device) for b in batch
                ]

                main_logits, fine_logits = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_positions=entity_positions,
                )

                main_loss = main_loss_fn(main_logits, main_labels)
                fine_loss = fine_loss_fn(fine_logits, fine_labels)
                loss = main_loss + fine_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(
                        f"Epoch {epoch+1}, Batch {batch_idx}, Average Loss: {avg_loss:.4f}")

            # Validation step
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids, attention_mask, embeddings, entity_positions, main_labels, fine_labels = [
                        b.to(self.device) for b in val_batch
                    ]

                    main_logits, fine_logits = self(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        entity_positions=entity_positions,
                    )

                    main_loss = main_loss_fn(main_logits, main_labels)
                    fine_loss = fine_loss_fn(fine_logits, fine_labels)
                    val_loss += (main_loss + fine_loss).item()

            avg_val_loss = val_loss / len(val_loader)
            print(
                f"Epoch {epoch+1} completed. Average validation loss: {avg_val_loss:.4f}")

    def save_pretrained(self, path: str):
        """Save model and tokenizer to path"""
        self.roberta.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save the classifiers
        torch.save({
            'main_classifier': self.main_classifier.state_dict(),
            'fine_classifier': self.fine_classifier.state_dict(),
            'config': self.config
        }, f"{path}/classifiers.pt")

    @classmethod
    def from_pretrained(cls, path: str, config: Dict, num_fine_labels: int):
        """Load model from path"""
        instance = cls(num_fine_labels=num_fine_labels, config=config)
        # Load base model and tokenizer
        instance.roberta = XLMRobertaModel.from_pretrained(path)
        instance.tokenizer = XLMRobertaTokenizerFast.from_pretrained(path)
        # Load classifiers
        classifiers = torch.load(f"{path}/classifiers.pt")
        instance.main_classifier.load_state_dict(
            classifiers['main_classifier'])
        instance.fine_classifier.load_state_dict(
            classifiers['fine_classifier'])
        return instance


def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    return model
