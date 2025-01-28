import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from typing import Tuple, Dict, Optional
from src.utils import get_device

@dataclass
class ModelOutput:
    main_logits: torch.Tensor
    fine_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    main_loss: Optional[torch.Tensor] = None
    fine_loss: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None

class EntityRoleClassifier(torch.nn.Module):
    def __init__(self, num_fine_labels: int, config: Dict):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.num_fine_labels = num_fine_labels

        # Initialize model and tokenizer
        model_name = config["model"]["name"]
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
        self.roberta = XLMRobertaModel.from_pretrained(
            model_name,
            hidden_dropout_prob=config["model"]["dropout_prob"],
            attention_probs_dropout_prob=config["model"]["attention_dropout"]
        )

        # Add special entity marker tokens
        special_tokens = ['[ENT]', '[/ENT]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.roberta.resize_token_embeddings(len(self.tokenizer))

        hidden_size = self.roberta.config.hidden_size
        combined_size = hidden_size * 2

        # Entity-aware attention layer
        self.entity_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,  # Increased for larger model
            dropout=config["model"]["attention_dropout"],
            batch_first=True
        )

        # Main role classifier (3 classes)
        self.main_classifier = torch.nn.Sequential(
            torch.nn.Linear(combined_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 3)
        )

        # Fine-grained role classifier (multi-label)
        self.fine_classifier = torch.nn.Sequential(
            torch.nn.Linear(combined_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, num_fine_labels)
        )

        # Loss weights
        self.main_loss_weight = 1.0
        self.fine_loss_weight = 1.0

    def get_entity_representation(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get enhanced entity representation using attention mechanism.
        
        Args:
            sequence_output: Transformer output of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            entity_positions: Entity positions of shape (batch_size, 1, 2)
            
        Returns:
            Tuple of (entity_embeddings, attention_weights)
            Note: attention_weights might be None due to variable entity lengths
        """
        batch_size = sequence_output.size(0)
        entity_embeddings = []
        attention_weights_list = []

        # Ensure entity_positions has the right shape
        if entity_positions.dim() != 3:
            raise ValueError(f"Expected entity_positions to have 3 dimensions, got {entity_positions.dim()}")

        for i in range(batch_size):
            # Get start and end positions
            start = entity_positions[i, 0, 0].item()
            end = entity_positions[i, 0, 1].item()
            
            # Get entity span
            entity_span = sequence_output[i, start:end+1]  # shape: (entity_len, hidden_size)
            
            # Create attention mask for entity tokens
            # Create 2D mask: (1, seq_length) as required by MultiheadAttention
            key_padding_mask = ~attention_mask[i].bool().unsqueeze(0)
            
            # Apply self-attention to get weighted entity representation
            entity_attn_out, attn_weights = self.entity_attention(
                entity_span.unsqueeze(0),  # shape: (1, entity_len, hidden_size)
                sequence_output[i].unsqueeze(0),  # shape: (1, seq_len, hidden_size)
                sequence_output[i].unsqueeze(0),  # shape: (1, seq_len, hidden_size)
                key_padding_mask=key_padding_mask  # shape: (1, seq_length)
            )
            
            # Pool the attention output
            entity_embedding = torch.mean(entity_attn_out.squeeze(0), dim=0)
            entity_embeddings.append(entity_embedding)
            attention_weights_list.append(attn_weights)

        entity_embeddings = torch.stack(entity_embeddings)
        
        # Don't try to stack attention weights since they have different sizes
        # Instead, return them as a list or None
        attention_weights = None  # or attention_weights_list if you need them
        
        return entity_embeddings, attention_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        main_labels: Optional[torch.Tensor] = None,
        fine_labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass of the model.

        Args:
            input_ids: Token IDs of shape (batch_size, 1, seq_length)
            attention_mask: Attention mask of shape (batch_size, 1, seq_length)
            entity_positions: Entity positions of shape (batch_size, 1, 2)
            embeddings: Pre-computed embeddings of shape (batch_size, 1, seq_length, hidden_size)
            main_labels: Optional main role labels of shape (batch_size,)
            fine_labels: Optional fine-grained role labels of shape (batch_size, num_fine_labels)

        Returns:
            ModelOutput containing logits and optional losses
        """
        batch_size = input_ids.size(0)
        
        # Reshape inputs to remove sequence dimension if needed
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(1)
        if embeddings is not None and embeddings.dim() == 4:
            embeddings = embeddings.squeeze(1)
        
        # Use pre-computed embeddings if provided, otherwise compute them
        if embeddings is not None:
            sequence_output = embeddings
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            sequence_output = outputs.last_hidden_state

        # Get enhanced entity representation
        entity_embeddings, _ = self.get_entity_representation(  # Ignore attention weights
            sequence_output, attention_mask, entity_positions
        )

        # Get CLS token representation for context
        context_embeddings = sequence_output[:, 0]

        # Combine entity and context representations
        combined_embedding = torch.cat([entity_embeddings, context_embeddings], dim=1)

        # Get predictions
        main_logits = self.main_classifier(combined_embedding)
        fine_logits = self.fine_classifier(combined_embedding)

        # Calculate losses if labels are provided
        loss = None
        main_loss = None
        fine_loss = None

        if main_labels is not None and fine_labels is not None:
            # Main role loss (cross entropy)
            main_loss = F.cross_entropy(
                main_logits,
                main_labels,
                label_smoothing=0.1
            )

            # Fine-grained role loss (binary cross entropy)
            fine_loss = F.binary_cross_entropy_with_logits(
                fine_logits,
                fine_labels,
                reduction='mean',
                pos_weight=self._get_pos_weights(fine_labels)
            )

            # Combine losses with weights
            loss = (self.main_loss_weight * main_loss +
                   self.fine_loss_weight * fine_loss)

        return ModelOutput(
            main_logits=main_logits,
            fine_logits=fine_logits,
            loss=loss,
            main_loss=main_loss,
            fine_loss=fine_loss,
            attention_weights=None  # Don't return attention weights
        )

    def _get_pos_weights(self, fine_labels: torch.Tensor) -> torch.Tensor:
        """Calculate positive weights for BCE loss to handle class imbalance."""
        pos_counts = torch.sum(fine_labels, dim=0)
        neg_counts = fine_labels.size(0) - pos_counts
        pos_weights = neg_counts / torch.clamp(pos_counts, min=1)
        return pos_weights.to(self.device)

    def update_loss_weights(self, main_loss: float, fine_loss: float):
        """Dynamically update loss weights based on current losses."""
        total = main_loss + fine_loss
        self.main_loss_weight = fine_loss / total
        self.fine_loss_weight = main_loss / total

    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """Single training step."""
        self.train()
        optimizer.zero_grad()

        # Unpack batch
        input_ids, attention_mask, entity_positions, embeddings, main_labels, fine_labels = [
            b.to(self.device) for b in batch
        ]

        # Forward pass
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_positions=entity_positions,
            embeddings=embeddings,
            main_labels=main_labels,
            fine_labels=fine_labels
        )

        # Backward pass
        outputs.loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Update loss weights
        self.update_loss_weights(
            outputs.main_loss.item(),
            outputs.fine_loss.item()
        )

        return {
            'loss': outputs.loss.item(),
            'main_loss': outputs.main_loss.item(),
            'fine_loss': outputs.fine_loss.item()
        }

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.eval()
        total_loss = 0
        main_correct = 0
        fine_correct = 0
        total_samples = 0

        for batch in val_loader:
            input_ids, attention_mask, entity_positions, embeddings, main_labels, fine_labels = [
                b.to(self.device) for b in batch
            ]

            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_positions=entity_positions,
                embeddings=embeddings,
                main_labels=main_labels,
                fine_labels=fine_labels
            )

            total_loss += outputs.loss.item()
            
            # Calculate accuracies
            main_preds = torch.argmax(outputs.main_logits, dim=1)
            main_correct += (main_preds == main_labels).sum().item()
            
            fine_preds = (outputs.fine_logits > 0).float()
            fine_correct += torch.all(fine_preds == fine_labels, dim=1).sum().item()
            
            total_samples += input_ids.size(0)

        return {
            'val_loss': total_loss / len(val_loader),
            'main_acc': main_correct / total_samples,
            'fine_acc': fine_correct / total_samples
        }

    def save_pretrained(self, path: str):
        """Save model, tokenizer, and additional components."""
        os.makedirs(path, exist_ok=True)
        
        # Save base model and tokenizer
        self.roberta.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save classifiers and attention layer
        torch.save({
            'main_classifier': self.main_classifier.state_dict(),
            'fine_classifier': self.fine_classifier.state_dict(),
            'entity_attention': self.entity_attention.state_dict(),
            'config': self.config,
            'num_fine_labels': self.num_fine_labels,
            'main_loss_weight': self.main_loss_weight,
            'fine_loss_weight': self.fine_loss_weight
        }, os.path.join(path, 'model_components.pt'))

    @classmethod
    def from_pretrained(cls, path: str, config: Dict, num_fine_labels: int):
        """Load model from path."""
        instance = cls(num_fine_labels=num_fine_labels, config=config)
        
        # Load base model and tokenizer
        instance.roberta = XLMRobertaModel.from_pretrained(path)
        instance.tokenizer = XLMRobertaTokenizerFast.from_pretrained(path)
        
        # Load additional components
        components = torch.load(os.path.join(path, 'model_components.pt'))
        instance.main_classifier.load_state_dict(components['main_classifier'])
        instance.fine_classifier.load_state_dict(components['fine_classifier'])
        instance.entity_attention.load_state_dict(components['entity_attention'])
        instance.main_loss_weight = components['main_loss_weight']
        instance.fine_loss_weight = components['fine_loss_weight']
        
        return instance


def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    return model
