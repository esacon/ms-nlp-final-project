import torch
from dataclasses import dataclass
from transformers import XLMRobertaModel
from typing import Dict, Optional
from src.utils import get_device, get_logger
from src.taxonomy import (
    get_main_roles, get_fine_roles,
    get_main_roles_count, get_fine_roles_count,
    get_role_indices, get_fine_role_indices,
    get_main_role_indices
)
logger = get_logger(__name__)


@dataclass
class ModelOutput:
    main_logits: torch.Tensor
    fine_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    main_loss: Optional[torch.Tensor] = None
    fine_loss: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


class EntityRoleClassifier(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.base_model = XLMRobertaModel.from_pretrained(
            config["model"]["name"])
        hidden_size = self.base_model.config.hidden_size

        self.num_main_roles = get_main_roles_count()
        self.num_fine_roles = get_fine_roles_count()

        # Role classifiers
        self.main_role_classifier = torch.nn.Linear(
            hidden_size, self.num_main_roles)
        self.fine_role_classifier = torch.nn.Linear(
            hidden_size, self.num_fine_roles)

        # Loss functions
        self.main_role_criterion = torch.nn.BCEWithLogitsLoss()
        self.fine_role_criterion = torch.nn.BCEWithLogitsLoss()

        # Load taxonomy mappings
        self.role_indices = get_role_indices()
        self.fine_role_indices = get_fine_role_indices()
        self.main_role_indices = get_main_role_indices()

        # Model parameters
        self.threshold = config["model"]["threshold"]
        self.l1_regularization = config["model"]["l1_regularization"]

        # Create valid role indices for each main role index
        self.valid_role_indices = {}
        for main_role in get_main_roles():
            main_idx = self.main_role_indices[main_role]
            valid_fine_roles = get_fine_roles(main_role)
            # Convert to indices in the fine-grained space only (0 to num_fine_roles-1)
            valid_indices = []
            for role in valid_fine_roles:
                if role in self.fine_role_indices:
                    # Adjust index to be relative to fine-grained space
                    fine_idx = self.fine_role_indices[role] - \
                        len(get_main_roles())
                    valid_indices.append(fine_idx)
            self.valid_role_indices[main_idx] = sorted(valid_indices)

    def forward(self, input_ids, attention_mask, entity_positions, labels=None):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Get entity embeddings
        batch_size = input_ids.shape[0]
        entity_embeddings = []
        for i in range(batch_size):
            start, end = entity_positions[i]
            span_embedding = hidden_states[i, start:end+1].mean(dim=0)
            entity_embeddings.append(span_embedding)
        entity_embeddings = torch.stack(entity_embeddings)

        # Get main role predictions using softmax
        main_role_logits = self.main_role_classifier(entity_embeddings)
        main_role_probs = torch.softmax(main_role_logits, dim=1)
        main_role_pred = torch.argmax(main_role_probs, dim=1)

        # Get fine role predictions
        fine_role_logits = self.fine_role_classifier(entity_embeddings)

        # Create masks for valid fine roles
        fine_role_mask = torch.zeros(
            (batch_size, self.num_fine_roles), device=self.device)
        for i in range(batch_size):
            main_role = main_role_pred[i].item()
            # valid_role_indices now contains indices in fine-grained space
            fine_role_mask[i, self.valid_role_indices[main_role]] = 1.0

        # Apply mask and get probabilities
        masked_fine_logits = fine_role_logits * fine_role_mask
        fine_role_probs = torch.sigmoid(masked_fine_logits)

        fine_role_pred = torch.zeros_like(fine_role_probs)
        for i in range(batch_size):
            valid_indices = self.valid_role_indices[main_role_pred[i].item()]
            valid_probs = fine_role_probs[i, valid_indices]

            # Get top 2 valid predictions above threshold
            k = min(2, len(valid_indices))
            top_values, top_local_indices = torch.topk(valid_probs, k=k)

            # Convert local indices to fine-grained space indices
            selected_indices = []
            for local_idx, value in enumerate(top_values):
                if value >= self.threshold:
                    fine_idx = valid_indices[top_local_indices[local_idx].item(
                    )]
                    selected_indices.append(fine_idx)

            # Always take at least one role (highest probability)
            if not selected_indices:
                max_local_idx = valid_probs.argmax().item()
                selected_indices = [valid_indices[max_local_idx]]

            fine_role_pred[i, selected_indices] = 1.0

        if labels is not None:
            main_role_labels = labels[:, :self.num_main_roles]
            fine_role_labels = labels[:, self.num_main_roles:]

            main_role_loss = self.main_role_criterion(
                main_role_logits, main_role_labels)
            fine_role_loss = self.fine_role_criterion(
                masked_fine_logits, fine_role_labels)

            # Add sparsity regularization
            l1_reg = torch.norm(fine_role_probs, p=1) * self.l1_regularization

            total_loss = main_role_loss + fine_role_loss + l1_reg

            return ModelOutput(
                main_logits=main_role_logits,
                fine_logits=masked_fine_logits,
                loss=total_loss,
                main_loss=main_role_loss,
                fine_loss=fine_role_loss
            )

        return ModelOutput(
            main_logits=main_role_logits,
            fine_logits=masked_fine_logits,
            loss=None,
            main_loss=None,
            fine_loss=None
        )


def load_model(config: Dict) -> EntityRoleClassifier:
    return EntityRoleClassifier(config)
