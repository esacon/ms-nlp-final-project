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
        
        # Base model for contextual embeddings
        self.base_model = XLMRobertaModel.from_pretrained(
            config["model"]["name"])
        hidden_size = self.base_model.config.hidden_size
        
        # Get role counts
        self.num_main_roles = get_main_roles_count()
        self.num_fine_roles = get_fine_roles_count()
        
        # Main role classifier (first layer)
        self.main_role_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["model"].get("dropout", 0.1)),
            torch.nn.Linear(hidden_size // 2, self.num_main_roles)
        )
        
        # Fine-grained role classifier (second layer)
        # Takes concatenated [entity_embedding, main_role_embedding]
        self.fine_role_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + self.num_main_roles, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(config["model"].get("dropout", 0.1)),
            torch.nn.Linear(hidden_size, self.num_fine_roles)
        )
        
        # Loss functions
        self.main_role_criterion = torch.nn.CrossEntropyLoss()
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
        main_roles = get_main_roles()
        for main_role in main_roles:
            main_idx = self.main_role_indices[main_role]
            valid_fine_roles = get_fine_roles(main_role)
            valid_indices = []
            for role in valid_fine_roles:
                if role in self.fine_role_indices:
                    fine_idx = self.fine_role_indices[role] - len(main_roles)
                    if 0 <= fine_idx < self.num_fine_roles:
                        valid_indices.append(fine_idx)
                    else:
                        logger.warning(f"Invalid fine role index {fine_idx} for role {role}")
            self.valid_role_indices[main_idx] = sorted(valid_indices)

    def forward(self, input_ids, attention_mask, entity_positions, labels=None):
        # Get contextual embeddings
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
        
        # First layer: Main role prediction
        main_role_logits = self.main_role_layer(entity_embeddings)
        main_role_probs = torch.softmax(main_role_logits, dim=1)
        
        # Second layer: Fine-grained role prediction
        # Use probabilities of all main roles
        # This allows the model to learn the hierarchy through backpropagation
        combined_features = torch.cat([entity_embeddings, main_role_probs], dim=1)
        fine_role_logits = self.fine_role_layer(combined_features)
        
        # Create hierarchical mask for each main role
        hierarchical_mask = torch.zeros(
            (batch_size, self.num_main_roles, self.num_fine_roles), 
            device=self.device
        )
        for main_idx, valid_indices in self.valid_role_indices.items():
            hierarchical_mask[:, main_idx, valid_indices] = 1.0
        
        # Apply hierarchical masking using main role probabilities
        # Shape: [batch_size, num_fine_roles]
        masked_fine_logits = torch.zeros(
            (batch_size, self.num_fine_roles), 
            device=self.device
        )
        
        # For each main role, apply its mask weighted by its probability
        for main_idx in range(self.num_main_roles):
            main_probs = main_role_probs[:, main_idx].unsqueeze(1)  # [batch_size, 1]
            role_mask = hierarchical_mask[:, main_idx, :]  # [batch_size, num_fine_roles]
            masked_fine_logits += fine_role_logits * role_mask * main_probs
        
        if labels is not None:
            main_labels = labels["main_labels"]
            fine_labels = labels["fine_labels"]
            
            # Main role loss
            main_loss = self.main_role_criterion(main_role_logits, main_labels)
            
            # Fine role loss - only on valid roles for the true main role
            valid_mask = torch.zeros_like(fine_labels)
            for i in range(batch_size):
                main_idx = main_labels[i].item()
                valid_mask[i, self.valid_role_indices[main_idx]] = 1.0
            
            # Apply BCE loss only on valid fine roles
            fine_role_loss = self.fine_role_criterion(
                masked_fine_logits * valid_mask,
                fine_labels * valid_mask
            )
            
            # Total loss combines both with higher weight on main role
            total_loss = (2 * main_loss) + fine_role_loss
            
            return ModelOutput(
                main_logits=main_role_logits,
                fine_logits=masked_fine_logits,
                loss=total_loss,
                main_loss=main_loss,
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
