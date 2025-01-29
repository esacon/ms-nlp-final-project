import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from typing import Tuple, Dict, Optional
from src.utils import get_device, get_logger
from src.taxonomy import (
    load_taxonomy, get_all_subroles, get_subroles, MAIN_ROLES, get_valid_fine_roles
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


class RoleClassifier(torch.nn.Module):
    def __init__(self, base_model_name, num_main_roles=3, num_fine_roles=10):
        super().__init__()
        self.base_model = XLMRobertaModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        self.num_main_roles = num_main_roles
        self.num_fine_roles = num_fine_roles
        
        # Role classifiers
        self.main_role_classifier = torch.nn.Linear(hidden_size, num_main_roles)
        self.fine_role_classifier = torch.nn.Linear(hidden_size, num_fine_roles)
        
        # Loss functions
        self.main_role_criterion = torch.nn.BCEWithLogitsLoss()
        self.fine_role_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Valid fine roles for each main role
        self.valid_roles = {
            0: [0, 1, 2],  # Innocent: Victim, Forgotten, Deceiver
            1: [3, 4, 5, 6],  # Antagonist: Tyrant, Traitor, Instigator, Conspirator
            2: [7, 8, 9]  # Protagonist: Guardian, Peacemaker, Underdog
        }
        
    def forward(self, input_ids, attention_mask, entity_positions, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
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

        # Get fine role predictions with masking
        fine_role_logits = self.fine_role_classifier(entity_embeddings)
        
        # Create role masks
        valid_roles = {
            0: [0, 1, 2],  # Innocent: Victim, Forgotten, Deceiver
            1: [3, 4, 5, 6],  # Antagonist: Tyrant, Traitor, Instigator, Conspirator
            2: [7, 8, 9]  # Protagonist: Guardian, Peacemaker, Underdog
        }
        
        fine_role_mask = torch.zeros((batch_size, self.num_fine_roles), device=main_role_pred.device)
        for i in range(batch_size):
            main_role = main_role_pred[i].item()
            fine_role_mask[i, valid_roles[main_role]] = 1.0

        # Apply mask and threshold
        masked_fine_logits = fine_role_logits * fine_role_mask
        fine_role_probs = torch.sigmoid(masked_fine_logits)
        
        threshold = 0.7
        fine_role_pred = torch.zeros_like(fine_role_probs)
        for i in range(batch_size):
            valid_indices = valid_roles[main_role_pred[i].item()]
            valid_probs = fine_role_probs[i, valid_indices]
            
            # Get top 2 valid predictions above threshold
            top_values, top_indices = torch.topk(valid_probs, k=min(2, len(valid_indices)))
            selected_indices = [valid_indices[idx] for idx in top_indices if top_values[idx] >= threshold]
            
            # Always take at least one role (highest probability)
            if not selected_indices:
                selected_indices = [valid_indices[valid_probs.argmax()]]
            
            fine_role_pred[i, selected_indices] = 1.0

        if labels is not None:
            main_role_labels = labels[:, :self.num_main_roles]
            fine_role_labels = labels[:, self.num_main_roles:]
            
            main_role_loss = self.main_role_criterion(main_role_logits, main_role_labels)
            fine_role_loss = self.fine_role_criterion(masked_fine_logits, fine_role_labels)
            
            # Add sparsity regularization
            l1_reg = torch.norm(fine_role_probs, p=1) * 0.01
            
            total_loss = main_role_loss + fine_role_loss + l1_reg
            return total_loss, main_role_pred, fine_role_pred

        return None, main_role_pred, fine_role_pred


class EntityRoleClassifier(torch.nn.Module):
    def __init__(self, base_model_name: str, num_main_roles: int = 3, num_fine_roles: int = 10):
        super().__init__()
        self.base_model = XLMRobertaModel.from_pretrained(base_model_name)
        hidden_size = self.base_model.config.hidden_size
        
        self.num_main_roles = num_main_roles
        self.num_fine_roles = num_fine_roles
        
        # Role classifiers
        self.main_role_classifier = torch.nn.Linear(hidden_size, num_main_roles)
        self.fine_role_classifier = torch.nn.Linear(hidden_size, num_fine_roles)
        
        # Loss functions
        self.main_role_criterion = torch.nn.BCEWithLogitsLoss()
        self.fine_role_criterion = torch.nn.BCEWithLogitsLoss()
        
        # Load taxonomy and create role mappings
        self.taxonomy = load_taxonomy()
        self.all_subroles = get_all_subroles(self.taxonomy)
        self.role_to_idx = {role: idx for idx, role in enumerate(self.all_subroles)}
        
        # Create valid role indices for each main role
        self.valid_role_indices = {}
        for main_idx, main_role in enumerate(MAIN_ROLES):
            valid_subroles = get_subroles(self.taxonomy, main_role)
            self.valid_role_indices[main_idx] = [
                self.role_to_idx[role] for role in valid_subroles if role in self.role_to_idx
            ]
        
    def forward(self, input_ids, attention_mask, entity_positions, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
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

        # Get fine role predictions with masking
        fine_role_logits = self.fine_role_classifier(entity_embeddings)
        
        # Create masks for valid fine roles
        fine_role_mask = torch.zeros((batch_size, self.num_fine_roles), device=main_role_pred.device)
        for i in range(batch_size):
            main_role = main_role_pred[i].item()
            fine_role_mask[i, self.valid_role_indices[main_role]] = 1.0

        # Apply mask and threshold
        masked_fine_logits = fine_role_logits * fine_role_mask
        fine_role_probs = torch.sigmoid(masked_fine_logits)
        
        threshold = 0.7
        fine_role_pred = torch.zeros_like(fine_role_probs)
        for i in range(batch_size):
            valid_indices = self.valid_role_indices[main_role_pred[i].item()]
            valid_probs = fine_role_probs[i, valid_indices]
            
            # Get top 2 valid predictions above threshold
            top_values, top_indices = torch.topk(valid_probs, k=min(2, len(valid_indices)))
            selected_indices = [valid_indices[idx] for idx in top_indices if top_values[idx] >= threshold]
            
            # Always take at least one role (highest probability)
            if not selected_indices:
                selected_indices = [valid_indices[valid_probs.argmax()]]
            
            fine_role_pred[i, selected_indices] = 1.0

        if labels is not None:
            main_role_labels = labels[:, :self.num_main_roles]
            fine_role_labels = labels[:, self.num_main_roles:]
            
            main_role_loss = self.main_role_criterion(main_role_logits, main_role_labels)
            fine_role_loss = self.fine_role_criterion(masked_fine_logits, fine_role_labels)
            
            # Add sparsity regularization
            l1_reg = torch.norm(fine_role_probs, p=1) * 0.01
            
            total_loss = main_role_loss + fine_role_loss + l1_reg
            return total_loss, main_role_pred, fine_role_pred

        return None, main_role_pred, fine_role_pred


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

        # Add special entity marker tokens with improved handling
        special_tokens = ['[ENT]', '[/ENT]']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = self.tokenizer.add_special_tokens(
            special_tokens_dict)

        # Resize token embeddings and initialize new embeddings
        self.roberta.resize_token_embeddings(len(self.tokenizer))

        # Initialize new token embeddings with similar statistics to existing ones
        with torch.no_grad():
            embedding_weights = self.roberta.embeddings.word_embeddings.weight
            existing_mean = embedding_weights[:-num_added_tokens].mean(dim=0)
            existing_std = embedding_weights[:-num_added_tokens].std(dim=0)

            for i in range(num_added_tokens):
                idx = -(i + 1)
                # Initialize each dimension separately with its own mean and std
                for dim in range(embedding_weights.size(1)):
                    torch.nn.init.normal_(
                        embedding_weights[idx:idx+1, dim:dim+1],
                        mean=float(existing_mean[dim].item()),
                        std=float(existing_std[dim].item())
                    )

        # Verify special tokens are treated as single tokens
        for token in special_tokens:
            tokenized = self.tokenizer.tokenize(token)
            if len(tokenized) != 1:
                logger.warning(
                    f"Special token {token} is split into {len(tokenized)} subwords: {tokenized}")

        hidden_size = self.roberta.config.hidden_size
        combined_size = hidden_size * 2

        # Entity-aware attention layer with improved initialization
        self.entity_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            dropout=config["model"]["attention_dropout"],
            batch_first=True
        )

        # Initialize attention weights for better gradient flow
        with torch.no_grad():
            for param in self.entity_attention.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param)

        # Entity representation layer with skip connection
        self.entity_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU()
        )

        # Shared representation layer with skip connection
        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(combined_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU()
        )

        # Main role classifier with improved architecture
        self.main_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.LayerNorm(hidden_size // 4),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 4, 3)
        )

        # Fine-grained role classifier with improved architecture
        self.fine_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.Dropout(config["model"]["dropout_prob"]),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, num_fine_labels)
        )

        # Initialize all linear layers with improved weight initialization
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # Loss weights with dynamic adjustment
        self.main_loss_weight = torch.nn.Parameter(torch.tensor([3.0]))
        self.fine_loss_weight = torch.nn.Parameter(torch.tensor([1.0]))

        # Class weights for main role classification with dynamic adjustment
        self.main_class_weights = torch.nn.Parameter(
            torch.tensor([1.5, 1.5, 1.0])
        ).to(self.device)

        # Temperature scaling for logits with improved initialization
        self.main_temperature = torch.nn.Parameter(torch.ones(1))
        self.fine_temperature = torch.nn.Parameter(torch.ones(1))

        # Load taxonomy for role constraints
        self.taxonomy = load_taxonomy()
        self.role_masks = self._create_role_masks()

        # Initialize role distribution priors (based on training data distribution)
        self.main_role_prior = torch.tensor([0.4, 0.4, 0.2]).to(
            self.device)  # [Innocent, Antagonist, Protagonist]

        # Initialize valid fine-grained roles per main role
        self.valid_fine_roles = {
            0: get_valid_fine_roles(self.taxonomy, 'Innocent'),  # Innocent
            1: get_valid_fine_roles(self.taxonomy, 'Antagonist'),  # Antagonist
            # Protagonist
            2: get_valid_fine_roles(self.taxonomy, 'Protagonist')
        }

        # Create indices for valid roles
        self.valid_role_indices = {}
        all_subroles = get_all_subroles(self.taxonomy)
        role_to_idx = {role: idx for idx, role in enumerate(all_subroles)}

        for main_role, valid_roles in self.valid_fine_roles.items():
            self.valid_role_indices[main_role] = torch.tensor(
                [role_to_idx[role]
                    for role in valid_roles if role in role_to_idx],
                device=self.device
            )

    def _create_role_masks(self) -> torch.Tensor:
        """Create binary masks for valid fine-grained roles per main role."""
        masks = torch.zeros(3, self.num_fine_labels)  # 3 main roles

        # Create mapping from role names to indices
        all_subroles = get_all_subroles(self.taxonomy)
        role_to_idx = {role: idx for idx, role in enumerate(all_subroles)}

        # Set 1s for valid fine roles per main role
        for main_idx, main_role in enumerate(MAIN_ROLES):
            valid_subroles = get_subroles(self.taxonomy, main_role)
            for subrole in valid_subroles:
                if subrole in role_to_idx:
                    masks[main_idx, role_to_idx[subrole]] = 1.0

        return masks.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_positions: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        main_labels: Optional[torch.Tensor] = None,
        fine_labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """Forward pass with enhanced entity representation."""
        batch_size = input_ids.size(0)

        # Validate input shapes
        if entity_positions.dim() != 2 and entity_positions.dim() != 3:
            raise ValueError(
                f"Entity positions should be 2D or 3D tensor, got {entity_positions.dim()}D")

        if entity_positions.size(-1) != 2:
            raise ValueError(
                f"Last dimension of entity_positions should be 2 (start, end), got {entity_positions.size(-1)}")

        # Debug logging for input verification
        # Log 1% of batches during training
        if self.training and torch.rand(1).item() < 0.01:
            logger.debug(f"\nInput verification:")
            logger.debug(f"Batch size: {batch_size}")
            logger.debug(f"Input shape: {input_ids.shape}")
            logger.debug(f"Entity positions shape: {entity_positions.shape}")
            if embeddings is not None:
                logger.debug(
                    f"Pre-computed embeddings shape: {embeddings.shape}")
            if main_labels is not None:
                logger.debug(f"Main labels: {main_labels}")
            if fine_labels is not None:
                logger.debug(f"Fine labels shape: {fine_labels.shape}")

        # Get base embeddings
        if embeddings is not None:
            sequence_output = embeddings
            sequence_length = sequence_output.size(1)
            # Truncate attention mask to match sequence length
            attention_mask = attention_mask[:, :sequence_length]

            # Scale entity positions to match embedding sequence length
            if entity_positions.dim() == 3:
                orig_length = input_ids.size(1)
                scale_factor = sequence_length / orig_length
                entity_positions = (entity_positions.float()
                                    * scale_factor).long()
            else:
                orig_length = input_ids.size(1)
                scale_factor = sequence_length / orig_length
                entity_positions = (entity_positions.float()
                                    * scale_factor).long()
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            sequence_output = outputs.last_hidden_state
            sequence_length = sequence_output.size(1)

        # Process each example in the batch
        entity_embeddings_list = []
        attention_weights_list = []

        for i in range(batch_size):
            try:
                # Get entity span
                if entity_positions.dim() == 3:
                    start_pos = entity_positions[i, 0, 0].item()
                    end_pos = entity_positions[i, 0, 1].item()
                else:
                    start_pos = entity_positions[i, 0].item()
                    end_pos = entity_positions[i, 1].item()

                # Ensure positions are within bounds
                start_pos = max(0, min(start_pos, sequence_length - 1))
                end_pos = max(start_pos + 1, min(end_pos, sequence_length))

                # Get entity span embeddings
                # [entity_len, hidden_size]
                entity_span = sequence_output[i, start_pos:end_pos]

                # Apply self-attention over the sequence
                key_padding_mask = ~attention_mask[i].bool(
                ).unsqueeze(0)  # [1, seq_len]

                # Prepare inputs for attention
                entity_span = entity_span.unsqueeze(
                    0)  # [1, entity_len, hidden_size]
                sequence = sequence_output[i].unsqueeze(
                    0)  # [1, seq_len, hidden_size]

                # Apply entity-aware attention
                entity_attn_out, attn_weights = self.entity_attention(
                    entity_span,
                    sequence,
                    sequence,
                    key_padding_mask=key_padding_mask,
                    need_weights=True
                )

                # Pool attention output
                entity_embedding = torch.mean(
                    entity_attn_out.squeeze(0), dim=0)  # [hidden_size]
                entity_embeddings_list.append(entity_embedding)
                attention_weights_list.append(attn_weights)

            except Exception as e:
                logger.error(f"Error processing entity {i} in batch: {str(e)}")
                # Use fallback strategy: mean pooling of the entire sequence
                masked_output = sequence_output[i] * \
                    attention_mask[i].unsqueeze(-1)
                entity_embedding = torch.mean(masked_output, dim=0)
                entity_embeddings_list.append(entity_embedding)
                attention_weights_list.append(None)

        # Stack entity embeddings [batch_size, hidden_size]
        entity_embeddings = torch.stack(entity_embeddings_list)

        # Get enhanced entity representation
        entity_features = self.entity_layer(entity_embeddings)

        # Get context representation from CLS token
        context_embeddings = sequence_output[:, 0]

        # Combine entity and context representations
        combined_embedding = torch.cat(
            [entity_features, context_embeddings], dim=1)
        shared_features = self.shared_layer(combined_embedding)

        # Get predictions with temperature scaling
        main_logits_raw = self.main_classifier(shared_features)
        fine_logits_raw = self.fine_classifier(shared_features)

        # Apply temperature scaling with numerical stability
        main_temperature = torch.clamp(
            self.main_temperature, min=0.1, max=10.0)
        fine_temperature = torch.clamp(
            self.fine_temperature, min=0.1, max=10.0)

        # Apply prior correction to main logits
        main_logits = (main_logits_raw -
                       torch.log(self.main_role_prior)) / main_temperature

        # Get main role predictions
        main_probs = F.softmax(main_logits, dim=1)  # [batch_size, 3]
        main_predictions = torch.argmax(main_logits, dim=1)  # [batch_size]

        # Apply temperature scaling to fine-grained logits
        fine_logits = fine_logits_raw / fine_temperature

        # Get fine-grained probabilities
        fine_probs = torch.sigmoid(fine_logits)

        # Initialize fine predictions tensor
        fine_predictions = torch.zeros_like(fine_probs, dtype=torch.bool)

        # Process each example individually
        for i in range(len(main_predictions)):
            main_role = main_predictions[i].item()
            valid_indices = self.valid_role_indices[main_role]

            # Get probabilities for valid roles
            role_probs = fine_probs[i, valid_indices]

            # Sort probabilities
            sorted_probs, sorted_indices = torch.sort(
                role_probs, descending=True)

            # Select roles with high confidence
            high_conf_mask = sorted_probs >= 0.7
            if high_conf_mask.sum() > 0:
                # Take up to 2 highest confidence roles
                num_selected = min(high_conf_mask.sum().item(), 2)
                selected_local_indices = sorted_indices[:num_selected]
                selected_global_indices = valid_indices[selected_local_indices]
                fine_predictions[i, selected_global_indices] = True
            else:
                # Take the single highest probability role
                best_local_index = sorted_indices[0]
                best_global_index = valid_indices[best_local_index]
                fine_predictions[i, best_global_index] = True

        # Calculate losses if labels are provided
        loss = None
        main_loss = None
        fine_loss = None

        if main_labels is not None and fine_labels is not None:
            # Main role loss with class weights and focal loss component
            main_log_probs = F.log_softmax(main_logits, dim=1)

            # Calculate cross entropy with label smoothing
            main_loss = F.cross_entropy(
                main_logits,
                main_labels,
                weight=self.main_class_weights,
                label_smoothing=0.1,
                reduction='none'
            )

            # Add focal loss component for main task
            pt = main_probs.gather(1, main_labels.unsqueeze(1)).squeeze()
            focal_weight = (1 - pt) ** 2
            focal_loss = -(focal_weight * main_log_probs.gather(1,
                           main_labels.unsqueeze(1))).squeeze()

            # Add distribution matching loss
            batch_dist = main_probs.mean(0)
            dist_loss = F.kl_div(
                batch_dist.log(),
                self.main_role_prior,
                reduction='batchmean'
            )

            # Combine losses with stability
            main_loss = main_loss + 0.5 * focal_loss + 0.2 * dist_loss
            main_loss = main_loss.mean()

            # Only compute fine-grained loss for valid roles
            fine_loss = torch.zeros(1, device=self.device)

            for i, (main_label, fine_label) in enumerate(zip(main_labels, fine_labels)):
                valid_indices = self.valid_role_indices[main_label.item()]

                # Compute loss only for valid roles
                valid_logits = fine_logits_raw[i, valid_indices]
                valid_labels = fine_label[valid_indices]

                if valid_labels.sum() > 0:
                    # Binary cross entropy for valid roles
                    role_loss = F.binary_cross_entropy_with_logits(
                        valid_logits,
                        valid_labels,
                        reduction='none'
                    )

                    # Weight positive examples more heavily
                    pos_weight = (valid_labels == 1).float() * 2.0 + 1.0
                    role_loss = role_loss * pos_weight

                    # Add focal component
                    valid_probs = torch.sigmoid(valid_logits)
                    pt = torch.where(valid_labels == 1,
                                     valid_probs, 1 - valid_probs)
                    focal_weight = (1 - pt) ** 2
                    role_loss = role_loss * focal_weight

                    fine_loss = fine_loss + role_loss.mean()

            fine_loss = fine_loss / len(main_labels)

            # Add sparsity regularization for valid roles only
            sparsity_loss = torch.zeros(1, device=self.device)
            for i, main_pred in enumerate(main_predictions):
                valid_indices = self.valid_role_indices[main_pred.item()]
                valid_probs = fine_probs[i, valid_indices]
                sparsity_loss = sparsity_loss + torch.abs(valid_probs).mean()
            sparsity_loss = sparsity_loss / len(main_predictions) * 0.1

            # Combine losses with weights and gradient scaling
            main_scale = torch.exp(-self.main_temperature)
            fine_scale = torch.exp(-self.fine_temperature)

            loss = (
                self.main_loss_weight * main_loss * main_scale +
                self.fine_loss_weight * fine_loss * fine_scale +
                sparsity_loss
            )

            # Debug predictions during training
            if self.training and torch.rand(1).item() < 0.01:
                with torch.no_grad():
                    main_preds = torch.argmax(main_logits, dim=1)
                    main_acc = (
                        main_preds == main_labels).float().mean().item()

                    # Calculate fine-grained metrics only for valid roles
                    fine_acc = 0.0
                    total_valid = 0
                    for i, (main_label, fine_label) in enumerate(zip(main_labels, fine_labels)):
                        valid_indices = self.valid_role_indices[main_label.item(
                        )]
                        valid_preds = fine_predictions[i, valid_indices]
                        valid_labels = fine_label[valid_indices]
                        if valid_labels.sum() > 0:
                            fine_acc += (valid_preds ==
                                         valid_labels).float().mean().item()
                            total_valid += 1

                    if total_valid > 0:
                        fine_acc /= total_valid

                    logger.debug(
                        f"Batch metrics - Main acc: {main_acc:.4f}, Fine acc: {fine_acc:.4f}")
                    logger.debug(
                        f"Main role distribution: {main_probs.mean(0)}")
                    logger.debug(
                        f"Average roles per entity: {fine_predictions.float().sum(1).mean().item():.2f}")
                    logger.debug(f"Loss components - Main: {main_loss.item():.4f}, Fine: {fine_loss.item():.4f}, "
                                 f"Sparsity: {sparsity_loss.item():.4f}")

        return ModelOutput(
            main_logits=main_logits,
            fine_logits=fine_logits,
            loss=loss,
            main_loss=main_loss,
            fine_loss=fine_loss,
            attention_weights=attention_weights_list[0] if attention_weights_list else None
        )

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
            fine_correct += torch.all(fine_preds ==
                                      fine_labels, dim=1).sum().item()

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
            'fine_loss_weight': self.fine_loss_weight,
            'main_temperature': self.main_temperature,
            'fine_temperature': self.fine_temperature
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
        instance.entity_attention.load_state_dict(
            components['entity_attention'])
        instance.main_loss_weight = components['main_loss_weight']
        instance.fine_loss_weight = components['fine_loss_weight']
        instance.main_temperature = components['main_temperature']
        instance.fine_temperature = components['fine_temperature']

        return instance


def load_trained_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    return model
