import torch
import torch.nn as nn
from transformers import XLMRobertaModel

class HierarchicalRoleClassifier(nn.Module):
    def __init__(self, model_name: str, main_role_classes: int, fine_role_classes: int):
        super(HierarchicalRoleClassifier, self).__init__()
        
        # Shared encoder (pre-trained transformer)
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.base_model = self.encoder
        
        # Main role classifier (multi-class)
        self.main_role_classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, main_role_classes)
        )

        # Fine-grained role classifier (multi-label)
        self.fine_role_classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, fine_role_classes)
        )

    def forward(self, input_ids, attention_mask):
        # Extract features using the shared encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # CLS token representation

        # Predict main roles and fine roles
        main_role_logits = self.main_role_classifier(pooled_output)
        fine_role_logits = self.fine_role_classifier(pooled_output)

        return main_role_logits, fine_role_logits

    def predict(self, input_ids, attention_mask, taxonomy, threshold=0.5):
        """
        Perform predictions with constraints.
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            taxonomy: Dictionary mapping main roles to allowed fine roles.
            threshold: Threshold for multi-label classification.
        Returns:
            main_role_pred: Predicted main role.
            fine_role_preds: Predicted fine roles constrained by the main role.
        """
        with torch.no_grad():
            main_role_logits, fine_role_logits = self.forward(input_ids, attention_mask)

            # Predict main role (argmax)
            main_role_pred = torch.argmax(main_role_logits, dim=1).item()

            # Get allowed fine roles for the predicted main role
            allowed_indices = taxonomy[main_role_pred]  # List of allowed fine role indices
            fine_role_probs = torch.sigmoid(fine_role_logits)

            # Predict fine roles based on threshold and allowed indices
            fine_role_preds = [
                idx for idx, prob in enumerate(fine_role_probs.squeeze().tolist())
                if prob > threshold and idx in allowed_indices
            ]

        return main_role_pred, fine_role_preds

# Example usage
if __name__ == "__main__":
    model_name = "xlm-roberta-base"
    main_role_classes = 3  # Protagonist, Antagonist, Innocent
    fine_role_classes = 10  # Total number of fine roles

    # Example taxonomy mapping main roles to fine roles
    taxonomy = {
        0: [0, 1, 2],  # Protagonist -> fine roles 0, 1, 2
        1: [3, 4, 5],  # Antagonist -> fine roles 3, 4, 5
        2: [6, 7, 8, 9]  # Innocent -> fine roles 6, 7, 8, 9
    }

    # Initialize model
    model = HierarchicalRoleClassifier(model_name, main_role_classes, fine_role_classes)

    # Dummy input
    input_ids = torch.randint(0, 100, (1, 50))  # Random input IDs
    attention_mask = torch.ones_like(input_ids)

    # Perform prediction
    main_role_pred, fine_role_preds = model.predict(input_ids, attention_mask, taxonomy)
    print(f"Predicted Main Role: {main_role_pred}")
    print(f"Predicted Fine Roles: {fine_role_preds}")
