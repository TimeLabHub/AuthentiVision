import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        Initialize the focal loss function.

        Args:
            alpha (float): Weighting factor for the classes.
            gamma (float): Focusing parameter to reduce the loss contribution from easy examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): Predicted logits of shape (batch_size,).
            targets (torch.Tensor): True labels of shape (batch_size,).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        BCE_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)
