"""Loss functions for image classification.

Classes
-------
LabelSmoothingCrossEntropyLoss
    Cross-entropy loss with label smoothing following Szegedy et al. (2016).
"""

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Replaces the hard one-hot target with a soft distribution: the correct
    class gets probability ``1 - smoothing`` and the remaining
    ``num_classes - 1`` classes each receive ``smoothing / (num_classes - 1)``.

    Reference: Szegedy et al., "Rethinking the Inception Architecture" (2016).

    Args:
        classes:   Number of output classes.
        smoothing: Label-smoothing factor in ``[0, 1)``.  ``0`` recovers
                   standard cross-entropy.  Defaults to ``0.0``.
        dim:       Dimension along which log-softmax is applied.  Defaults
                   to ``-1``.
    """

    def __init__(self, classes: int, smoothing: float = 0.0, dim: int = -1) -> None:
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Args:
            pred:   Logit tensor of shape ``(B, num_classes)``.
            target: Integer class-index tensor of shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
