"""
CORN ordinal loss for PHQ-8 item severity (Shi, Cao & Raschka, 2021).

PHQ-8 severity is ordinal (0 < 1 < 2 < 3). A plain 4-way softmax ignores that
ordering, so an "off-by-one" error costs the same as a wild miss. The near-miss
analysis showed most errors are between *adjacent* severities — exactly what a
rank-consistent ordinal loss is built to fix.

CORN reframes K classes as K-1 binary tasks with *conditional* training sets:
task k predicts P(y > k | y > k-1). The model emits K-1 logits. Rank consistency
(P(y>0) >= P(y>1) >= ...) is guaranteed because the unconditional probabilities
are cumulative products of the conditional sigmoids.

Reference: https://github.com/Raschka-research-group/coral-pytorch
"""

import torch
import torch.nn.functional as F


def corn_loss(logits, y_true, num_classes):
    """CORN loss.

    Parameters
    ----------
    logits : (N, num_classes - 1) float tensor
    y_true : (N,) long tensor of labels in [0, num_classes - 1]
    """
    losses = 0.0
    n_total = 0
    for k in range(num_classes - 1):
        # Conditional subset for task k: examples with y > k-1  (i.e. y >= k).
        mask = y_true > (k - 1)
        if mask.sum() == 0:
            continue
        # Binary target: 1 if y > k.
        target = (y_true[mask] > k).float()
        z = logits[mask, k]
        # Numerically stable BCE-with-logits, summed over the subset.
        # log(sigmoid(z)) = logsigmoid(z); log(1 - sigmoid(z)) = logsigmoid(z) - z
        losses += -torch.sum(
            F.logsigmoid(z) * target + (F.logsigmoid(z) - z) * (1.0 - target)
        )
        n_total += int(mask.sum())
    return losses / max(n_total, 1)


def corn_loss_weighted(logits, y_true, num_classes, class_weight):
    """Class-balanced CORN: weight each example's contribution by ``class_weight[y]``.

    Plain CORN treats every example equally, so the rare severe classes (9.5% of
    items) are drowned out — the near-miss analysis showed severe is *under*-called.
    Up-weighting tail examples in the conditional binary tasks pushes the decision
    boundaries toward catching them, while keeping CORN's rank-consistent head.

    Parameters
    ----------
    logits : (N, num_classes - 1) float tensor
    y_true : (N,) long tensor of labels in [0, num_classes - 1]
    class_weight : (num_classes,) float tensor (e.g. sklearn "balanced" weights)
    """
    w_all = class_weight[y_true]                      # (N,) per-example weight
    total = logits.new_zeros(())
    wsum = logits.new_zeros(())
    for k in range(num_classes - 1):
        mask = y_true > (k - 1)
        if mask.sum() == 0:
            continue
        target = (y_true[mask] > k).float()
        z = logits[mask, k]
        w = w_all[mask]
        bce = -(F.logsigmoid(z) * target + (F.logsigmoid(z) - z) * (1.0 - target))
        total = total + torch.sum(w * bce)
        wsum = wsum + torch.sum(w)
    return total / wsum.clamp(min=1.0)


def corn_loss_focal(logits, y_true, num_classes, gamma=2.0):
    """Focal CORN: down-weight already-easy threshold decisions by ``(1 - p_t)**gamma``.

    Focuses optimisation on the hard, ambiguous boundary cases (Lin et al. 2017)
    while preserving CORN's ordinal structure. Complementary to class balancing:
    targets *hard* examples rather than *rare* classes.
    """
    total = logits.new_zeros(())
    n_total = 0
    for k in range(num_classes - 1):
        mask = y_true > (k - 1)
        if mask.sum() == 0:
            continue
        target = (y_true[mask] > k).float()
        z = logits[mask, k]
        p = torch.sigmoid(z)
        p_t = p * target + (1.0 - p) * (1.0 - target)        # prob of the true side
        focal = (1.0 - p_t).clamp(min=0.0) ** gamma
        bce = -(F.logsigmoid(z) * target + (F.logsigmoid(z) - z) * (1.0 - target))
        total = total + torch.sum(focal * bce)
        n_total += int(mask.sum())
    return total / max(n_total, 1)


def corn_probas(logits):
    """Convert CORN logits to a full (N, num_classes) probability distribution.

    P(y>k)   = prod_{j<=k} sigmoid(logit_j)   (cumulative / rank-consistent)
    P(y=0)   = 1 - P(y>0)
    P(y=k)   = P(y>k-1) - P(y>k)
    P(y=K-1) = P(y>K-2)
    """
    cond = torch.sigmoid(logits)                       # (N, K-1) P(y>k | y>k-1)
    cum = torch.cumprod(cond, dim=1)                   # (N, K-1) P(y>k)
    n, km1 = cum.shape
    probs = torch.zeros(n, km1 + 1, device=logits.device)
    probs[:, 0] = 1.0 - cum[:, 0]
    for k in range(1, km1):
        probs[:, k] = cum[:, k - 1] - cum[:, k]
    probs[:, km1] = cum[:, km1 - 1]
    return probs.clamp(min=0.0)


def corn_predict(logits):
    """Predicted label = number of thresholds with P(y>k) > 0.5 (canonical CORN)."""
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)
    return (cum > 0.5).sum(dim=1)
