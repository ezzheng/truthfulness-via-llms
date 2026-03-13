# CREDIT: Marks & Tegemark (2023)
# The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets

import torch as t

class LRProbe(t.nn.Module):
    """
    Logistic regression probe: one linear layer (no bias) + sigmoid.
    Trained to predict label in {0, 1} from activations using binary cross-entropy.
    The learned weight vector is the "truth direction" (or concept direction).
    """

    def __init__(self, d_in):
        # d_in: activation dimension (e.g. 5120 for LLaMA-13B hidden size)
        super().__init__()
        # Single linear layer: R^d_in -> R^1, no bias (centered activations typical)
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        # x: (batch, d_in). Returns (batch,) of probabilities in [0, 1].
        # iid is unused; kept for API compatibility with MMProbe.
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        # Hard prediction: 0 or 1 (round of probability).
        return self(x).round()

    @classmethod
    def from_data(cls, acts, labels, lr=0.001, weight_decay=0.1, epochs=200, device='cpu'):
        """
        Train probe on (activations, labels). Returns fitted LRProbe.
        acts: (N, d_in), labels: (N,) in {0, 1}.

        Loss: Binary Cross-Entropy. For each example i,
          BCE_i = -[ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
        We minimize the mean over the batch. Gradient descent updates the linear weights
        so that p_i is close to 1 when y_i=1 and close to 0 when y_i=0.
        """
        acts, labels = acts.to(device), labels.to(device)
        probe = cls(acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            # BCELoss expects float labels in [0,1]; formula: mean over batch of
            # -[ y*log(p) + (1-y)*log(1-p) ]. labels must match shape of probe(acts).
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        return probe

    def __str__(self):
        return "LRProbe"

    @property
    def direction(self):
        # The linear layer's weight row (shape (d_in,)); points toward "positive" class.
        return self.net[0].weight.data[0]

class MMProbe(t.nn.Module):
    """
    "Mean-difference" probe: direction = pos_mean - neg_mean.
    Optional inverse covariance (inv) for Mahalanobis-style scoring: x @ inv @ direction.
    No training; direction and (optionally) inv are set from data in from_data().
    """

    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        # direction: (d_in,) tensor, the mean-difference vector
        # covariance: (d_in, d_in) to compute inv = pinv(covariance) if inv is None
        # inv: (d_in, d_in) inverse (or pseudo-inverse) of covariance; used if provided
        # atol: tolerance for torch.linalg.pinv (singular values below atol are treated as zero)
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False)
        if inv is None:
            # Pseudo-inverse of covariance for stable Mahalanobis-style projection.
            # pinv with hermitian=True uses the fact that covariance is symmetric.
            self.inv = t.nn.Parameter(
                t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False
            )
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False)

    def forward(self, x, iid=False):
        # x: (batch, d_in). iid=True: use inv (Mahalanobis); iid=False: use raw direction.
        if iid:
            # Project onto direction in whitened space: x @ inv @ direction.
            # This is like (x - mean) @ Sigma^{-1} @ direction when data are centered;
            # it downweights high-variance/correlated dimensions.
            return t.nn.Sigmoid()(x @ self.inv @ self.direction)
        else:
            # Simple linear projection: x @ direction (Euclidean dot product).
            return t.nn.Sigmoid()(x @ self.direction)

    def pred(self, x, iid=False):
        return self(x, iid=iid).round()

    @classmethod
    def from_data(cls, acts, labels, atol=1e-3, device='cpu'):
        """
        Build MMProbe from (activations, labels). No training.
        - direction = mean(positive) - mean(negative) (vector from neg to pos centroid).
        - covariance = (1/N) * X_c^T @ X_c where X_c is all activations centered by class mean.
        - inv = pseudo-inverse of covariance for Mahalanobis (iid=True) scoring.
        acts: (N, d_in), labels: (N,) in {0, 1}.
        """
        acts, labels = acts.to(device), labels.to(device)
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        # Centered data for empirical covariance (each class centered by its own mean).
        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]

        probe = cls(direction, covariance=covariance, atol=atol).to(device)
        return probe

    def __str__(self):
        return "MMProbe"

def ccs_loss(probe, acts, neg_acts):
    """
    Contrast-consistent search loss (Burns et al.).
    acts: activations for "positive" (e.g. true) statements; neg_acts: "negative" (e.g. false).
    We do not use per-example labels. Two terms:

    1) Consistency: we want p(positive) and p(negative) to be opposite sides of 0.5, i.e.
       p_pos ≈ 1 - p_neg. So (p_pos - (1 - p_neg))^2.

    2) Confidence: we want the probe to be confident (not stuck at 0.5). We penalize the
       *minimum* of p_pos and p_neg (so if either is near 0.5 we get a penalty), squared.

    Total loss = mean over batch of (consistency_loss + confidence_loss).
    """
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    # Consistency: probe(positive) should equal 1 - probe(negative)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    # Confidence: min(p_pos, p_neg) small => at least one side is confident; we want both
    # confident, so we push min up by penalizing min^2 (min near 0.5 gives high penalty).
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(t.nn.Module):
    """
    Same architecture as LRProbe (linear + sigmoid) but trained with ccs_loss(acts, neg_acts)
    instead of BCE. Used when you have paired positive/negative activations but no single
    label vector, or for contrast-consistent search (e.g. true vs false statement activations).
    """

    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, acts, iid=None):
        return self(acts).round()

    @classmethod
    def from_data(cls, acts, neg_acts, labels=None, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        """
        Train with ccs_loss(probe, acts, neg_acts). acts/neg_acts: (N, d_in).
        If labels is provided, we optionally flip the direction so that accuracy on (acts, labels) >= 0.5
        (so "positive" activations are on the high-probability side).
        """
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = cls(acts.shape[-1]).to(device)

        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        # If labels given, ensure direction points toward "1" (positive) by flipping if acc < 0.5
        if labels is not None:
            labels = labels.to(device)
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1

        return probe

    def __str__(self):
        return "CCSProbe"

    @property
    def direction(self):
        return self.net[0].weight.data[0]
