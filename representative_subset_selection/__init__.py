"""Representative subset selection on CIFAR: random-instance cluster selection from variable-size sets.

The task: given a "bag" of ``N`` ResNet-18 features sampled from ``K``
CIFAR classes, predict a subset that *covers every cluster*. The ground
truth is one randomly-sampled token per cluster, refreshed every time
the bag is drawn from the dataset (so the model cannot memorise a
specific exemplar; what matters is hitting the right cluster).

Four training modes are supported:

* ``cpl``       -- Contextual Plackett-Luce ordering loss with sequential
  greedy decoding and a learnable EOS token.
* ``bce``       -- Per-token binary cross-entropy against the random-GT
  mask, evaluated with a sweep of decision thresholds.
* ``hungarian`` -- DETR-style bipartite matching between valid tokens and
  the ``K`` random GT indices (cost = classification + transformer-h
  distance), followed by BCE on matched / unmatched targets and a
  SmoothL1 consistency term on matched pairs. Inference reuses the BCE
  threshold sweep.
* ``ar``        -- Auto-regressive set-coverage. The bag is encoded once;
  a small ``nn.TransformerDecoder`` consumes a teacher-forced length-
  ``K + 1`` sequence (BOS + a random permutation of the ``K`` GT indices)
  under a causal self-attention mask. A pointer-network readout produces
  per-step logits over ``N + 1`` slots; the loss is the same EOS-weighted
  cross-entropy as ``cpl`` (uniform target over still-unselected GTs at
  non-EOS prefix steps with a ``pre_eos_weight * p[EOS]`` penalty,
  switching to ``post_eos_weight * (-log p[EOS])`` once all ``K`` are in
  the prefix). Greedy AR decode at inference.

K-means with oracle K (number of GT semantic clusters) is reported as a
non-trained reference baseline in all four modes.
"""
