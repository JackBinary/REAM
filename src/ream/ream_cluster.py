"""
REAM clustering: REAP-score-based centroid selection with pseudo-pruning grouping
and gated similarity.

Implements Steps 1-4 of the REAM algorithm from:
  https://bknyaz.github.io/blog/2026/moe/
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def gated_similarity(
    expert_outputs: torch.Tensor,
    gate_logits: torch.Tensor,
    expert_i: int,
    expert_j: int,
) -> float:
    """
    Compute gated similarity between two experts.

    REAM uses an average of:
      - cosine similarity of expert outputs
      - cosine similarity of gate logits
    Both weighted (multiplied) by gate logits to produce "gated similarity".

    Args:
        expert_outputs: (num_experts, num_tokens, hidden_dim) expert activations
        gate_logits: (num_tokens, num_experts) router softmax outputs
        expert_i: index of first expert
        expert_j: index of second expert

    Returns:
        Scalar similarity score (higher = more similar).
    """
    # Expert output cosine similarity (per-token, then mean)
    out_i = expert_outputs[expert_i]  # (num_tokens, hidden_dim)
    out_j = expert_outputs[expert_j]  # (num_tokens, hidden_dim)
    cos_out = F.cosine_similarity(out_i, out_j, dim=-1)  # (num_tokens,)

    # Gate logit cosine similarity: treat each expert's gate column as a vector
    # across tokens
    gate_i = gate_logits[:, expert_i]  # (num_tokens,)
    gate_j = gate_logits[:, expert_j]  # (num_tokens,)
    cos_gate = F.cosine_similarity(gate_i.unsqueeze(0), gate_j.unsqueeze(0), dim=-1)

    # Gated weighting: multiply by gate logits
    gate_weight_i = gate_logits[:, expert_i]
    gate_weight_j = gate_logits[:, expert_j]
    gate_weight = (gate_weight_i + gate_weight_j) / 2.0  # (num_tokens,)

    # Weighted expert output similarity
    gated_cos_out = (cos_out * gate_weight).sum() / (gate_weight.sum() + 1e-8)

    # Average of expert-output sim and gate-logit sim
    sim = 0.5 * (gated_cos_out + cos_gate.item())
    return sim


def compute_gated_similarity_matrix(
    expert_outputs: torch.Tensor,
    gate_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute the full NxN gated similarity matrix between all experts.

    Args:
        expert_outputs: (num_experts, num_tokens, hidden_dim)
        gate_logits: (num_tokens, num_experts) softmax router outputs
        num_experts: N

    Returns:
        (N, N) similarity matrix
    """
    sim_matrix = torch.zeros(num_experts, num_experts)
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            s = gated_similarity(expert_outputs, gate_logits, i, j)
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s
    return sim_matrix


def ream_clustering(
    reap_scores: torch.Tensor,
    expert_outputs: torch.Tensor,
    gate_logits: torch.Tensor,
    num_experts: int,
    num_clusters: int,
    max_cluster_size: int = 16,
) -> torch.Tensor:
    """
    REAM clustering with pseudo-pruning grouping.

    Steps:
      1. Pick k experts with highest REAP scores as centroids.
      2. Starting from the centroid with the highest score, assign the C most
         similar unassigned experts to it. Repeat for remaining centroids.
      3. Similarity is computed via "gated similarity" (average of cosine
         similarity of expert outputs and gate logits, weighted by gate values).

    Most clusters will contain 1 expert (the centroid alone). Only a few clusters
    will be large, dominated by high-scoring centroids. This is "pseudo-pruning":
    the weighted average in the merge step will be dominated by the centroid.

    Args:
        reap_scores: (N,) REAP saliency scores per expert
        expert_outputs: (N, num_tokens, hidden_dim)
        gate_logits: (num_tokens, N)
        num_experts: N
        num_clusters: k (target number of merged experts)
        max_cluster_size: C, max experts per cluster (default 16)

    Returns:
        (N,) cluster label tensor. Centroid experts get their own cluster id.
    """
    assert num_clusters <= num_experts, (
        f"num_clusters ({num_clusters}) must be <= num_experts ({num_experts})"
    )

    # Step 1: Pick k centroids with highest REAP scores
    _, centroid_indices = torch.topk(reap_scores, num_clusters)
    centroid_indices = centroid_indices.tolist()

    # Sort centroids by REAP score descending (highest first for greedy assignment)
    centroid_indices = sorted(
        centroid_indices, key=lambda idx: reap_scores[idx].item(), reverse=True
    )

    # Step 2: Compute gated similarity matrix
    logger.info("Computing gated similarity matrix...")
    sim_matrix = compute_gated_similarity_matrix(
        expert_outputs, gate_logits, num_experts
    )

    # Step 3: Pseudo-pruning grouping
    labels = torch.full((num_experts,), -1, dtype=torch.long)

    # Assign each centroid to its own cluster
    for cluster_id, centroid_idx in enumerate(centroid_indices):
        labels[centroid_idx] = cluster_id

    assigned = set(centroid_indices)
    unassigned = set(range(num_experts)) - assigned

    # Total non-centroid experts to distribute
    total_non_centroid = num_experts - num_clusters
    # Each centroid can absorb at most (max_cluster_size - 1) non-centroids
    budget_per_centroid = max_cluster_size - 1

    for cluster_id, centroid_idx in enumerate(centroid_indices):
        if not unassigned:
            break

        # Find most similar unassigned experts to this centroid
        unassigned_list = list(unassigned)
        sims = sim_matrix[centroid_idx, unassigned_list]
        num_to_assign = min(budget_per_centroid, len(unassigned_list))

        if num_to_assign > 0:
            _, top_indices = torch.topk(sims, num_to_assign)
            for idx in top_indices:
                expert_idx = unassigned_list[idx.item()]
                labels[expert_idx] = cluster_id
                assigned.add(expert_idx)
                unassigned.discard(expert_idx)

    # Any remaining unassigned experts get assigned to the nearest centroid
    if unassigned:
        logger.warning(
            f"{len(unassigned)} experts still unassigned after pseudo-pruning. "
            "Assigning to nearest centroid."
        )
        for expert_idx in list(unassigned):
            best_sim = -float("inf")
            best_cluster = 0
            for cluster_id, centroid_idx in enumerate(centroid_indices):
                s = sim_matrix[centroid_idx, expert_idx].item()
                if s > best_sim:
                    best_sim = s
                    best_cluster = cluster_id
            labels[expert_idx] = best_cluster

    return labels


def ream_clustering_from_observer(
    observer_data_layer: dict,
    num_clusters: int,
    max_cluster_size: int = 16,
    expert_outputs: Optional[torch.Tensor] = None,
    gate_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper that extracts REAP scores from observer data and runs REAM clustering.

    When expert_outputs and gate_logits are provided directly (for sequential mode),
    uses them instead of observer data.

    Args:
        observer_data_layer: Observer data dict for one layer
        num_clusters: target number of merged experts
        max_cluster_size: max experts per cluster
        expert_outputs: optional direct expert outputs (N, tokens, hidden)
        gate_logits: optional direct gate logits (tokens, N)

    Returns:
        (N,) cluster labels
    """
    reap_scores = observer_data_layer["reap"].mean
    num_experts = reap_scores.shape[0]

    if expert_outputs is None or gate_logits is None:
        # Fall back to observer-stored data
        # Use characteristic activations as proxy for expert outputs
        ca = observer_data_layer["characteristic_activation"]
        # Use router logit similarity as proxy
        router_sim = observer_data_layer["router_logit_similiarity"]

        # Create pseudo expert outputs from characteristic activations
        expert_outputs = ca.unsqueeze(1)  # (N, 1, hidden)
        gate_logits = F.softmax(
            router_sim.mean(dim=0).unsqueeze(0), dim=-1
        )  # (1, N)

    return ream_clustering(
        reap_scores=reap_scores,
        expert_outputs=expert_outputs,
        gate_logits=gate_logits,
        num_experts=num_experts,
        num_clusters=num_clusters,
        max_cluster_size=max_cluster_size,
    )
