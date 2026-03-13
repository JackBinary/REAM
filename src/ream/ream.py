"""
REAM: Router-weighted Expert Activation Merging

Main pipeline implementing the sequential merging approach from:
  https://bknyaz.github.io/blog/2026/moe/

Key differences from REAP's existing merging pipeline:
  1. REAP-score-based centroid selection (not frequency)
  2. Pseudo-pruning grouping with gated similarity
  3. Permutation alignment using both activations AND weights
  4. Sequential layer processing (recompute activations after each merge)
  5. Gate weight pruning (remove non-centroid expert rows from router)
  6. Mixed calibration data (c4 + math + coding)
"""

from __future__ import annotations
import time
import pickle
import logging
import dataclasses
import pathlib
import gc
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import load_dataset

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module

from ream.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    KdArgs,
    EvalArgs,
    MergeArgs,
)
from ream.merge import MergeMethod, MoEExpertMerger
from ream.model_util import (
    MODEL_ATTRS, patched_model_map, get_moe, assert_merge,
    get_layers, get_num_experts, fused_expert_forward, load_model_text_only,
)
from ream.permute import ActivationWeightPermuter, PERMUTER_REGISTRY
from ream.ream_cluster import ream_clustering
from ream.eval import run_evaluate
from ream.main import (
    parse_args,
    create_results_directory,
    save_merged_model,
    smoke_test,
    dump_args_to_yaml,
    get_model_dir,
    str_to_directory_name,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Observer: collect per-layer REAP scores, expert outputs, and gate logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_layer_data(
    model: nn.Module,
    calibration_inputs: list[torch.Tensor],
    layer_idx: int,
    model_attrs: dict[str, Any],
    device: torch.device,
    max_sim_tokens: int = 65536,
    max_hidden_tokens: int = 8192,
) -> dict[str, torch.Tensor]:
    """
    Forward calibration data through the model up to and including the target
    MoE layer, collecting expert outputs, gate logits, and computing REAP scores.

    Memory-efficient: REAP scores are computed incrementally (running sums).
    Only a capped reservoir of tokens is kept for the similarity matrix used
    during clustering, and expert_hidden is capped per expert.

    GPU-optimized: Minimizes CPU-GPU synchronization by batching transfers
    and keeping computations on GPU as long as possible.

    Args:
        max_sim_tokens: Maximum tokens to retain for the gated-similarity matrix.
            4096 is plenty for accurate cosine similarity estimation.
        max_hidden_tokens: Maximum tokens per expert to retain for permutation
            alignment in expert_hidden.

    Returns a dict with:
      - reap_scores: (N,) per-expert REAP saliency
      - expert_outputs: (N, <=max_sim_tokens, hidden_dim) sampled expert activations
      - gate_logits: (<=max_sim_tokens, N) sampled softmax router outputs
      - expert_hidden: dict[int, (d, <=max_hidden_tokens)] hidden activations per expert
      - expert_frequency: (N,) how often each expert is selected
    """
    import numpy as np
    
    moe = get_moe(model, layer_idx)
    experts = getattr(moe, model_attrs["experts"])
    router = getattr(moe, model_attrs["router"])
    is_fused = model_attrs.get("fused", False)
    num_experts = get_num_experts(experts, model_attrs)

    # Determine topk from model config (handle nested text_config for VL models)
    topk_key = model_attrs.get("num_experts_per_tok", "num_experts_per_tok")
    cfg = getattr(model.config, "text_config", model.config)
    topk = getattr(cfg, topk_key, 8)

    # --- Online accumulators for REAP scores (keep on GPU - small fixed size) ---
    reap_numer = torch.zeros(num_experts, device=device, dtype=torch.float32)
    reap_denom = torch.zeros(num_experts, device=device, dtype=torch.float32)
    expert_frequency = torch.zeros(num_experts, device=device, dtype=torch.float32)
    
    # --- CPU buffers for expert_hidden (move to CPU immediately to avoid GPU accumulation) ---
    expert_hidden_cpu: dict[int, torch.Tensor] = {}

    # --- CPU buffers for similarity data (move to CPU immediately) ---
    sim_expert_outputs_cpu: list[torch.Tensor] = []
    sim_gate_logits_cpu: list[torch.Tensor] = []
    sim_tokens_collected = 0
    total_tokens = 0

    # Hook to capture the input to the MoE block
    captured = {}

    def capture_moe_input(module, args, output):
        if isinstance(args, tuple) and len(args) > 0:
            captured["moe_input"] = args[0].detach()

    hook = moe.register_forward_hook(capture_moe_input)

    logger.info(f"Collecting data for layer {layer_idx}: processing {len(calibration_inputs)} calibration batches")
    
    for batch_idx, batch_input in enumerate(tqdm(calibration_inputs, desc=f"Layer {layer_idx} forward passes", leave=False)):
        if isinstance(batch_input, torch.Tensor):
            inputs = batch_input.to(device, non_blocking=True)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            model_input = {"input_ids": inputs}
        elif isinstance(batch_input, dict):
            model_input = {k: v.to(device, non_blocking=True) for k, v in batch_input.items()}
        else:
            continue

        captured.clear()
        try:
            _ = model(**model_input)
        except Exception as e:
            logger.warning(f"Forward pass failed for batch {batch_idx}: {e}")
            continue

        if "moe_input" not in captured:
            continue

        moe_input = captured["moe_input"]
        if moe_input.dim() == 3:
            moe_input = moe_input.view(-1, moe_input.shape[-1])
        n_tokens = moe_input.shape[0]

        # Compute gate logits manually
        gate_out = router(moe_input)  # (n_tokens, N)
        # Some routers (e.g. Qwen3.5) return a tuple (logits, scores, indices)
        if isinstance(gate_out, tuple):
            gate_out = gate_out[0]
        gate_probs = F.softmax(gate_out, dim=-1)

        # Compute expert outputs for all experts - OPTIMIZED with batched matmul where possible
        if is_fused:
            # For fused experts, compute sequentially (fused experts have special handling)
            expert_outs = torch.empty(
                num_experts, n_tokens, moe_input.shape[-1],
                device=device, dtype=moe_input.dtype
            )
            for i in range(num_experts):
                expert_outs[i] = fused_expert_forward(experts, i, moe_input)
        else:
            # Try batched computation by stacking expert weights
            # This runs all experts in parallel via batched matmul
            try:
                # Get first expert to check structure
                first_expert = experts[0]
                param_names = list(first_expert.state_dict().keys())
                
                # Check for standard MLP structure (gate_proj, up_proj, down_proj)
                has_gate = any('gate_proj' in n for n in param_names)
                has_up = any('up_proj' in n for n in param_names)
                has_down = any('down_proj' in n for n in param_names)
                
                if has_gate and has_up and has_down:
                    # SwiGLU-style expert - stack weights for batched matmul
                    gate_weights = torch.stack([getattr(e, 'gate_proj').weight for e in experts], dim=0)
                    up_weights = torch.stack([getattr(e, 'up_proj').weight for e in experts], dim=0)
                    down_weights = torch.stack([getattr(e, 'down_proj').weight for e in experts], dim=0)
                    
                    gate_bias = torch.stack([getattr(e, 'gate_proj').bias for e in experts], dim=0) if hasattr(first_expert.gate_proj, 'bias') and first_expert.gate_proj.bias is not None else None
                    up_bias = torch.stack([getattr(e, 'up_proj').bias for e in experts], dim=0) if hasattr(first_expert.up_proj, 'bias') and first_expert.up_proj.bias is not None else None
                    down_bias = torch.stack([getattr(e, 'down_proj').bias for e in experts], dim=0) if hasattr(first_expert.down_proj, 'bias') and first_expert.down_proj.bias is not None else None
                    
                    # Batched matmul: (num_experts, hidden, in) @ (num_experts, in, n_tokens) -> (num_experts, hidden, n_tokens)
                    gate_out = torch.bmm(gate_weights, moe_input.T.unsqueeze(0).expand(num_experts, -1, -1))
                    up_out = torch.bmm(up_weights, moe_input.T.unsqueeze(0).expand(num_experts, -1, -1))
                    
                    if gate_bias is not None:
                        gate_out = gate_out + gate_bias.unsqueeze(-1)
                    if up_bias is not None:
                        up_out = up_out + up_bias.unsqueeze(-1)
                    
                    # SwiGLU activation
                    hidden = F.silu(gate_out) * up_out  # (num_experts, hidden, n_tokens)
                    
                    # Down projection
                    down_out = torch.bmm(down_weights, hidden)  # (num_experts, out, n_tokens)
                    if down_bias is not None:
                        down_out = down_out + down_bias.unsqueeze(-1)
                    
                    expert_outs = down_out.transpose(1, 2)  # (num_experts, n_tokens, out)
                    
                    # Clean up intermediate tensors to prevent memory accumulation
                    del gate_weights, up_weights, down_weights, gate_out, up_out, hidden, down_out
                    if gate_bias is not None:
                        del gate_bias
                    if up_bias is not None:
                        del up_bias
                    if down_bias is not None:
                        del down_bias
                    
                elif has_up and has_down:
                    # Simple MLP expert (up_proj, down_proj)
                    up_weights = torch.stack([getattr(e, 'up_proj').weight for e in experts], dim=0)
                    down_weights = torch.stack([getattr(e, 'down_proj').weight for e in experts], dim=0)
                    
                    up_bias = torch.stack([getattr(e, 'up_proj').bias for e in experts], dim=0) if hasattr(first_expert.up_proj, 'bias') and first_expert.up_proj.bias is not None else None
                    down_bias = torch.stack([getattr(e, 'down_proj').bias for e in experts], dim=0) if hasattr(first_expert.down_proj, 'bias') and first_expert.down_proj.bias is not None else None
                    
                    up_out = torch.bmm(up_weights, moe_input.T.unsqueeze(0).expand(num_experts, -1, -1))
                    if up_bias is not None:
                        up_out = up_out + up_bias.unsqueeze(-1)
                    
                    hidden = F.silu(up_out)
                    down_out = torch.bmm(down_weights, hidden)
                    if down_bias is not None:
                        down_out = down_out + down_bias.unsqueeze(-1)
                    
                    expert_outs = down_out.transpose(1, 2)
                    
                    # Clean up intermediate tensors
                    del up_weights, down_weights, up_out, hidden, down_out
                    if up_bias is not None:
                        del up_bias
                    if down_bias is not None:
                        del down_bias
                    
                else:
                    # Unknown structure - fall back to sequential
                    raise ValueError("Unknown expert structure")
                    
            except Exception as e:
                # Fall back to sequential for non-standard expert structures
                logger.debug(f"Batched expert forward failed, using sequential: {e}")
                expert_outs = torch.empty(
                    num_experts, n_tokens, moe_input.shape[-1],
                    device=device, dtype=moe_input.dtype
                )
                for i, expert in enumerate(experts):
                    expert_outs[i] = expert(moe_input)

        # --- Online REAP score computation (fully GPU-resident) ---
        topk_indices = torch.topk(gate_probs, k=topk, dim=-1).indices
        
        # Vectorized mask creation using broadcasting
        expert_mask = (topk_indices.unsqueeze(0) == torch.arange(num_experts, device=device).view(-1, 1, 1)).any(dim=-1)
        
        # Compute counts and REAP scores on GPU
        counts = expert_mask.sum(dim=1).float()
        expert_frequency += counts
        
        # Compute norms for all experts at once (GPU)
        norms = expert_outs.norm(dim=-1)  # (num_experts, n_tokens)
        
        # Compute REAP numerator contributions on GPU - vectorized
        gate_probs_t = gate_probs.T  # (num_experts, n_tokens)
        weighted_norms = norms * gate_probs_t
        reap_contributions = (weighted_norms * expert_mask.float()).sum(dim=1)
        
        reap_numer += reap_contributions
        reap_denom += counts
        
        # --- Keep a capped reservoir for similarity matrix (move to CPU immediately) ---
        if sim_tokens_collected < max_sim_tokens:
            n_keep = min(n_tokens, max_sim_tokens - sim_tokens_collected)
            sim_expert_outputs_cpu.append(expert_outs[:, :n_keep, :].cpu())
            sim_gate_logits_cpu.append(gate_probs[:n_keep, :].cpu())
            sim_tokens_collected += n_keep
        
        # Accumulate expert_hidden (move to CPU immediately to avoid GPU memory leak)
        # Get all active (expert_idx, token_idx) pairs at once
        active_positions = torch.nonzero(expert_mask)  # (n_active, 2) - (expert_idx, token_idx)
        if active_positions.shape[0] > 0:
            # Group positions by expert for efficient batched extraction
            expert_ids = active_positions[:, 0]
            token_ids = active_positions[:, 1]
            
            # Sort by expert_id to group consecutive entries
            sorted_indices = torch.argsort(expert_ids)
            sorted_expert_ids = expert_ids[sorted_indices]
            sorted_token_ids = token_ids[sorted_indices]
            
            # Find boundaries where expert_id changes
            unique_experts, expert_counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
            
            # Process each unique expert
            offset = 0
            for idx in range(unique_experts.shape[0]):
                expert_i = unique_experts[idx].item()
                count = expert_counts[idx].item()
                
                # Get all tokens for this expert at once
                expert_token_indices = sorted_token_ids[offset:offset + count]
                expert_state = expert_outs[expert_i, expert_token_indices]  # (n_tokens, hidden_dim)
                
                # Move to CPU immediately
                expert_state_cpu = expert_state[:min(expert_state.shape[0], max_hidden_tokens)].T.cpu()
                
                if expert_i not in expert_hidden_cpu:
                    expert_hidden_cpu[expert_i] = expert_state_cpu
                elif expert_hidden_cpu[expert_i].shape[1] < max_hidden_tokens:
                    remaining = max_hidden_tokens - expert_hidden_cpu[expert_i].shape[1]
                    if remaining > 0:
                        n_to_take = min(expert_state.shape[0], remaining)
                        expert_hidden_cpu[expert_i] = torch.cat(
                            [expert_hidden_cpu[expert_i], expert_state[:n_to_take].T.cpu()], dim=1
                        )
                
                offset += count
        
        # Explicitly delete GPU tensors to prevent accumulation
        del expert_outs, expert_mask, norms, gate_probs_t, weighted_norms, reap_contributions
        del active_positions, expert_ids, token_ids, sorted_indices, sorted_expert_ids, sorted_token_ids
        del unique_experts, expert_counts
        if 'expert_state' in locals():
            del expert_state
        if 'expert_state_cpu' in locals():
            del expert_state_cpu
        
        total_tokens += n_tokens
        
        # Periodic GPU cache clearing to prevent fragmentation (every 10 batches)
        if batch_idx % 10 == 9:
            torch.cuda.empty_cache()

    hook.remove()

    if not sim_expert_outputs_cpu:
        raise RuntimeError(
            f"No data collected for layer {layer_idx}. "
            "Check that calibration data flows through the model correctly."
        )

    # Finalize REAP scores: mean = sum / count
    reap_scores = reap_numer / (reap_denom + 1e-8)

    # Concatenate CPU buffers (already on CPU, no transfer needed)
    expert_outputs = torch.cat(sim_expert_outputs_cpu, dim=1)
    gate_logits = torch.cat(sim_gate_logits_cpu, dim=0)
    
    # expert_hidden is already on CPU
    expert_hidden = expert_hidden_cpu
    
    # Calculate statistics using NumPy for faster CPU operations (before deleting GPU tensors)
    expert_freq_np = expert_frequency.cpu().numpy()
    reap_scores_np = reap_scores.cpu().numpy()
    
    # Free GPU memory
    del sim_expert_outputs_cpu, sim_gate_logits_cpu, expert_hidden_cpu
    del reap_numer, reap_denom
    torch.cuda.empty_cache()
    
    active_experts = int(np.sum(expert_freq_np > 0))
    avg_reap_score = float(np.mean(reap_scores_np))
    max_reap_score = float(np.max(reap_scores_np))
    min_reap_score = float(np.min(reap_scores_np[expert_freq_np > 0])) if active_experts > 0 else 0.0
    
    logger.info(f"Layer {layer_idx}: Data collection complete!")
    logger.info(f"  Processed {int(total_tokens)} tokens total")
    logger.info(f"  Similarity reservoir: {expert_outputs.shape[1]} tokens")
    logger.info(f"  Active experts: {active_experts}/{num_experts}")
    logger.info(f"  REAP scores: avg={avg_reap_score:.4f}, max={max_reap_score:.4f}, min={min_reap_score:.4f}")
    logger.info(f"  Hidden activations collected for {len(expert_hidden)} experts")

    return {
        "reap_scores": reap_scores,
        "expert_outputs": expert_outputs,
        "gate_logits": gate_logits,
        "expert_hidden": expert_hidden,
        "expert_frequency": expert_frequency,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Gate weight adjustment
# ---------------------------------------------------------------------------

def prune_gate_weights(
    moe: nn.Module,
    centroid_indices: list[int],
    model_attrs: dict[str, Any],
):
    """
    Remove non-centroid expert weights from the gate/router.

    After merging, the non-centroid experts have identical weights to their
    centroids. REAM follows REAP and removes the non-centroid rows from the
    gate weights so the router only routes to the k centroid experts.

    Args:
        moe: The MoE module
        centroid_indices: List of expert indices that are centroids
        model_attrs: Model attribute name mapping
    """
    router = getattr(moe, model_attrs["router"])
    centroid_idx = torch.tensor(sorted(centroid_indices), dtype=torch.long)

    # Prune router weight rows
    if hasattr(router, "weight"):
        router.weight.data = router.weight.data[centroid_idx, :]
        if hasattr(router, "bias") and router.bias is not None:
            router.bias.data = router.bias.data[centroid_idx]
        if hasattr(router, "out_features"):
            router.out_features = len(centroid_idx)

    # Handle e_score_correction_bias (DeepSeek, GLM)
    if hasattr(router, "e_score_correction_bias"):
        router.e_score_correction_bias.data = (
            router.e_score_correction_bias.data[centroid_idx]
        )

    logger.info(
        f"Pruned gate weights: {len(centroid_idx)} experts retained "
        f"out of original router."
    )


# ---------------------------------------------------------------------------
# REAM merge for one layer
# ---------------------------------------------------------------------------

@torch.no_grad()
def ream_merge_layer(
    model: nn.Module,
    layer_idx: int,
    layer_data: dict[str, torch.Tensor],
    num_clusters: int,
    max_cluster_size: int,
    model_attrs: dict[str, Any],
    merge_method: str = "frequency_weighted_average",
    use_activation_weight_permute: bool = True,
    prune_gate: bool = True,
):
    """
    Run the full REAM merge pipeline for a single MoE layer.

    Steps:
      1. Cluster using REAM (REAP scores + pseudo-pruning + gated similarity)
      2. Permute experts using activation+weight alignment
      3. Merge expert weights (REAP-score-weighted average)
      4. Prune gate weights (remove non-centroid rows)

    Args:
        model: The full model
        layer_idx: Which transformer layer to merge
        layer_data: Output of collect_layer_data for this layer
        num_clusters: k (target number of experts after merge)
        max_cluster_size: C (max experts per group)
        model_attrs: Model attribute name mapping
        merge_method: How to combine expert weights
        use_activation_weight_permute: Whether to use activation+weight permutation
        prune_gate: Whether to prune gate weights after merging
    """
    reap_scores = layer_data["reap_scores"]
    expert_outputs = layer_data["expert_outputs"]
    gate_logits = layer_data["gate_logits"]
    expert_hidden = layer_data["expert_hidden"]
    expert_frequency = layer_data["expert_frequency"]
    total_tokens = layer_data["total_tokens"]
    num_experts = reap_scores.shape[0]

    logger.info(f"Layer {layer_idx}: Merging {num_experts} -> {num_clusters} experts")

    # Step 1: REAM clustering
    logger.info(f"Step 1: Running REAM clustering...")
    start_time = time.time()
    cluster_labels = ream_clustering(
        reap_scores=reap_scores,
        expert_outputs=expert_outputs,
        gate_logits=gate_logits,
        num_experts=num_experts,
        num_clusters=num_clusters,
        max_cluster_size=max_cluster_size,
    )
    cluster_time = time.time() - start_time
    logger.info(f"Clustering completed in {cluster_time:.2f}s")

    # Identify centroids (the expert with highest REAP score in each cluster)
    logger.info(f"Step 2: Identifying centroids...")
    centroid_indices = []
    for cluster_id in cluster_labels.unique():
        experts_in_cluster = torch.where(cluster_labels == cluster_id)[0]
        cluster_reap = reap_scores[experts_in_cluster]
        centroid_local = torch.argmax(cluster_reap)
        centroid_idx = experts_in_cluster[centroid_local].item()
        centroid_indices.append(centroid_idx)
        logger.info(f"  Cluster {cluster_id.item()}: {len(experts_in_cluster)} experts, centroid = expert {centroid_idx} (REAP score: {reap_scores[centroid_idx]:.4f})")

    # Compute expert probabilities from REAP scores (for weighted merge)
    expert_proba = reap_scores / (reap_scores.sum() + 1e-8)

    moe = get_moe(model, layer_idx)

    # Step 2 & 3: Merge with permutation alignment
    logger.info(f"Step 3: Aligning and merging experts...")
    is_fused = model_attrs.get("fused", False)
    if is_fused and use_activation_weight_permute:
        # ActivationWeightPermuter doesn't support fused experts; fall back to wm
        logger.info("Fused experts detected — using weight-matching permuter instead of activation+weight")
    permute_method = "activation_weight" if (use_activation_weight_permute and not is_fused) else "wm"
    logger.info(f"Using permutation method: {permute_method}")

    # For activation_weight permuter, we need to pass hidden activations
    if permute_method == "activation_weight":
        # Register the custom permuter with hidden activations
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_labels,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=False,
            permute=None,  # We'll handle permutation manually
            tie_tensors=False,
        )

        # Manually handle permutation with activation+weight alignment
        experts = getattr(moe, model_attrs["experts"])
        logger.info(f"Running activation+weight permutation for {len(cluster_labels.unique())} clusters...")
        permute_start = time.time()
        
        for cluster_id in cluster_labels.unique():
            expert_indices = torch.where(cluster_labels == cluster_id)[0].tolist()
            if len(expert_indices) <= 1:
                logger.info(f"  Cluster {cluster_id.item()}: {len(expert_indices)} expert (skipping permutation)")
                continue

            # Find dominant expert (highest REAP score in cluster)
            dom_expert = max(expert_indices, key=lambda idx: reap_scores[idx].item())
            logger.info(f"  Cluster {cluster_id.item()}: aligning {len(expert_indices)} experts to centroid {dom_expert}")

            # Build hidden activation dict for this cluster
            cluster_hidden = {}
            for idx in expert_indices:
                if idx in expert_hidden:
                    cluster_hidden[idx] = expert_hidden[idx].float()

            # Create and run activation+weight permuter
            permuter = ActivationWeightPermuter(
                model_attrs=model_attrs,
                expert_hidden=cluster_hidden,
            )
            permuter.permute(experts, expert_indices, dom_expert_idx=dom_expert)

        permute_time = time.time() - permute_start
        logger.info(f"Permutation completed in {permute_time:.2f}s")

        # Now run the merge (without permutation since we already did it)
        logger.info("Merging aligned experts...")
        merge_start = time.time()
        merger.merge_experts()
        merge_time = time.time() - merge_start
        logger.info(f"Merging completed in {merge_time:.2f}s")
    else:
        logger.info(f"Running weight-matching permutation and merge...")
        merge_start = time.time()
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_labels,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=False,
            permute=permute_method,
            tie_tensors=False,
        )
        merger.merge_experts()
        merge_time = time.time() - merge_start
        logger.info(f"Permutation and merging completed in {merge_time:.2f}s")

    # Step 4: Prune gate weights
    if prune_gate:
        logger.info("Step 4: Pruning gate weights...")
        prune_start = time.time()
        prune_gate_weights(moe, centroid_indices, model_attrs)
        prune_time = time.time() - prune_start
        logger.info(f"Gate pruning completed in {prune_time:.2f}s")

    # Calculate cluster distribution
    cluster_sizes = [int((cluster_labels == c).sum()) for c in cluster_labels.unique()]
    logger.info(f"Layer {layer_idx}: Merge complete!")
    logger.info(f"  Cluster distribution: {cluster_sizes}")
    logger.info(f"  Total experts merged: {num_experts} -> {num_clusters}")
    logger.info(f"  Centroids: {sorted(centroid_indices)}")
    
    return cluster_labels, centroid_indices


# ---------------------------------------------------------------------------
# Sequential REAM pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def ream_sequential_merge(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    calibration_inputs: list[torch.Tensor],
    num_clusters: int,
    max_cluster_size: int = 16,
    merge_method: str = "frequency_weighted_average",
    use_activation_weight_permute: bool = True,
    prune_gate: bool = True,
    skip_first: bool = False,
    skip_last: bool = False,
) -> dict:
    """
    Run the full REAM sequential merging pipeline.

    The key insight: after merging layer L, we recompute the activations
    for layer L+1 using the *merged* model. This means each subsequent layer
    sees updated inputs, producing better merging decisions than computing
    all activations from the original model upfront.

    Args:
        model: The MoE model to compress
        tokenizer: The model's tokenizer
        calibration_inputs: List of tokenized calibration batches
        num_clusters: k (target experts per layer after merge)
        max_cluster_size: C (max group size, default 16)
        merge_method: Expert weight merging method
        use_activation_weight_permute: Use activation+weight permutation alignment
        prune_gate: Prune gate weights after each layer merge
        skip_first: Skip merging the first MoE layer
        skip_last: Skip merging the last MoE layer

    Returns:
        Dict with cluster_labels and centroid_indices per layer
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    device = next(model.parameters()).device

    # Determine number of MoE layers
    layers = get_layers(model)
    num_layers = len(layers)
    # Find which layers are MoE layers
    moe_layer_indices = []
    for i in range(num_layers):
        moe_attr = model_attrs.get("moe_block", "mlp")
        layer_module = layers[i]
        if hasattr(layer_module, moe_attr):
            moe = getattr(layer_module, moe_attr)
            if hasattr(moe, "experts"):
                moe_layer_indices.append(i)

    if not moe_layer_indices:
        raise RuntimeError("No MoE layers found in the model.")

    logger.info(
        f"Found {len(moe_layer_indices)} MoE layers. "
        f"Merging to {num_clusters} experts per layer."
    )

    all_cluster_labels = {}
    all_centroid_indices = {}

    layers_to_process = moe_layer_indices[:]
    if skip_first and layers_to_process:
        logger.info(f"Skipping first MoE layer {layers_to_process[0]}")
        layers_to_process = layers_to_process[1:]
    if skip_last and layers_to_process:
        logger.info(f"Skipping last MoE layer {layers_to_process[-1]}")
        layers_to_process = layers_to_process[:-1]

    layer_start_time = time.time()
    for layer_idx in tqdm(layers_to_process, desc="REAM sequential merge"):
        layer_iter_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing MoE layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect fresh activations through the (possibly already merged) model
        logger.info(f"Starting data collection for layer {layer_idx}...")
        data_start = time.time()
        layer_data = collect_layer_data(
            model=model,
            calibration_inputs=calibration_inputs,
            layer_idx=layer_idx,
            model_attrs=model_attrs,
            device=device,
        )
        data_time = time.time() - data_start
        logger.info(f"Data collection completed in {data_time:.2f}s")

        # Run REAM merge for this layer
        logger.info(f"Starting REAM merge for layer {layer_idx}...")
        merge_start = time.time()
        cluster_labels, centroid_indices = ream_merge_layer(
            model=model,
            layer_idx=layer_idx,
            layer_data=layer_data,
            num_clusters=num_clusters,
            max_cluster_size=max_cluster_size,
            model_attrs=model_attrs,
            merge_method=merge_method,
            use_activation_weight_permute=use_activation_weight_permute,
            prune_gate=prune_gate,
        )
        merge_time = time.time() - merge_start

        all_cluster_labels[layer_idx] = cluster_labels
        all_centroid_indices[layer_idx] = centroid_indices

        # Free memory
        del layer_data
        gc.collect()
        torch.cuda.empty_cache()
        
        layer_iter_time = time.time() - layer_iter_start
        logger.info(f"Layer {layer_idx} completed in {layer_iter_time:.2f}s (data: {data_time:.2f}s, merge: {merge_time:.2f}s)")
        logger.info(f"{'='*60}")
    
    total_time = time.time() - layer_start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"REAM sequential merge completed!")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Processed {len(layers_to_process)} MoE layers")
    logger.info(f"{'='*60}")

    return {
        "cluster_labels": all_cluster_labels,
        "centroid_indices": all_centroid_indices,
    }


# ---------------------------------------------------------------------------
# Calibration data loading (mixed: c4 + math + coding + roleplay)
# ---------------------------------------------------------------------------

def load_ream_calibration_data(
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    c4_samples: int = 512,
    math_samples: int = 512,
    code_samples: int = 512,
    roleplay_samples: int = 512,
    c4_max_tokens: int = 128,
    math_max_tokens: int = 512,
    code_max_tokens: int = 512,
    roleplay_max_tokens: int = 512,
    seed: int = 42,
) -> list[torch.Tensor]:
    """
    Load the mixed calibration dataset used in REAM experiments.

    Table 1 from the blog post:
      - General: allenai/c4/en (512 samples, 128 max tokens, ~8% of data)
      - Math: AI-MO/NuminaMath-1.5 cn_k12+olympiads (1024 samples, 512 max tokens, ~68%)
      - Coding: bigcode/the-stack-smol (512 samples, 512 max tokens, ~24%)

    Args:
        tokenizer: The model tokenizer
        max_length: Maximum sequence length for packing
        c4_samples: Number of C4 samples
        math_samples: Number of math samples
        code_samples: Number of coding samples
        c4_max_tokens: Max tokens per C4 sample
        math_max_tokens: Max tokens per math sample
        code_max_tokens: Max tokens per code sample
        seed: Random seed

    Returns:
        List of tokenized input tensors
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info(f"Loading REAM mixed calibration data with seed {seed}...")
    logger.info(f"Dataset mix: C4={c4_samples}, Math={math_samples}, Code={code_samples}")
    logger.info(f"Token limits: C4={c4_max_tokens}, Math={math_max_tokens}, Code={code_max_tokens}")

    all_inputs = []

    # 1. C4 (general text)
    logger.info("Loading C4 calibration data...")
    try:
        c4_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
        c4_ds = load_dataset("json", data_files={"train": c4_url}, split="train", streaming=False)
        c4_indices = random.sample(range(len(c4_ds)), min(c4_samples, len(c4_ds)))
        for idx in c4_indices:
            text = c4_ds[idx]["text"]
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=c4_max_tokens)
            all_inputs.append(tokens["input_ids"])
        logger.info(f"Loaded {len(c4_indices)} C4 samples")
    except Exception as e:
        logger.warning(f"Failed to load C4 data: {e}. Skipping.")

    # 2. Math (NuminaMath)
    logger.info("Loading NuminaMath calibration data...")
    try:
        math_ds = load_dataset(
            "AI-MO/NuminaMath-1.5",
            split="train",
            streaming=False,
        )
        # Filter for cn_k12 and olympiads subsets
        math_ds = math_ds.filter(
            lambda x: x.get("source", "") in ["cn_k12", "olympiads"]
        )
        math_indices = random.sample(range(len(math_ds)), min(math_samples, len(math_ds)))
        for idx in math_indices:
            text = math_ds[idx].get("problem", "") + " " + math_ds[idx].get("solution", "")
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=math_max_tokens)
            all_inputs.append(tokens["input_ids"])
        logger.info(f"Loaded {len(math_indices)} NuminaMath samples")
    except Exception as e:
        logger.warning(f"Failed to load NuminaMath data: {e}. Skipping.")

    # 3. Code (the-stack-smol)
    logger.info("Loading the-stack-smol calibration data...")
    try:
        code_ds = load_dataset(
            "bigcode/the-stack-smol",
            "default",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )
        code_indices = random.sample(range(len(code_ds)), min(code_samples, len(code_ds)))
        for idx in code_indices:
            text = code_ds[idx].get("content", "")
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=code_max_tokens)
            all_inputs.append(tokens["input_ids"])
        logger.info(f"Loaded {len(code_indices)} code samples")
    except Exception as e:
        logger.warning(f"Failed to load the-stack-smol data: {e}. Skipping.")

    # 4. Creative/Roleplay (bluemoon_roleplay_chat_data - 300k roleplay messages)
    logger.info("Loading roleplay/creative writing calibration data...")
    try:
        roleplay_ds = load_dataset(
            "rickRossie/bluemoon_roleplay_chat_data_300k_messages",
            split="train",
            streaming=False,
            trust_remote_code=True,
        )
        roleplay_indices = random.sample(range(len(roleplay_ds)), min(roleplay_samples, len(roleplay_ds)))
        for idx in roleplay_indices:
            text = roleplay_ds[idx].get("message", "")
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=roleplay_max_tokens)
            all_inputs.append(tokens["input_ids"])
        logger.info(f"Loaded {len(roleplay_indices)} roleplay samples")
    except Exception as e:
        logger.warning(f"Failed to load roleplay data: {e}. Skipping.")

    if not all_inputs:
        raise RuntimeError("No calibration data loaded. Check dataset availability.")

    # Shuffle
    random.shuffle(all_inputs)
    
    # Calculate total tokens
    total_tokens = sum(input_ids.shape[1] for input_ids in all_inputs)
    avg_tokens = total_tokens / len(all_inputs) if all_inputs else 0
    
    logger.info(f"Calibration data loading complete!")
    logger.info(f"  Total samples: {len(all_inputs)}")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Average tokens per sample: {avg_tokens:.1f}")
    logger.info(f"  Sample shape example: {all_inputs[0].shape if all_inputs else 'N/A'}")
    
    return all_inputs


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    REAM main entry point.

    Runs the full pipeline:
      1. Load model and tokenizer
      2. Load mixed calibration data
      3. Run sequential REAM merging
      4. Save compressed model
      5. Optionally evaluate
    """
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        kd_args,
        eval_args,
        merge_args,
    ) = parse_args()
    set_seed(reap_args.seed)

    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # Load model
    logger.info(f"Loading model: {model_args.model_name}")
    model_name = patched_model_map(model_args.model_name)
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"Loading model from {model_name}...")
    model_start = time.time()
    model = load_model_text_only(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model_time = time.time() - model_start
    logger.info(f"Model loaded in {model_time:.2f}s")
    
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    logger.info(f"Model class: {model.__class__.__name__}")
    logger.info(f"Model attributes: {model_attrs}")

    # Load calibration data
    logger.info("Loading REAM mixed calibration data...")
    if ds_args.dataset_name == "ream_mixed":
        calibration_inputs = load_ream_calibration_data(
            tokenizer=tokenizer,
            max_length=obs_args.model_max_length or 2048,
            seed=reap_args.seed,
        )
    else:
        # Fall back to single dataset mode
        from ream.main import record_activations
        logger.info(f"Using single dataset: {ds_args.dataset_name}")
        # Use the standard observer pipeline to get calibration data
        from ream.data import DATASET_REGISTRY
        raw_ds = load_dataset(ds_args.dataset_name, split=ds_args.split)
        proc_cls = DATASET_REGISTRY.get(ds_args.dataset_name)
        if proc_cls is None:
            raise ValueError(f"No DatasetProcessor for '{ds_args.dataset_name}'")
        processor = proc_cls(
            dataset=raw_ds,
            tokenizer=tokenizer,
            max_input_len=obs_args.model_max_length,
            split=ds_args.split,
            split_by_category=False,
        )
        category_data = processor.get_processed_dataset(obs_args.samples_per_category)
        calibration_inputs = []
        for cat_data in category_data.values():
            calibration_inputs.extend(cat_data)

    # Compute num_clusters
    logger.info("Analyzing model structure...")
    experts_sample = getattr(get_moe(model, 0), model_attrs["experts"])
    num_experts_sample = get_num_experts(experts_sample, model_attrs)
    logger.info(f"Found {num_experts_sample} experts per MoE layer")
    
    num_clusters = cluster_args.num_clusters
    if num_clusters is None:
        if cluster_args.compression_ratio is None:
            raise ValueError("Either num_clusters or compression_ratio must be set.")
        num_clusters = int(num_experts_sample * (1 - cluster_args.compression_ratio))
        logger.info(f"Using compression ratio {cluster_args.compression_ratio} -> {num_clusters} clusters")

    max_cluster_size = cluster_args.max_cluster_size or 16

    logger.info(f"REAM configuration:")
    logger.info(f"  Original experts per layer: {num_experts_sample}")
    logger.info(f"  Target clusters per layer: {num_clusters}")
    logger.info(f"  Compression: {num_experts_sample} -> {num_clusters} ({(1 - num_clusters/num_experts_sample)*100:.1f}% reduction)")
    logger.info(f"  Max cluster size: {max_cluster_size}")
    logger.info(f"  Merge method: {merge_args.merge_method}")
    logger.info(f"  Permutation method: {'activation+weight' if merge_args.permute != 'wm' else 'weight-matching'}")
    logger.info(f"  Skip first layer: {merge_args.skip_first}")
    logger.info(f"  Skip last layer: {merge_args.skip_last}")

    # Run REAM sequential merge
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting REAM sequential merge pipeline")
    logger.info(f"{'='*60}")
    merge_start = time.time()
    result = ream_sequential_merge(
        model=model,
        tokenizer=tokenizer,
        calibration_inputs=calibration_inputs,
        num_clusters=num_clusters,
        max_cluster_size=max_cluster_size,
        merge_method=merge_args.merge_method,
        use_activation_weight_permute=(merge_args.permute != "wm"),
        prune_gate=True,
        skip_first=merge_args.skip_first,
        skip_last=merge_args.skip_last,
    )
    merge_total_time = time.time() - merge_start
    logger.info(f"\n{'='*60}")
    logger.info(f"REAM merge pipeline completed in {merge_total_time:.2f}s")
    logger.info(f"{'='*60}")

    cluster_labels = result["cluster_labels"]

    # Save model
    logger.info(f"\nSaving merged model...")
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
    )
    logger.info(f"Model will be saved to: {merged_model_dir}")

    save_start = time.time()
    merged_model_dir = save_merged_model(
        model, tokenizer, merged_model_dir, safe_serialization=True
    )
    save_time = time.time() - save_start
    logger.info(f"Model saved in {save_time:.2f}s")

    # Save clustering info
    logger.info("Saving clustering information...")
    cluster_analysis_dir = merged_model_dir / "clusters"
    cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
        pickle.dump(cluster_labels, f)
    with open(cluster_analysis_dir / "centroids.pkl", "wb") as f:
        pickle.dump(result["centroid_indices"], f)
    logger.info(f"Clustering info saved to {cluster_analysis_dir}")

    # Smoke test
    if reap_args.smoke_test:
        logger.info("Running smoke test...")
        try:
            smoke_test(model, tokenizer)
        except Exception as e:
            logger.error(f"Smoke test failed: {e}")

    # Save args
    dump_args_to_yaml(
        merged_model_dir, reap_args, model_args, ds_args,
        obs_args, cluster_args, kd_args, eval_args, merge_args,
    )

    # Evaluate
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del cluster_labels
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = merged_model_dir
        run_evaluate(model_args, merged_model_dir / "eval", eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
