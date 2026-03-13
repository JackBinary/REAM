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
    moe = get_moe(model, layer_idx)
    experts = getattr(moe, model_attrs["experts"])
    router = getattr(moe, model_attrs["router"])
    is_fused = model_attrs.get("fused", False)
    num_experts = get_num_experts(experts, model_attrs)

    # Determine topk from model config (handle nested text_config for VL models)
    topk_key = model_attrs.get("num_experts_per_tok", "num_experts_per_tok")
    cfg = getattr(model.config, "text_config", model.config)
    topk = getattr(cfg, topk_key, 8)

    # --- Online accumulators for REAP scores (no large tensor storage) ---
    reap_numer = torch.zeros(num_experts)   # running sum of (norm * gate_val)
    reap_denom = torch.zeros(num_experts)   # running count of routed tokens
    expert_frequency = torch.zeros(num_experts)
    expert_hidden: dict[int, torch.Tensor] = {}

    # --- Capped reservoir for similarity computation ---
    sim_expert_outputs: list[torch.Tensor] = []
    sim_gate_logits: list[torch.Tensor] = []
    sim_tokens_collected = 0
    total_tokens = 0

    # Hook to capture the input to the MoE block
    captured = {}

    def capture_moe_input(module, args, output):
        if isinstance(args, tuple) and len(args) > 0:
            captured["moe_input"] = args[0].detach()

    hook = moe.register_forward_hook(capture_moe_input)

    for batch_input in calibration_inputs:
        if isinstance(batch_input, torch.Tensor):
            inputs = batch_input.to(device)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            model_input = {"input_ids": inputs}
        elif isinstance(batch_input, dict):
            model_input = {k: v.to(device) for k, v in batch_input.items()}
        else:
            continue

        captured.clear()
        try:
            _ = model(**model_input)
        except Exception as e:
            logger.warning(f"Forward pass failed for batch: {e}")
            continue

        if "moe_input" not in captured:
            continue

        moe_input = captured["moe_input"]
        if moe_input.dim() == 3:
            moe_input = moe_input.view(-1, moe_input.shape[-1])
        n_tokens = moe_input.shape[0]

        # Compute gate logits manually
        with torch.no_grad():
            gate_out = router(moe_input)  # (n_tokens, N)
            # Some routers (e.g. Qwen3.5) return a tuple (logits, scores, indices)
            if isinstance(gate_out, tuple):
                gate_out = gate_out[0]
            gate_probs = F.softmax(gate_out, dim=-1)

        # Compute expert outputs for all experts
        expert_outs = torch.zeros(
            num_experts, n_tokens, moe_input.shape[-1],
            device=device, dtype=moe_input.dtype
        )
        if is_fused:
            for i in range(num_experts):
                expert_outs[i] = fused_expert_forward(experts, i, moe_input)
        else:
            for i, expert in enumerate(experts):
                expert_outs[i] = expert(moe_input)

        # Move to CPU once, then free GPU copy
        expert_outs_cpu = expert_outs.cpu()
        gate_probs_cpu = gate_probs.cpu()
        del expert_outs

        # --- Online REAP score computation ---
        topk_values, topk_indices = torch.topk(gate_probs_cpu, k=topk, dim=-1)
        for i in range(num_experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            count = mask.sum().float()
            expert_frequency[i] += count
            expert_state = expert_outs_cpu[i, mask]
            gate_vals = gate_probs_cpu[mask, i]
            norms = expert_state.norm(dim=-1)
            reap_numer[i] += (norms * gate_vals).sum()
            reap_denom[i] += count

            # Accumulate expert_hidden (capped per expert)
            if i not in expert_hidden:
                expert_hidden[i] = expert_state.T[:, :max_hidden_tokens]
            elif expert_hidden[i].shape[1] < max_hidden_tokens:
                remaining = max_hidden_tokens - expert_hidden[i].shape[1]
                expert_hidden[i] = torch.cat(
                    [expert_hidden[i], expert_state[:remaining].T], dim=1
                )

        # --- Keep a capped reservoir for similarity matrix ---
        if sim_tokens_collected < max_sim_tokens:
            n_keep = min(n_tokens, max_sim_tokens - sim_tokens_collected)
            sim_expert_outputs.append(expert_outs_cpu[:, :n_keep, :])
            sim_gate_logits.append(gate_probs_cpu[:n_keep, :])
            sim_tokens_collected += n_keep

        del expert_outs_cpu, gate_probs_cpu
        total_tokens += n_tokens

    hook.remove()

    if not sim_expert_outputs:
        raise RuntimeError(
            f"No data collected for layer {layer_idx}. "
            "Check that calibration data flows through the model correctly."
        )

    # Finalize REAP scores: mean = sum / count
    reap_scores = reap_numer / (reap_denom + 1e-8)

    # Concatenate the capped reservoir
    expert_outputs = torch.cat(sim_expert_outputs, dim=1)
    gate_logits = torch.cat(sim_gate_logits, dim=0)

    logger.info(
        f"Layer {layer_idx}: collected REAP scores over {int(total_tokens)} tokens, "
        f"similarity reservoir {expert_outputs.shape[1]} tokens"
    )

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
    cluster_labels = ream_clustering(
        reap_scores=reap_scores,
        expert_outputs=expert_outputs,
        gate_logits=gate_logits,
        num_experts=num_experts,
        num_clusters=num_clusters,
        max_cluster_size=max_cluster_size,
    )

    # Identify centroids (the expert with highest REAP score in each cluster)
    centroid_indices = []
    for cluster_id in cluster_labels.unique():
        experts_in_cluster = torch.where(cluster_labels == cluster_id)[0]
        cluster_reap = reap_scores[experts_in_cluster]
        centroid_local = torch.argmax(cluster_reap)
        centroid_indices.append(experts_in_cluster[centroid_local].item())

    # Compute expert probabilities from REAP scores (for weighted merge)
    expert_proba = reap_scores / (reap_scores.sum() + 1e-8)

    moe = get_moe(model, layer_idx)

    # Step 2 & 3: Merge with permutation alignment
    is_fused = model_attrs.get("fused", False)
    if is_fused and use_activation_weight_permute:
        # ActivationWeightPermuter doesn't support fused experts; fall back to wm
        logger.info("Fused experts detected — using weight-matching permuter instead of activation+weight")
    permute_method = "activation_weight" if (use_activation_weight_permute and not is_fused) else "wm"

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
        for cluster_id in cluster_labels.unique():
            expert_indices = torch.where(cluster_labels == cluster_id)[0].tolist()
            if len(expert_indices) <= 1:
                continue

            # Find dominant expert (highest REAP score in cluster)
            dom_expert = max(expert_indices, key=lambda idx: reap_scores[idx].item())

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

        # Now run the merge (without permutation since we already did it)
        merger.merge_experts()
    else:
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

    # Step 4: Prune gate weights
    if prune_gate:
        prune_gate_weights(moe, centroid_indices, model_attrs)

    logger.info(f"Layer {layer_idx}: Merge complete. Cluster sizes: "
                f"{[int((cluster_labels == c).sum()) for c in cluster_labels.unique()]}")

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

    for layer_idx in tqdm(layers_to_process, desc="REAM sequential merge"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing MoE layer {layer_idx}")
        logger.info(f"{'='*60}")

        # Collect fresh activations through the (possibly already merged) model
        layer_data = collect_layer_data(
            model=model,
            calibration_inputs=calibration_inputs,
            layer_idx=layer_idx,
            model_attrs=model_attrs,
            device=device,
        )

        # Run REAM merge for this layer
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

        all_cluster_labels[layer_idx] = cluster_labels
        all_centroid_indices[layer_idx] = centroid_indices

        # Free memory
        del layer_data
        gc.collect()
        torch.cuda.empty_cache()

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
    math_samples: int = 1024,
    code_samples: int = 512,
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
        roleplay_indices = random.sample(range(len(roleplay_ds)), min(code_samples, len(roleplay_ds)))
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
    logger.info(f"Total calibration samples: {len(all_inputs)}")
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
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_model_text_only(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    model_attrs = MODEL_ATTRS[model.__class__.__name__]

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
    experts_sample = getattr(get_moe(model, 0), model_attrs["experts"])
    num_experts_sample = get_num_experts(experts_sample, model_attrs)
    num_clusters = cluster_args.num_clusters
    if num_clusters is None:
        if cluster_args.compression_ratio is None:
            raise ValueError("Either num_clusters or compression_ratio must be set.")
        num_clusters = int(num_experts_sample * (1 - cluster_args.compression_ratio))

    max_cluster_size = cluster_args.max_cluster_size or 16

    logger.info(f"REAM config: {num_experts_sample} -> {num_clusters} experts, "
                f"max_cluster_size={max_cluster_size}")

    # Run REAM sequential merge
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

    cluster_labels = result["cluster_labels"]

    # Save model
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
    )

    merged_model_dir = save_merged_model(
        model, tokenizer, merged_model_dir, safe_serialization=True
    )

    # Save clustering info
    cluster_analysis_dir = merged_model_dir / "clusters"
    cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
        pickle.dump(cluster_labels, f)
    with open(cluster_analysis_dir / "centroids.pkl", "wb") as f:
        pickle.dump(result["centroid_indices"], f)

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
