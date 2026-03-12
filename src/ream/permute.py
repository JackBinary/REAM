from typing import Any
from abc import ABC, abstractmethod
import copy
import gc

import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

import logging

logger = logging.getLogger(__name__)


def _weight_match_dist(a, dom):
    return torch.cdist(a, dom, p=2) ** 2


def assert_invariance(permuted_expert, orig_expert, model_attrs):
    up_proj = getattr(permuted_expert, model_attrs["up_proj"])
    inp = torch.rand(
        (1, up_proj.weight.shape[1]),
        dtype=up_proj.weight.dtype,
        device=up_proj.weight.device,
    )
    out1 = permuted_expert(inp)
    out2 = orig_expert(inp)
    if not torch.allclose(out1, out2, atol=1e-2):
        logger.warning(
            "Output of permuted expert should match original expert. "
            "Sum(abs(out1 - out2)) = {}".format(torch.sum(torch.abs(out1 - out2)))
        )


def assert_improved_weight_dist(permuted_expert, orig_expert, dom_expert, model_attrs):
    orig_dist = 0
    permuted_dist = 0
    for attr in ["up_proj", "gate_proj", "down_proj"]:
        permuted_weight = getattr(permuted_expert, model_attrs[attr]).weight
        orig_weight = getattr(orig_expert, model_attrs[attr]).weight
        dom_weight = getattr(dom_expert, model_attrs[attr]).weight
        orig_dist += _weight_match_dist(orig_weight, dom_weight).sum().item()
        permuted_dist += _weight_match_dist(permuted_weight, dom_weight).sum().item()
    if not permuted_dist < orig_dist:
        logger.warning(
            "Permuted expert should have a lower distance to the original expert than the dominant expert. ({}) > ({})".format(
                permuted_dist, orig_dist
            )
        )
    return permuted_dist, orig_dist


def assert_not_equal(permuted_expert, orig_expert, model_attrs):
    for attr in ["up_proj", "gate_proj", "down_proj"]:
        permuted_weight = getattr(permuted_expert, model_attrs[attr]).weight
        orig_weight = getattr(orig_expert, model_attrs[attr]).weight
        if torch.equal(permuted_weight, orig_weight):
            logger.warning(
                f"Permuted expert's {attr} weights should not be equal to the original expert's weights."
            )


class ExpertPermuter(ABC):
    def __init__(self, model_attrs: dict[str, Any]):
        self.model_attrs = model_attrs
        self.fused = False
        if model_attrs.get("fused", False):
            self.fused = True

    @torch.no_grad()
    def permute(
        self, experts: list[nn.Module], expert_indices: list[int], dom_expert_idx: int
    ):
        if self.fused:
            self._fused_permute(experts, expert_indices, dom_expert_idx)
        else:
            self._permute(experts, expert_indices, dom_expert_idx)

    @abstractmethod
    def _permute(
        self,
        experts: list[nn.Module],
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """
        Abstract method to permute experts to a canonical order.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _fused_permute(
        self,
        experts: nn.Module,
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """
        Abstract method to permute experts in a fused model.
        Must be implemented by subclasses.
        """
        pass

    def _run_assertions(
        self, permuted_expert, orig_expert, cost_matrix_np, row_ind, col_ind
    ):
        try:
            assert_invariance(permuted_expert, orig_expert, self.model_attrs)
            assert_not_equal(permuted_expert, orig_expert, self.model_attrs)
            permuted_cost = cost_matrix_np[row_ind, col_ind].sum()
            original_cost = np.trace(cost_matrix_np)
            assert permuted_cost <= original_cost, (
                f"Permuted cost {permuted_cost} should be less than or equal to original cost {original_cost}."
            )
        except AssertionError as e:
            logger.warning(f"Assertion failed during permutation: {e}")
            print(e)
            pass


class WeightMatchingPermuter(ExpertPermuter):
    def _permute(
        self,
        experts: list[nn.Module],
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Permutes experts using weight matching."""
        for expert_idx in expert_indices:
            if expert_idx == dom_expert_idx:
                continue
            expert = experts[expert_idx]
            orig_expert = copy.deepcopy(expert)
            cost_matrix = self._expert_cost_matrix(
                expert, experts[dom_expert_idx], self.model_attrs
            )
            cost_matrix_np = cost_matrix.cpu().to(torch.float16).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            permutation = torch.tensor(col_ind, dtype=torch.long).argsort()
            self.apply_permutation(expert, permutation, self.model_attrs)
            logger.debug("Checking expert %d...", expert_idx)
            self._run_assertions(expert, orig_expert, cost_matrix_np, row_ind, col_ind)

    def _expert_cost_matrix(self, expert, dominant_expert, model_attrs):
        """
        Computes the L1 distance between two experts.
        """
        up = _weight_match_dist(
            getattr(expert, model_attrs["up_proj"]).weight,
            getattr(dominant_expert, model_attrs["up_proj"]).weight,
        )
        gate = _weight_match_dist(
            getattr(expert, model_attrs["gate_proj"]).weight,
            getattr(dominant_expert, model_attrs["gate_proj"]).weight,
        )
        down = _weight_match_dist(
            getattr(expert, model_attrs["down_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["down_proj"]).weight.T,
        )
        return up + gate + down

    def apply_permutation(self, expert, permutation, model_attrs):
        """
        Applies a permutation to the weights of an expert.
        """
        up_proj = getattr(expert, model_attrs["up_proj"])
        gate_proj = getattr(expert, model_attrs["gate_proj"])
        down_proj = getattr(expert, model_attrs["down_proj"])

        up_proj.weight.data = up_proj.weight[permutation]
        gate_proj.weight.data = gate_proj.weight[permutation]
        down_proj.weight.data = down_proj.weight[:, permutation]

    @staticmethod
    def _detect_fused_layout(gate_up_proj, down_proj):
        """
        Detect the fused gate_up_proj layout by matching dimensions against down_proj.

        Fused experts concatenate gate and up projections (equal size) into one tensor.
        The intermediate dimension (d_inter) appears as 2*d_inter in gate_up_proj.

        Returns:
            (d_inter, split_axis, perm_axis_gate_up, perm_axis_down)
            split_axis: which dim of gate_up_proj[expert] holds 2*d_inter (0 or 1)
            perm_axis_gate_up: the axis to permute in gate_up_proj[expert]
            perm_axis_down: the axis to permute in down_proj[expert]
        """
        # gate_up_proj: (N, A, B)  down_proj: (N, C, D)
        # One of {A, B} equals 2*d_inter, one of {C, D} equals d_inter
        A, B = gate_up_proj.shape[1], gate_up_proj.shape[2]
        C, D = down_proj.shape[1], down_proj.shape[2]

        # Llama4 layout: gate_up = (N, d_model, 2*d_inter), down = (N, d_inter, d_model)
        #   → B == 2*C, split last dim (axis=1 of 2D slice), perm cols of gate_up, perm rows of down
        if B == 2 * C:
            return C, 1, 1, 0
        # Transposed layout (Qwen3.5): gate_up = (N, 2*d_inter, d_model), down = (N, d_model, d_inter)
        #   → A == 2*D, split first dim (axis=0 of 2D slice), perm rows of gate_up, perm cols of down
        if A == 2 * D:
            return D, 0, 0, 1
        # Try remaining combos
        if B == 2 * D:
            return D, 1, 1, 1
        if A == 2 * C:
            return C, 0, 0, 0

        raise ValueError(
            f"Cannot detect fused layout: gate_up_proj per-expert shape ({A}, {B}), "
            f"down_proj per-expert shape ({C}, {D}). Expected one dim to be 2x the "
            f"intermediate size found in down_proj."
        )

    def _fused_permute(
        self,
        experts: nn.Module,
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Permutes experts in a fused model using weight matching.

        Handles both Llama4-style layout:
            gate_up_proj = (num_experts, hidden_size, 2 * expert_dim)
            down_proj    = (num_experts, expert_dim, hidden_size)
        and Qwen3.5-style (transposed) layout:
            gate_up_proj = (num_experts, 2 * expert_dim, hidden_size)
            down_proj    = (num_experts, hidden_size, expert_dim)
        """
        if len(expert_indices) == 1:
            return  # No permutation needed if only one expert
        orig_experts = copy.deepcopy(experts).cpu()
        up_gate_proj_param = getattr(experts, self.model_attrs["up_proj"])
        down_proj_param = getattr(experts, self.model_attrs["down_proj"])
        device = up_gate_proj_param.device

        up_gate_proj = up_gate_proj_param.data.cpu()
        down_proj = down_proj_param.data.cpu()

        d_inter, split_axis, perm_axis_gu, perm_axis_down = self._detect_fused_layout(
            up_gate_proj, down_proj
        )
        logger.debug(
            f"Fused layout: d_inter={d_inter}, split_axis={split_axis}, "
            f"gate_up perm axis={perm_axis_gu}, down perm axis={perm_axis_down}"
        )

        def _split_gate_up(tensor_2d):
            """Split a single expert's gate_up tensor into gate and up halves."""
            if split_axis == 1:
                return tensor_2d[:, :d_inter], tensor_2d[:, d_inter:]
            else:
                return tensor_2d[:d_inter, :], tensor_2d[d_inter:, :]

        def _get_cost_vectors(gate, up, down):
            """Get per-neuron vectors for cost matrix computation.
            Returns tensors of shape (d_inter, features) suitable for cdist."""
            if perm_axis_gu == 1:
                # Permuting columns of gate_up → each neuron is a column → transpose
                g_vecs, u_vecs = gate.T, up.T
            else:
                # Permuting rows of gate_up → each neuron is a row → use directly
                g_vecs, u_vecs = gate, up
            if perm_axis_down == 0:
                d_vecs = down
            else:
                d_vecs = down.T
            return g_vecs, u_vecs, d_vecs

        dom_gate, dom_up = _split_gate_up(up_gate_proj[dom_expert_idx])
        dom_down = down_proj[dom_expert_idx]
        dom_g, dom_u, dom_d = _get_cost_vectors(dom_gate, dom_up, dom_down)

        for expert_idx in expert_indices:
            if expert_idx == dom_expert_idx:
                continue
            this_gate, this_up = _split_gate_up(up_gate_proj[expert_idx])
            this_down = down_proj[expert_idx]
            this_g, this_u, this_d = _get_cost_vectors(this_gate, this_up, this_down)

            gate_cost = _weight_match_dist(this_g, dom_g)
            up_cost = _weight_match_dist(this_u, dom_u)
            down_cost = _weight_match_dist(this_d, dom_d)
            cost_matrix = up_cost + gate_cost + down_cost
            del up_cost, gate_cost, down_cost
            cost_matrix_np = cost_matrix.to(torch.float16).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            permutation = torch.tensor(col_ind, dtype=torch.long).argsort()

            # Apply permutation to gate_up_proj
            if split_axis == 1:
                up_gate_proj[expert_idx, :, :d_inter] = this_gate[:, permutation]
                up_gate_proj[expert_idx, :, d_inter:] = this_up[:, permutation]
            else:
                up_gate_proj[expert_idx, :d_inter, :] = this_gate[permutation, :]
                up_gate_proj[expert_idx, d_inter:, :] = this_up[permutation, :]

            # Apply permutation to down_proj
            if perm_axis_down == 0:
                down_proj[expert_idx] = this_down[permutation, :]
            else:
                down_proj[expert_idx] = this_down[:, permutation]

            del this_down, this_gate, this_up
            gc.collect()
            torch.cuda.empty_cache()

        up_gate_proj_param.data = up_gate_proj.to(device)
        down_proj_param.data = down_proj.to(device)

        # Check permutation invariance and weights changed
        input = torch.rand(
            (up_gate_proj.shape[0], up_gate_proj.shape[1]),
            dtype=up_gate_proj.dtype,
            device=device,
        )
        orig_out = orig_experts.to(device)(input)
        permuted_out = experts(input)
        if not torch.allclose(orig_out, permuted_out, atol=1e-2):
            logger.warning(
                "Output of permuted experts should match original expert. "
                "Sum(abs(out1 - out2)) = {}".format(
                    torch.sum(torch.abs(orig_out - permuted_out))
                )
            )
        del input
        del orig_out
        del permuted_out

        gu_attr = self.model_attrs["up_proj"]
        dp_attr = self.model_attrs["down_proj"]
        if torch.allclose(
            getattr(orig_experts, gu_attr).cpu(), getattr(experts, gu_attr).cpu()
        ) or torch.allclose(
            getattr(orig_experts, dp_attr).cpu(), getattr(experts, dp_attr).cpu()
        ):
            logger.warning(
                "Permuted experts' weights should not be equal to the original experts'"
                " weights."
            )


class DirectAlignmentPermuter(ExpertPermuter):
    def _permute(
        self,
        experts: list[nn.Module],
        dom_expert_idx: int,
    ):
        for expert in experts:
            if expert is not experts[dom_expert_idx]:
                cost_matrix = self._expert_cost_matrix(
                    expert, experts[dom_expert_idx], self.model_attrs
                )
                cost_matrix_np = cost_matrix.cpu().to(torch.float16).numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
                permutation = torch.tensor(col_ind, dtype=torch.long)
                self.apply_permutation_direct_alignment(
                    expert, permutation, self.model_attrs
                )

    def _l2_dist(self, a, dom):
        return torch.cdist(a, dom, p=2) ** 2

    def _expert_cost_matrix(self, expert, dominant_expert, model_attrs):
        """
        Computes the L1 distance between two experts.
        """
        up = self._l2_dist(
            getattr(expert, model_attrs["up_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["up_proj"]).weight.T,
        )
        gate = self._l2_dist(
            getattr(expert, model_attrs["gate_proj"]).weight.T,
            getattr(dominant_expert, model_attrs["gate_proj"]).weight.T,
        )
        down = self._l2_dist(
            getattr(expert, model_attrs["down_proj"]).weight,
            getattr(dominant_expert, model_attrs["down_proj"]).weight,
        )
        return up + gate + down

    def apply_permutation_direct_alignment(self, expert, permutation, model_attrs):
        """
        Applies a permutation to the weights of an expert.
        """
        up_proj = getattr(expert, model_attrs["up_proj"])
        gate_proj = getattr(expert, model_attrs["gate_proj"])
        down_proj = getattr(expert, model_attrs["down_proj"])

        up_proj.weight.data = up_proj.weight.data[:, permutation]
        gate_proj.weight.data = gate_proj.weight.data[:, permutation]
        down_proj.weight.data = down_proj.weight.data[permutation, :]


class ActivationWeightPermuter(ExpertPermuter):
    """
    REAM-style permutation alignment using both hidden activations and weights.

    From the REAM blog post (Step 5):
      cost1 = cdist(hidden[0], hidden[i])     # activation-based
      cost2 = cdist(weights[0], weights[i])    # weight-based
      perm = linear_sum_assignment(cost1 + cost2)

    This combines information from how experts behave (activations) with
    their parameter structure (weights) for better alignment before merging.
    """

    def __init__(self, model_attrs: dict[str, Any], expert_hidden: dict[int, torch.Tensor] | None = None):
        """
        Args:
            model_attrs: Model attribute names dict.
            expert_hidden: Dict mapping expert_index -> (d, n) hidden activation tensor.
                           d = bottleneck dim, n = num tokens.
                           If None, falls back to weight-only matching.
        """
        super().__init__(model_attrs)
        self.expert_hidden = expert_hidden or {}

    def _permute(
        self,
        experts: list[nn.Module],
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Permutes experts using combined activation + weight matching."""
        import copy

        dom_expert = experts[dom_expert_idx]
        dom_weights = self._get_concat_weights(dom_expert)

        # Get dominant expert hidden activations if available
        dom_hidden = self.expert_hidden.get(dom_expert_idx)

        for expert_idx in expert_indices:
            if expert_idx == dom_expert_idx:
                continue

            expert = experts[expert_idx]
            orig_expert = copy.deepcopy(expert)
            expert_weights = self._get_concat_weights(expert)

            # Weight-based cost matrix
            cost_weight = torch.cdist(dom_weights, expert_weights, p=2)

            # Activation-based cost matrix (if available)
            expert_hidden = self.expert_hidden.get(expert_idx)
            if dom_hidden is not None and expert_hidden is not None:
                cost_activation = torch.cdist(dom_hidden, expert_hidden, p=2)
                cost_matrix = cost_activation + cost_weight
            else:
                cost_matrix = cost_weight

            cost_matrix_np = cost_matrix.cpu().to(torch.float32).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            permutation = torch.tensor(col_ind, dtype=torch.long).argsort()

            # Apply permutation to hidden dimension (rows of up/gate, cols of down)
            WeightMatchingPermuter.apply_permutation(
                WeightMatchingPermuter, expert, permutation, self.model_attrs
            )

            logger.debug("Checking expert %d with activation+weight permutation...", expert_idx)
            self._run_assertions(expert, orig_expert, cost_matrix_np, row_ind, col_ind)

    def _get_concat_weights(self, expert: nn.Module) -> torch.Tensor:
        """
        Concatenate expert weight matrices: [gate; up; down^T] along dim=1.
        Result shape: (d_bottleneck, d_model * 3) matching blog pseudocode.
        """
        gate_w = getattr(expert, self.model_attrs["gate_proj"]).weight  # (d, d_model)
        up_w = getattr(expert, self.model_attrs["up_proj"]).weight      # (d, d_model)
        down_w = getattr(expert, self.model_attrs["down_proj"]).weight  # (d_model, d)
        return torch.cat([gate_w, up_w, down_w.T], dim=1).float()

    def _fused_permute(
        self,
        experts: nn.Module,
        expert_indices: list[int],
        dom_expert_idx: int,
    ):
        """Fused expert permutation not implemented for activation+weight method."""
        raise NotImplementedError(
            "ActivationWeightPermuter does not support fused experts. "
            "Use WeightMatchingPermuter for fused models."
        )


PERMUTER_REGISTRY = {
    "wm": WeightMatchingPermuter,
    "direct": DirectAlignmentPermuter,
    "activation_weight": ActivationWeightPermuter,
}
