# REAM: Router-weighted Expert Activation Merging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JackBinary/REAM/blob/main/ream_colab.ipynb)

Fork of [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) implementing REAM ([blog post](https://bknyaz.github.io/blog/2026/moe/)) instead of REAP.

## What is REAM?

REAM compresses Mixture-of-Experts LLMs by **merging** groups of experts rather than pruning them. It reduces the expert count by 25% (e.g., 128 → 96) with strong benchmark retention across Qwen3-30B, Qwen3-235B, and Qwen3-Next-80B.

Qwen3.5 models (35B-A3B, 122B-A10B, 397B-A17B) are supported. Because Qwen3.5 ships as a vision-language model, REAM automatically loads only the text decoder and discards the vision encoder so you don't pay for weights you're not compressing.

Six things make REAM different from prior merging methods like HC-SMoE:

1. **REAP-score centroids** — The k retained experts are chosen by REAP saliency (router-gate-weighted activation norms), not by frequency.
2. **Pseudo-pruning grouping** — Starting from the highest-scoring centroid, the C most similar unassigned experts are grouped with it. Most groups end up as singletons; a few grow large. The weighted merge is dominated by the centroid, hence "pseudo-pruning."
3. **Gated similarity** — Expert similarity is the average of (a) cosine similarity of expert outputs weighted by gate values and (b) cosine similarity of gate logit columns.
4. **Activation + weight permutation alignment** — Before averaging, expert neurons are aligned using the Hungarian algorithm on a cost matrix combining hidden-activation distances and concatenated-weight distances.
5. **Sequential layer processing** — Each layer is merged, then calibration data is re-forwarded through the updated model before merging the next layer. Prior methods compute all activations from the original model upfront.
6. **Gate weight pruning** — After merging, non-centroid rows are removed from the router weight matrix (following REAP's gate adjustment).

## Results (from the [blog post](https://bknyaz.github.io/blog/2026/moe/))

| Model | Experts | Method | MC avg | GEN avg |
|---|---|---|---|---|
| Qwen3-30B-A3B | 128 (original) | — | 69.7 | 70.9 |
| Qwen3-30B-A3B | 96 | HC-SMoE | 64.8 | 66.3 |
| Qwen3-30B-A3B | 96 | REAP | 65.2 | 64.1 |
| Qwen3-30B-A3B | 96 | **REAM** | **65.8** | **67.7** |
| Qwen3-235B-A22B | 96 | REAP | — | 72.9 |
| Qwen3-235B-A22B | 96 | **REAM** | — | 71.7 |
| Qwen3-Next-80B | 384 | REAP | — | 68.5 |
| Qwen3-Next-80B | 384 | **REAM** | — | **69.3** |

## Calibration Data

REAM uses a fixed mix of 2048 sequences:

| Domain | Dataset | Samples | Max tokens | Share |
|---|---|---|---|---|
| General | allenai/c4/en | 512 | 128 | 8% |
| Math | AI-MO/NuminaMath-1.5 (cn_k12, olympiads) | 1024 | 512 | 68% |
| Coding | bigcode/the-stack-smol | 512 | 512 | 24% |

## Quick Start

```bash
# Install
bash scripts/build.sh

# Run REAM on Qwen3-30B-A3B (25% compression, GPU 0)
bash experiments/ream-cli.sh 0

# Custom model and compression ratio
bash experiments/ream-cli.sh 0,1 Qwen/Qwen3-30B-A3B 42 0.25 16
```

## New Files

| File | Purpose |
|---|---|
| `src/reap/ream.py` | Main pipeline: sequential merging, gate pruning, calibration data loading |
| `src/reap/ream_cluster.py` | REAM clustering: REAP-score centroids, pseudo-pruning, gated similarity |
| `experiments/ream-cli.sh` | CLI wrapper for running REAM experiments |

Modified from REAP:

| File | Change |
|---|---|
| `src/reap/permute.py` | Added `ActivationWeightPermuter` (activation + weight cost matrix) |
| `src/reap/model_util.py` | Added `Qwen3_5MoeForCausalLM` attrs, `get_layers`, `get_num_experts`, `fused_expert_forward`, `load_model_text_only` |
| `src/reap/data.py` | Added `NuminaMathLMDataset`, `TheStackSmolLMDataset` processors |
| `src/reap/args.py` | Added `ream_mixed`, `AI-MO/NuminaMath-1.5`, `bigcode/the-stack-smol` to dataset choices |
| `pyproject.toml` | Renamed to `ream`, added `scipy` dependency |

---

## Original REAP Documentation

Everything below is from the original REAP repository. The REAP pruning and merging pipelines remain fully functional.

---


## Installation

### venv
To build the project and setup a virtual environment install `uv` and run:
```bash
bash scripts/build.sh
```

### Docker
Alternatively, use docker:
```bash
docker compose up --build -d
docker compose exec app bash
```

The `docker-compose.yaml` file is setup to mount the default huggingface cache (`~/.cache/huggingface`), but if you use a different cache directory then we suggest updating the mount path to avoid excessive container storage sizes.  

### Configuration
Copy `.env.template` and rename as `.env`. Populate the empty fields.

For WildBench, copy `config/wildbench_prod_env_XXXX.example`. Update the copied subdir name with the port used to launch vLLM, defaults to 800X where X is rank of the first GPU used to run the eval script. I.e, `wildbench_prod_env_8000` if running `eval.py` with `CUDA_VISIBLE_DEVICES=0,1,2,3`. In the copied subdir, update `credentials.conf` with your OpenAI API key and in `model_deployments.yaml` substitute `base_url:XXXX` with the port selected. i.e., `http://0.0.0.0:XXXX/v1/ -> http://0.0.0.0:8000/v1/` for the example above. 


### Adding a new model
Add model attribute names to `MODEL_ATTRS` in `src/reap/model_util.py`. Each entry is identified by the class name of the model `model.__class__.__name__` as key. The values correspond to the following:
- `moe_block`: Attribute name of SMoE submodule in the decoder module. 
- `*_proj`: Attribute names for the expert projections. 
- `experts`: Attribute of the ModuleList containing the experts in the SMoE module. 
- `fused`: If true, the model uses a FusedMoE layer. Of the currently supported models, only Llama-4 is fused. 
- `router`: Attribute name of the router/gate layer in the SMoE module. 
- `num_experts`: The key in the huggingface config containing the number of experts per layer. (ie., `num_experts` if `model.config.num_experts` contains this value)
- `num_experts_per_tok`: The key in the huggingface config containing the number of experts activated per token. 


## Reproducing Experiments

### Expert Merging

To run merging experiments, use:

```bash
bash experiments/merging-cli.sh <CUDA_DEVICES> [MODEL_NAME] [MERGE_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL] [RUN_EVALPLUS] [RUN_LIVE_CODE_BENCH] [RUN_MATH] [RUN_WILDBENCH] [SINGLETON_SUPER_EXPERTS] [SINGLETON_OUTLIER_EXPERTS]
```

- `CUDA_DEVICES`: e.g. `0` or `0,1`
- `MODEL_NAME`: (default: Qwen/Qwen3-30B-A3B)
- `MERGE_METHOD`: `hc_smoe`, `m_smoe`, or `submoe` (default: hc_smoe)
- `SEED`: (default: 42)
- `COMPRESSION_RATIO`: (default: 0.25)
- `DATASET_NAME`: (default: theblackcat102/evol-codealpaca-v1)
- `RUN_*`: Flags control which evaluations to run (true/false)
- `SINGLETON_SUPER_EXPERTS` and `SINGLETON_OUTLIER_EXPERTS` force super and outlier experts into singleton clusters, respectively. See [Unveiling Super Experts in Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2507.23279) paper for definitions.

Example:
```bash
bash experiments/merging-cli.sh 0 Qwen/Qwen3-30B-A3B hc_smoe 42 0.25 theblackcat102/evol-codealpaca-v1 true true true false false
```

### Expert Pruning

To run pruning experiments, use:

```bash
bash experiments/pruning-cli.sh <CUDA_DEVICES> [MODEL_NAME] [PRUNING_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL] [RUN_EVALPLUS] [RUN_LIVE_CODE_BENCH] [RUN_MATH] [RUN_WILDBENCH] [SINGLETON_SUPER_EXPERTS] [SINGLETON_OUTLIER_EXPERTS]
```

- `PRUNING_METHOD`: e.g. `reap` will use the REAP expert saliency criterion for expert pruning. 
- Other arguments are similar to merging.

Example:
```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B frequency 42 0.25 theblackcat102/evol-codealpaca-v1 true true true false false
```

---

## Source Directory Structure

The `src/reap` directory contains the main codebase:

- **args.py**: Argument dataclasses for experiment configuration.
- **cluster.py**: Clustering algorithms for grouping experts.
- **data.py**: Dataset loading and processing utilities.
- **eval.py**: Evaluation routines for models and experiments.
- **main.py**: Main entry point for running merging experiments and pipelines.
- **merge.py**: Core logic for merging experts in Mixture-of-Experts (MoE) models.
- **metrics.py**: Distance and similarity metrics for model analysis.
- **model_util.py**: Utilities for model introspection and manipulation.
- **observer.py**: Hooks and classes for collecting model activations.
- **permute.py**: Permutation and alignment utilities for expert weights.
- **prune.py**: Main entry point for expert pruning.
- **restricted_cluster.py**: Clustering with constraints (e.g., max cluster size).

### Models Subdirectory

- **models/**: Contains patched model definitions and configurations for select architectures that do not return router_logits in the SMoE module forward method. (e.g., GLM, ERNIE).


## Citation
Please consider using the following citations if you found this work useful:
```
@misc{knyazev2026compressing,
    title       = {REAM: Compressing Mixture-of-Experts LLMs},
    author      = {Boris Knyazev},
    year        = {2026},
    url         = {https://bknyaz.github.io/blog/2026/moe/},
}
```
```
@misc{lasby-reap,
    title       = {{REAP the Experts: Why Pruning Prevails for One-Shot MoE compression}},
    author      = {Lasby, Mike and Lazarevich, Ivan and Sinnadurai, Nish and Lie, Sean and Ioannou, Yani and Thangarasa, Vithursan},
    year        = {2025},
    publisher   = {arXiv},
    note        = {arXiv:2510.13999v1 [cs]},
    url         = {https://arxiv.org/abs/2510.13999v1}, 
}
```

