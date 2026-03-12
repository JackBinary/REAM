# REAM Checkpointing and Interrupt Handling

This document describes the checkpointing, interrupt handling, and multi-GPU parallelization features added to the REAM pipeline, enabling cost-effective execution on interruptible cloud instances.

## Overview

The REAM pipeline now supports:
- **Graceful interruption**: Handles SIGTERM and SIGINT signals to save progress before exiting
- **Checkpointing**: Saves complete state after each layer (configurable interval)
- **Resume from checkpoint**: Restores model state and continues from where it left off
- **Multi-GPU parallelization**: Uses multiple GPUs in parallel for calibration data collection

This allows you to use **interruptible instances** (typically 2/3 the price of continuous instances) without losing progress, and **multi-GPU instances** for near-linear speedup.

## Usage

### Basic Usage with Auto-Resume

```bash
# Start a new run (or auto-resume from latest checkpoint)
python -m ream.ream \
    --model_name Qwen/Qwen3-30B-A3B \
    --dataset_name ream_mixed \
    --compression_ratio 0.5 \
    --auto_resume
```

### Resume from Specific Checkpoint

```bash
# Resume from a specific checkpoint
python -m ream.ream \
    --model_name Qwen/Qwen3-30B-A3B \
    --dataset_name ream_mixed \
    --compression_ratio 0.5 \
    --resume_from_checkpoint /path/to/results/checkpoints/checkpoint_20260312_143022
```

### Multi-GPU Parallelization

```bash
# Use 4 GPUs in parallel for faster calibration data collection
python -m ream.ream \
    --model_name Qwen/Qwen3-30B-A3B \
    --dataset_name ream_mixed \
    --compression_ratio 0.5 \
    --num_gpus 4 \
    --auto_resume
```

### Using the Helper Script

```bash
# Make executable
chmod +x scripts/ream-resume.sh

# Run with auto-resume (single GPU)
./scripts/ream-resume.sh

# Run with 4 GPUs
NUM_GPUS=4 ./scripts/ream-resume.sh

# Or resume from specific checkpoint
./scripts/ream-resume.sh --resume_from_checkpoint /path/to/checkpoint
```

## How It Works

### Interrupt Handling

When a SIGTERM or SIGINT signal is received (e.g., cloud instance preemption, Ctrl+C):

1. The current layer merge completes
2. A checkpoint is saved with:
   - Model state dict (all merged layers so far)
   - Calibration data (tokenized)
   - Cluster labels and centroids
   - All configuration parameters
3. The process exits gracefully with code 0

### Multi-GPU Parallelization

When `num_gpus > 1`:

1. Calibration data is split across GPUs
2. Each GPU loads a copy of the model
3. GPUs process their data subset in parallel
4. Results are aggregated (REAP scores, expert outputs, gate logits)
5. Provides near-linear speedup (4 GPUs ≈ 4x faster)

**Note**: The model must fit on a single GPU. Multi-GPU is for parallelizing calibration data collection, not for model parallelism.

### Checkpoint Contents

Each checkpoint directory contains:

```
checkpoint_20260312_143022/
├── checkpoint_metadata.json    # Configuration and progress info
├── model_state.pt              # Model state dict
├── calibration_data.pt         # Tokenized calibration inputs
└── clusters.pt                 # Cluster labels and centroids
```

### Checkpoint Metadata

The `checkpoint_metadata.json` includes:

```json
{
  "version": 1,
  "timestamp": "2026-03-12T14:30:22.123456",
  "layer_idx": 15,
  "layers_to_process": [15, 16, 17, ...],
  "num_clusters": 8,
  "max_cluster_size": 16,
  "merge_method": "frequency_weighted_average",
  "use_activation_weight_permute": true,
  "prune_gate": true,
  "skip_first": false,
  "skip_last": false,
  "seed": 42,
  "model_name": "Qwen/Qwen3-30B-A3B",
  "results_dir": "/path/to/results"
}
```

## Configuration Options

### MergeArgs

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `resume_from_checkpoint` | str \| None | None | Path to checkpoint directory to resume from |
| `auto_resume` | bool | False | Auto-detect and resume from latest checkpoint |
| `checkpoint_interval` | int | 1 | Save checkpoint every N layers |
| `num_gpus` | int | 1 | Number of GPUs to use in parallel |

## Cost Savings Example

For a typical 30B MoE model with 64 experts merging to 8 experts:

### Interruptible Instance Savings

| Instance Type | Hourly Cost | Interruption Risk | Total Cost |
|---------------|-------------|-------------------|------------|
| Continuous | $4.00/hr | 0% | $40.00 (10 hrs) |
| Interruptible | $2.67/hr | ~30% | ~$27.00 (with 2 resumes) |

**Savings: ~32%**

### Multi-GPU Speedup

| GPUs | Time | Cost (Interruptible) | Speedup |
|------|------|----------------------|---------|
| 1 | 10 hrs | $26.70 | 1x |
| 2 | 5.2 hrs | $27.77 | 1.9x |
| 4 | 2.7 hrs | $28.84 | 3.7x |
| 8 | 1.5 hrs | $32.04 | 6.7x |

**Note**: RunPod pricing scales linearly with GPUs, so total cost remains similar, but wall-clock time decreases dramatically.

### Combined Savings

Using 4 GPUs on an interruptible instance:
- **Time**: 2.7 hours (vs 10 hours single GPU continuous)
- **Cost**: ~$29 (vs $40 single GPU continuous)
- **Total savings**: 27% cost, 73% time

## Best Practices

1. **Use auto-resume**: Always run with `--auto_resume` to automatically resume from interruptions
2. **Use multiple GPUs**: Set `--num_gpus` to the number of available GPUs for faster processing
3. **Checkpoint frequently**: Default interval of 1 is recommended; increase only if checkpoint overhead is significant
4. **Monitor disk space**: Checkpoints can be large (model state + calibration data)
5. **Clean up old checkpoints**: Remove old checkpoints after successful completion to save disk space
6. **Match GPU count**: Ensure `--num_gpus` matches available GPUs on your instance

## Troubleshooting

### Checkpoint Version Mismatch

If you see:
```
ValueError: Checkpoint version mismatch: 0 != 1
```

The checkpoint was created with an older version. You may need to manually migrate or start fresh.

### Missing Calibration Data

If calibration data fails to load from checkpoint, the pipeline will attempt to reload it from the original dataset. Ensure dataset credentials are still valid.

### Out of Memory During Resume

If you encounter OOM when resuming:
1. Ensure the same GPU configuration is used
2. Check that `device_map="auto"` is compatible with your setup
3. Consider using `--max_cluster_size` to reduce memory usage

### Multi-GPU Issues

If multi-GPU fails:
1. Ensure all GPUs have enough memory for the model
2. Check CUDA visibility: `nvidia-smi` should show all GPUs
3. Try reducing `--num_gpus` if some GPUs have less memory
4. Check logs for which GPU failed

## Implementation Details

### Signal Handlers

The `InterruptHandler` class manages signal handling:

```python
with InterruptHandler() as handler:
    for layer in layers:
        if handler.interrupted:
            save_checkpoint(...)
            sys.exit(0)
        # Process layer...
```

### Multi-GPU Processing

The `collect_layer_data` function handles multi-GPU:

```python
# Split calibration data across GPUs
chunks = [calibration_inputs[i::num_gpus] for i in range(num_gpus)]

# Launch parallel workers
for gpu_id, chunk in zip(gpus, chunks):
    p = mp.Process(target=worker_fn, args=(gpu_id, chunk, ...))
    p.start()

# Aggregate results
reap_scores = sum(results) / total_tokens
expert_outputs = torch.cat([r["expert_outputs"] for r in results], dim=1)
```

### Checkpoint Save/Load

- `save_checkpoint()`: Saves all state to disk
- `load_checkpoint()`: Restores state from disk
- `find_latest_checkpoint()`: Auto-detects most recent checkpoint

### Memory Management

Checkpoints are saved to disk immediately after each layer to minimize memory overhead. The calibration data is kept in memory during processing but saved to disk in the checkpoint.

For multi-GPU, each worker process loads its own model copy and frees memory after processing its chunk.
