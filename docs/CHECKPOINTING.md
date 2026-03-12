# REAM Checkpointing and Interrupt Handling

This document describes the checkpointing and interrupt handling features added to the REAM pipeline, enabling cost-effective execution on interruptible cloud instances.

## Overview

The REAM pipeline now supports:
- **Graceful interruption**: Handles SIGTERM and SIGINT signals to save progress before exiting
- **Checkpointing**: Saves complete state after each layer (configurable interval)
- **Resume from checkpoint**: Restores model state and continues from where it left off

This allows you to use **interruptible instances** (typically 2/3 the price of continuous instances) without losing progress.

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

### Using the Helper Script

```bash
# Make executable
chmod +x scripts/ream-resume.sh

# Run with auto-resume
./scripts/ream-resume.sh

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

## Cost Savings Example

For a typical 30B MoE model with 64 experts merging to 8 experts:

| Instance Type | Hourly Cost | Interruption Risk | Total Cost |
|---------------|-------------|-------------------|------------|
| Continuous | $4.00/hr | 0% | $40.00 (10 hrs) |
| Interruptible | $2.67/hr | ~30% | ~$27.00 (with 2 resumes) |

**Savings: ~32%**

## Best Practices

1. **Use auto-resume**: Always run with `--auto_resume` to automatically resume from interruptions
2. **Checkpoint frequently**: Default interval of 1 is recommended; increase only if checkpoint overhead is significant
3. **Monitor disk space**: Checkpoints can be large (model state + calibration data)
4. **Clean up old checkpoints**: Remove old checkpoints after successful completion to save disk space

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

### Checkpoint Save/Load

- `save_checkpoint()`: Saves all state to disk
- `load_checkpoint()`: Restores state from disk
- `find_latest_checkpoint()`: Auto-detects most recent checkpoint

### Memory Management

Checkpoints are saved to disk immediately after each layer to minimize memory overhead. The calibration data is kept in memory during processing but saved to disk in the checkpoint.
