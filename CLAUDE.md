# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A **Masked Autoencoder (MAE) foundation model for ultrasound A-mode signals**, designed for multi-dataset pretraining on CINECA Leonardo HPC (multi-node DDP via SLURM). The codebase is split into three decoupled stages: ETL → DataModule → Model/Training.

## Environment setup (CINECA Leonardo)

```bash
module load profile/deeplrn
module load cineca-ai/4.3.0
source ~/usf_venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.11/site-packages:$PYTHONPATH"
cd ~/us_foundation
```

All `python -m` commands must be run from `us_foundation/`.

## Running the pipeline

### ETL

```bash
python -m runners.run_etl --config configs/etl/etl_config_sassauna.yaml
# With overrides:
python -m runners.run_etl --config configs/etl/etl_config_sassauna.yaml \
    --preprocessing_mode envelope --output_dir /scratch/output_envelope
```

### Training

```bash
# Standard run (4xGPU, single node):
python -m runners.run_train --config configs/model/experiments/hdf5_17M_DynamicSampling_FixedS_raw.yaml

# With CLI overrides (dot-path syntax, yaml.safe_load values):
python -m runners.run_train --config configs/model/experiments/hdf5_17M_DynamicSampling_FixedS_raw.yaml \
    --override train.max_epochs=50 data.batch_size=128

# Multi-node via SLURM:
srun python -m runners.run_train --config ... \
    --override train.devices=4 train.num_nodes=4

# Resume from checkpoint:
python -m runners.run_train --config ... --ckpt-path /path/to/last.ckpt

# Smoke test (1 batch, CPU, no GPU needed):
python -m runners.run_train --config configs/model/experiments/hdf5_17M_DynamicSampling_FixedS_raw.yaml \
    --override train.max_epochs=1 train.devices=1 train.accelerator=cpu train.precision=32 data.batch_size=4 data.num_workers=0
```

### Testing / Inference

```bash
python -m runners.run_test \
    --test-data /path/to/test.h5 \
    --config /path/to/saved/config.yaml \
    --checkpoint /path/to/model.ckpt \
    --num-samples 3 --output-dir /path/to/output
```

### Syntax check

```bash
python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]"
```

## Config system

Experiment YAMLs under `configs/model/experiments/` use a `defaults: [base]` key. The runner (`runners/run_train.py:_load_composed_yaml`) merges them recursively with `configs/model/base.yaml` — no Hydra dependency.

After training, a flat `config.yaml` (no `defaults:`) is saved to `{output_dir}/{run_name}/`. This flat format is what `run_test.py` expects.

Three top-level sections: `data`, `model`, `train`.

## Architecture overview

```
MultiTokenizer → CTRoPE (in-attention) → USEncoder (MAE masking) → USDecoder (multi-size head) → USReconstructionLoss
```

Key design decisions:

**Multi-branch tokenizer** (`model/tokenizer/multi_tokenizer.py`): Routes each sample to the best window size `W* ∈ window_sizes` by minimizing `|W·c/(2·fs) − target_patch_mm|` where `c=1540 m/s`. Uses `MultiInSizeLinear` (MOIRAI-style) for the MLP branch: weight `(num_W, E, max_W)` with masking. The same routing function is shared between DataModule and Tokenizer — they must stay in sync.

**CT-RoPE** (`model/positional/ct_rope.py`): Continuous-Time Rotary Embeddings. Angles are proportional to absolute timestamps in µs (not position indices), making attention sensitive to physical time distance between patches across different sampling frequencies.

**MAE masking** (`model/backbone/us_encoder.py`): Padded tokens are guaranteed to be masked by injecting noise `2.0` before argsort. `len_keep[b] = round((1−masking_ratio)·n_valid[b])` clamped ≥ 1. Encoder processes only visible tokens; decoder unshuffles via `ids_restore`.

**Multi-size reconstruction head** (`model/backbone/us_decoder.py`): Separate `nn.Linear` per window size, aggregated via masked sum. Decoder output is `(B, S, W_max)`.

**Loss** (`criterion/us_reconstruction_loss.py`): Smooth-L1 on masked tokens only by default. `norm_target=True` applies per-patch zero-mean/unit-std normalization to the target (MAE He et al. 2021 trick). `alpha > 0` adds an auxiliary visible-token loss (TimeFM-style).

## Data pipeline

### HDF5 layout (CSR-style, `etl/writers.py`)

Single flat buffer `data: (M,) float32` with `offsets: (N+1,) int64`. Signals stored at **native length** (no padding). Per-sample metadata arrays: `sampling_frequencies`, `dataset_sources`, `signal_means/stds/mins/maxs`.

`offsets` is loaded fully into RAM at `HDF5Dataset.__init__` (~136 MB for 17M samples). The HDF5 file itself is opened lazily per-worker (DDP fork-safe).

### Variable-S vs Fixed-S batching

| Mode | `target_patches` | Description |
|---|---|---|
| variable-S | `null` | Native length signals, collated with padding to batch max `T` |
| fixed-S | `int` (e.g. 50) | Each acquisition chunked into `target_patches·W*` samples deterministically; all chunks seen; transformer always sees `(B, 50, E)` |

**Fixed-S is incompatible with online preprocessing** (bandpass/envelope/interpolation). Those transforms must be applied offline in the ETL stage instead.

### Sampling strategies (`data/hdf5_datamodule.py`)

- `naive`: all samples each epoch
- `static`: fixed caps per dataset via `dataset_caps`
- `dynamic_epoch`: `epoch_k` samples drawn from `lateral_gastrocnemius_verasonics` (the dominant dataset) per epoch with a new shuffle each epoch; `EpochSubsetSampler` ensures DDP consistency via `seed + epoch`
- `proportional`: MOIRAI-style threshold-ratio caps

`dynamic_epoch` is only supported with HDF5 (not WebDataset).

### WebDataset pipeline

Shard-based streaming: `SimpleShardList → shuffle_shards → split_by_node → split_by_worker → tarfile_to_samples → shuffle_buffer → decode → map → filter(filler) → batch`. Last shard is padded with filler samples (marked `is_filler: true`) to avoid DDP NCCL hangs.

## Key invariants

- **No interpolation in ETL**: signals keep native fs and length. The multi-tokenizer routing is physically meaningful only on native-fs data.
- **Routing function must be consistent**: `select_branch(fs, window_sizes, target_patch_mm)` in `multi_tokenizer.py` must match what the DataModule pre-computes in `hdf5_datamodule.py`.
- **`samples_per_shard % batch_size == 0`** and `n_shards % (world_size·num_workers) == 0` are validated at ETL config time.
- **CT-RoPE rotates Q/K at every attention layer** (not additive positional embeddings on tokens).
- WandbLogger runs offline by default on CINECA; sync later with `wandb sync`.