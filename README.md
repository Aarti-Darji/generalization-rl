# generalization-rl

Selection bias and sources of variance in RL post-training of small reasoning models.

Train small reasoning models with SFT then GRPO across several seeds, log every sampled
answer, and run two analyses on the same logs:

- **Part A — Winner's curse in checkpoint selection.** How much does picking the best of K
  RL checkpoints on a small benchmark overstate its accuracy, and does a correlation-aware
  Gaussian plug-in predict the inflation better than the Hoeffding / union bound?
- **Part B — Where run-to-run variation comes from.** Split variation in RL results into
  training-seed, problem, seed × problem and evaluation-sampling components with a
  random-effects ANOVA, and lightly replicate the pass@k finding of Yue et al. (2025).

See [`docs/proposal.md`](docs/proposal.md) for the full plan and the math.

## Layout

```
configs/            YAML configs for model, SFT, GRPO and evaluation
src/genrl/
  data/             GSM8K / MATH-500 loading and the eval pool
  grading/          answer extraction and exact-match grading
  training/         SFT and GRPO entry points (TRL), reward functions
  eval/             vLLM sampling and the per-sample log schema
  analysis/         selection bias, variance components, pass@k, stats helpers
scripts/            one command per pipeline stage / figure
tests/              unit tests for grading and analysis (CPU only)
notebooks/          exploration
results/            logs/ and figures/ (git-ignored; logs released separately)
```

## Pipeline

```
base model ──SFT──▶ SFT model ──GRPO × S seeds──▶ checkpoints every 25 steps
                                                        │
                                   vLLM sampling on the eval pool (GSM8K test + MATH-500)
                                                        │
                                       per-sample log table (parquet)
                                          ┌─────────────┴─────────────┐
                                     Part A analysis             Part B analysis
```

Everything after the log runs without a GPU.

## Quickstart

```bash
pip install -e ".[dev]"          # analysis + tests (CPU)
pip install -e ".[train]"        # adds torch, trl, vllm for training / sampling

pytest                            # unit tests
python scripts/synthetic_check.py # reproduce the synthetic sanity check of the math

python scripts/run_sft.py  --config configs/sft/gsm8k.yaml
python scripts/run_grpo.py --config configs/grpo/gsm8k.yaml --seed 0
python scripts/eval_checkpoints.py --config configs/eval/pool.yaml --checkpoints outputs/grpo/seed0
python scripts/handcheck_grader.py --log results/logs/samples.parquet --n 100

python scripts/part_a.py --log results/logs/samples.parquet
python scripts/part_b.py --log results/logs/samples.parquet
```

## Status

Starter code. Week-by-week plan in the proposal.
