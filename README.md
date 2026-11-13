# Scheme C + KL + es0.9 + KL-Validated Neuron Release (Continual World CW10, 4×256 Actor)

Prepared on 2026-11-13. The code comes from `~/nevo_runs/cw_v2/`. This package is a self-contained copy,
with only unused draft files removed from `jaxrl`.
The original release passed the end-to-end smoke test (`scripts/smoke_test.sh`, 3 tasks with all four
components enabled) and the unit tests: 57/57 in `test_maskdecay.py` and 18/18 in `test_release.py`.
**The full CW10 combination of C+KL+es0.9+neuron release has not been run yet** (see Section 5 for existing results).

## 1. The Four Components

| Component | Description | Code location |
|---|---|---|
| Scheme C | For each task, sample an independent random neuron mask for each of 5 candidates (25% of neurons per layer). Only masked-in neurons participate in the forward pass (Nevo architecture). After each Adam update, multiply updates to weights within previous task winners' weight blocks by λ^h (λ = 0.01, h = the number of previous tasks that used that weight), so those weights barely change. | Mask sampling: `random_population` in `code/maskdecay.py`; update scaling: `code/jaxrl/agents/sac/cumulative_learner.py` (`scale_position="post_adam"`). |
| + KL anchors | After training each task, save 4,096 states from its replay buffer and the policy's (μ, σ) at those states. On each subsequent actor update, sample 256 states from the anchors of all previous tasks and add 30 × KL(π_old ‖ π_θ) to the loss. Evaluate each previous state using its own task's mask. | `cumulative_learner.py` (`anchor_mode="kl"`); anchor storage: `finish_task` near the end of `train_maskdecay_candidate.py`. |
| + es (success-based early stopping) | Every 20,000 steps, evaluate the current task for 10 episodes. If the success rate reaches the threshold (0.9 by default), evaluate another 50 episodes with a different seed to confirm. The first candidate to pass confirmation wins the task, and the other 4 stop immediately. Each candidate trains for at most 1 million steps. If none passes confirmation, all train to the limit, and the winner is selected by return. | `train_maskdecay_candidate.py` (`--early-stop-success`, `--confirm-episodes`); winner selection: `run_maskdecay.py`. |
| + Neuron release (our method) | Before task t (t > 0), consider removing neurons from each previous task j's mask, layer by layer, in ascending order of contribution U = E\|h\| × Σ\|downstream weights\|. Accept a removal only if, on task j's anchor states, **the single-step KL is ≤ 0.1 and the cumulative KL relative to the policy at task completion is ≤ 0.3**. Remove at most 5% per layer and make at most 16 attempts per layer. Reinitialize neurons no longer referenced by any previous task and return them to the free pool. Prefer these neurons when sampling the new task's masks (`--mask-priority released`). U only determines the attempt order; KL alone determines whether a removal is accepted. | `code/neuron_release.py` (called by `run_maskdecay.py` before each task); priority pool: `priority_population` in `maskdecay.py`. |

## 2. Directory Layout

```text
README.md                      This file
RESULTS_existing_runs.txt       Results from 5 existing runs (Section 5)
code/
  run_maskdecay.py              Driver: sample masks → (release neurons) → train 5 candidates
                               in parallel → select a winner → commit; resume at task boundaries
  train_maskdecay_candidate.py  SAC training for one candidate (masked forward passes, λ^h
                               update scaling, KL anchors, early-stop evaluation and confirmation)
  neuron_release.py            KL-validated neuron removal and reclamation
  maskdecay.py                 Mask sampling, weight blocks, and usage history counts
  train_cumulative.py          Shared replay, evaluation, environment stepping, and other utilities
  continual_world.py           CW10 environment
  jaxrl/                       Networks and SAC learners (cumulative_learner.py implements this method)
  test_maskdecay.py            Unit tests (masks, update scaling, forward masking, etc.; 57 tests)
  test_release.py              Neuron-release unit tests (18 tests; requires start_task03.pkl
                               from an existing run)
  environment.yml             Official Nevo-CRL environment; use its metaworld version
scripts/
  env.sh / env.sh.example      Runtime environment variables (actual local RTX 4090 configuration;
                               adapt paths for other machines)
  launch_ckl_es_release.sh     Full run (CW10, 10 tasks, 4×256, runs in the background)
  smoke_test.sh               End-to-end smoke test (3 tasks, a very small budget, about 5 minutes)
  run_tests.sh                Both unit-test suites
  summarize_ckl_runs.py        Summarize results from ledger.json
```

## 3. Environment

Python 3.8, JAX 0.3.17 (CUDA 11.8 runtime libraries), mujoco-py 2.0.2.8 + MuJoCo 2.0, and metaworld commit 0875192.
The comments in `scripts/env.sh.example` describe two local adjustments: mujoco-py is built with the EGL
backend (no rendering is used; physics is unchanged), and XLA uses ptxas from the system CUDA 12.4 toolkit.
On another machine, copy `env.sh.example` to `env.sh` and adapt the paths.

## 4. Usage

```bash
# Smoke test (checks that the entire pipeline runs; not an experimental result)
bash scripts/smoke_test.sh /tmp/ckl_smoke            # The last line should be SMOKE_TEST PASS

# Unit tests
CKL_TEST_EXP=/path/to/a/ckl/run bash scripts/run_tests.sh

# Full run: C+KL+es0.9+neuron release, CW10, 4x256
#   Arguments: output directory [seed=110] [mask_seed=7] [threshold=0.9] [mask priority=released|none]
bash scripts/launch_ckl_es_release.sh ~/nevo_runs/new_runs/ckl_es0.9_release_s220 220 220 0.9 released

# Summarize results (multiple runs can be supplied together)
python3 scripts/summarize_ckl_runs.py es0.9_rel_s220=~/nevo_runs/new_runs/ckl_es0.9_release_s220
```

A full run launches 5 candidate processes concurrently on a single RTX 4090. As a reference, one es0.9
run without neuron release takes about 13 hours; the release step adds a few minutes before each task.
If interrupted, rerun the same command to resume from the last committed task.

**Outputs** (in the output directory):

- `ledger.json`: each task's winner, whether confirmation passed, steps used by each candidate, the winner's final 50-episode evaluation on all tasks learned so far, and neuron-release records.
- `taskXX/candK/`: training and evaluation logs for each candidate.
- `release/`: logs for each release step, updated masks, and starting states after reclamation.
- `winner_mask_taskXX.npz`, `start_taskXX.pkl`: each task winner's mask and the starting state for the next task.
- CW's `random_init_all` samples initial states from NumPy's **global** RNG. Setting only `env.seed` does not make evaluation reproducible; for paired evaluations, seed both `np.random` and `random`.
