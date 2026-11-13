"""Single-actor CW SAC with permanent, task-wise magnitude LR decay.

Only this entry point and cumulative_learner.py implement the new experiment.
Checkpoints resume at episode boundaries, including replay and all RNG states;
they never serialize an in-progress MuJoCo trajectory.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


LOG_FILES = ("train.jsonl", "diagnostics.jsonl", "episodes.jsonl",
             "evaluations.jsonl", "training.csv")
MAX_WALLTIME_SECONDS = 45 * 3600
STOP_REQUESTED = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="cw10")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--run-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "runs" /
                        "cumulative_cw10_s110_20261015")
    parser.add_argument("--steps-per-task", type=int, default=1_000_000)
    parser.add_argument("--task-limit", type=int, default=0,
                        help="Number of tasks; zero runs the full sequence.")
    parser.add_argument("--start-training", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--actor-width", type=int, default=1024)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--actor-variant", choices=("legacy", "stable"), default="legacy")
    parser.add_argument("--zero-grad-patience", type=int, default=0,
                        help="Stop after this many consecutive exact-zero actor updates; zero disables the guard.")
    parser.add_argument("--retain-task-checkpoints", action="store_true",
                        help="Keep immutable task-start/end checkpoint archives as well as latest.pkl.")
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--mask-fraction", type=float, default=0.1)
    parser.add_argument("--refresh-interval", type=int, default=10_000,
                        help="Actor optimizer steps between TopK refreshes.")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000,
                        help="Environment steps; saved at the next episode boundary.")
    parser.add_argument("--walltime-seconds", type=int, default=MAX_WALLTIME_SECONDS)
    parser.add_argument("--resume", default="auto",
                        help="auto, none, or an explicit checkpoint path.")
    parser.add_argument("--require-gpu", action="store_true")
    # v2 protocol options (E0 task-boundary ablation). Defaults reproduce the frozen protocol.
    parser.add_argument("--reset-critic", type=int, choices=(0, 1), default=1,
                        help="Re-initialize critics/target critics at every task start (frozen protocol: 1).")
    parser.add_argument("--reset-actor-optimizer", type=int, choices=(0, 1), default=1,
                        help="Re-initialize the actor Adam state at every task start (frozen protocol: 1).")
    parser.add_argument("--temperature-mode", choices=("shared_reset", "shared_keep", "per_task"),
                        default="shared_reset",
                        help="shared_reset: one log alpha reset each task (frozen); shared_keep: never reset; "
                             "per_task: one log alpha per task id.")
    parser.add_argument("--init-log-temperature", type=float, default=1.0,
                        help="Initial log alpha (frozen protocol: 1.0, i.e. alpha=e).")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help="For tasks > 0, train only the critics for this many updates while the actor is frozen.")
    parser.add_argument("--early-eval-updates", default="",
                        help="Comma-separated actor-update counts within each task > 0 at which all seen tasks are "
                             "evaluated, e.g. 0,100,1000,5000,10000 (dense post-switch forgetting curve).")
    parser.add_argument("--protocol-tag", default="", help="Free-text label recorded in config.json.")
    # v2 protection options (E1 importance, E2 anchors). Defaults reproduce the frozen mechanism.
    parser.add_argument("--importance-mode", choices=("magnitude", "mas", "fisher"), default="magnitude")
    parser.add_argument("--importance-beta", type=float, default=10.0)
    parser.add_argument("--scale-floor", type=float, default=0.05)
    parser.add_argument("--importance-samples", type=int, default=2048)
    parser.add_argument("--importance-accumulate", choices=("max", "sum"), default="max")
    parser.add_argument("--anchor-mode", choices=("none", "kl", "ewc"), default="none")
    parser.add_argument("--anchor-coef", type=float, default=1.0)
    parser.add_argument("--anchor-states-per-task", type=int, default=4096)
    parser.add_argument("--anchor-batch", type=int, default=256)
    parser.add_argument("--actor-lr-ramp-updates", type=int, default=0,
                        help="For tasks > 0, linearly ramp the actor increment from 0 to full over this many updates.")
    parser.add_argument("--task-condition", choices=("position", "name"), default="position",
                        help="position: one-hot per task position (frozen); name: positions with the same "
                             "environment name share one condition (revisits in cw20 / cw10x3).")
    parser.add_argument("--anchor-budget", type=int, default=0,
                        help="Total anchor states kept across all conditions (0 = unbounded).")
    # Phase B plasticity options (capacity study). Defaults reproduce the Phase A behaviour.
    parser.add_argument("--anchor-target", choices=("full", "mean"), default="full",
                        help="full: Gaussian KL on (mu, sigma); mean: only the old mean action is anchored, "
                             "the current sigma stays free.")
    parser.add_argument("--log-temperature-min", type=float, default=None,
                        help="Floor on log alpha after every temperature step (default: no floor).")
    parser.add_argument("--task-code-scale", type=float, default=1.0,
                        help="Multiplier on the task one-hot fed to actor and critics.")
    args = parser.parse_args(argv)
    args.early_eval_updates = sorted({int(v) for v in args.early_eval_updates.split(",") if v.strip()})
    if args.critic_warmup_updates < 0:
        parser.error("--critic-warmup-updates must be nonnegative")
    positive = ("steps_per_task", "batch_size", "buffer_size", "eval_interval",
                "eval_episodes", "log_interval", "actor_width", "critic_width",
                "hidden_layers", "refresh_interval", "checkpoint_interval",
                "walltime_seconds")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    if args.task_limit < 0 or args.start_training < 0 or args.zero_grad_patience < 0:
        parser.error("task-limit, start-training and zero-grad-patience must be nonnegative")
    if not 0 < args.decay_factor <= 1 or not 0 < args.mask_fraction <= 1:
        parser.error("decay-factor and mask-fraction must be in (0,1]")
    if args.buffer_size < args.batch_size:
        parser.error("buffer-size must be at least batch-size")
    args.walltime_seconds = min(args.walltime_seconds, MAX_WALLTIME_SECONDS)
    args.run_dir = args.run_dir.expanduser().resolve()
    return args


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def scalar_metrics(values, jax):
    result = {}
    for key, value in jax.device_get(values).items():
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError("Non-scalar metric: " + key)
        number = float(array.reshape(()))
        if not math.isfinite(number):
            raise FloatingPointError("Non-finite training metric: " + key)
        result[key] = number
    return result


def record_update_health(nonfinite, nonzero_count, max_abs, progress, patience):
    """Inspect every raw actor gradient, rather than sampled float32 norms."""
    if patience <= 0:
        return None
    progress["checked_actor_updates"] += 1
    values = (float(nonfinite), float(nonzero_count), float(max_abs))
    if not all(math.isfinite(value) for value in values) or values[0] != 0:
        return "nonfinite_update"
    if values[1] < 0 or values[2] < 0:
        return "invalid_gradient_diagnostics"
    if values[1] == 0:
        progress["zero_grad_update_count"] += 1
        progress["zero_grad_streak"] += 1
    else:
        progress["zero_grad_streak"] = 0
    progress["max_zero_grad_streak"] = max(
        progress["max_zero_grad_streak"], progress["zero_grad_streak"])
    if progress["zero_grad_streak"] >= patience:
        return "zero_gradient"
    return None


class Replay:
    """Raw-observation ring buffer; checkpoints contain only initialized rows."""
    FIELDS = ("observations", "actions", "rewards", "masks", "next_observations")

    def __init__(self, capacity, observation_shape, action_shape):
        self.capacity, self.size, self.insert_index = capacity, 0, 0
        self.arrays = {
            "observations": np.empty((capacity,) + tuple(observation_shape), np.float32),
            "actions": np.empty((capacity,) + tuple(action_shape), np.float32),
            "rewards": np.empty(capacity, np.float32),
            "masks": np.empty(capacity, np.float32),
            "next_observations": np.empty((capacity,) + tuple(observation_shape), np.float32),
        }

    def insert(self, observation, action, reward, mask, next_observation):
        for key, value in zip(self.FIELDS,
                              (observation, action, reward, mask, next_observation)):
            self.arrays[key][self.insert_index] = value
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return {key: value[indices] for key, value in self.arrays.items()}

    def sample_observations(self, count, rng):
        """Distinct raw observations for task-end importance/anchor estimation (own RNG)."""
        count = min(count, self.size)
        indices = rng.choice(self.size, count, replace=False)
        return self.arrays["observations"][indices].copy()

    def state_dict(self):
        return {"capacity": self.capacity, "size": self.size,
                "insert_index": self.insert_index,
                "arrays": {key: value[:self.size].copy()
                           for key, value in self.arrays.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.size, self.insert_index = state["size"], state["insert_index"]
        for key, value in state["arrays"].items():
            self.arrays[key][:self.size] = value


def capture_rng(rng):
    if isinstance(rng, np.random.RandomState):
        return {"kind": "RandomState", "state": rng.get_state()}
    if hasattr(rng, "bit_generator"):
        return {"kind": "Generator", "state": copy.deepcopy(rng.bit_generator.state)}
    raise TypeError("Unsupported environment RNG: " + str(type(rng)))


def restore_rng(rng, state):
    if state["kind"] == "RandomState":
        rng.set_state(state["state"])
    else:
        if rng.bit_generator.state["bit_generator"] != state["state"]["bit_generator"]:
            raise ValueError("Checkpoint environment RNG type differs")
        rng.bit_generator.state = state["state"]


def env_rng_state(env):
    return {"environment": capture_rng(env.unwrapped.np_random),
            "action_space": capture_rng(env.action_space.np_random),
            "observation_space": capture_rng(env.observation_space.np_random)}


def restore_env_rng(env, state):
    restore_rng(env.unwrapped.np_random, state["environment"])
    restore_rng(env.action_space.np_random, state["action_space"])
    restore_rng(env.observation_space.np_random, state["observation_space"])


def reset_env(env):
    result = env.reset()
    return np.asarray(result[0] if isinstance(result, tuple) else result, np.float32)


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = terminated or truncated
        mask = 0.0 if terminated else 1.0
    else:
        observation, reward, done, info = result
        mask = 1.0 if not done or bool(info.get("TimeLimit.truncated", False)) else 0.0
    return np.asarray(observation, np.float32), float(reward), bool(done), info, mask


def forgetting_metrics(history, scores):
    previous_tasks = len(history)
    if not previous_tasks:
        return {"endpoint_forgetting": 0.0, "peak_forgetting": 0.0}
    endpoint = [history[j][j] - scores[j] for j in range(previous_tasks)]
    peak = [max(row[j] for row in history[j:]) - scores[j]
            for j in range(previous_tasks)]
    return {"endpoint_forgetting": float(np.mean(endpoint)),
            "peak_forgetting": float(np.mean(peak))}


def evaluate(agent, eval_envs, task_names, current_task, episodes, history):
    results = []
    for task_id in range(current_task + 1):
        returns, successes = [], []
        for _ in range(episodes):
            observation = reset_env(eval_envs[task_id])
            total_return, success, done = 0.0, 0.0, False
            while not done:
                action = np.asarray(agent.sample_actions(
                    observation, task_id=task_id, temperature=0.0), np.float32).reshape(-1)
                observation, reward, done, info, _ = step_env(eval_envs[task_id], action)
                total_return += reward
                success = max(success, float(info.get("success", 0.0)))
            returns.append(total_return)
            successes.append(success)
        results.append({"eval_task_index": task_id, "task_name": task_names[task_id],
                        "success": float(np.mean(successes)),
                        "return": float(np.mean(returns)), "episodes": episodes})
    scores = [row["success"] for row in results]
    return {"avg_seen": float(np.mean(scores)), "results": results,
            **forgetting_metrics(history, scores)}


def write_matrix(root, progress, task_names):
    path = root / "task_success_matrix.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = ["trained_task_index", "total_env_steps"] + task_names + [
        "avg_seen", "endpoint_forgetting", "peak_forgetting"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in progress["endpoint_evaluations"]:
            values = {"trained_task_index": row["task_index"],
                      "total_env_steps": row["total_env_steps"],
                      "avg_seen": row["avg_seen"],
                      "endpoint_forgetting": row["endpoint_forgetting"],
                      "peak_forgetting": row["peak_forgetting"]}
            values.update({item["task_name"]: item["success"] for item in row["results"]})
            writer.writerow(values)
    os.replace(temporary, path)


def condition_ids_for(task_names, mode):
    """Map task positions to condition indices (identity, or shared per environment name)."""
    if mode == "position":
        return list(range(len(task_names)))
    order = []
    for name in task_names:
        if name not in order:
            order.append(name)
    return [order.index(name) for name in task_names]


def build_config(args, task_names):
    config = {
        "schema_version": 1, "algorithm": "permanent_taskwise_magnitude_sac",
        "env_name": args.env_name, "seed": args.seed, "task_names": task_names,
        "num_tasks": len(task_names), "raw_observation_dim": 12, "action_dim": 4,
        "task_conditioning": "fixed_num_tasks_one_hot_in_learner",
        "actor_hidden_dims": [args.actor_width] * args.hidden_layers,
        "critic_hidden_dims": [args.critic_width] * args.hidden_layers,
        "learning_rate": 3e-4, "discount": 0.99, "tau": 0.005,
        "target_entropy": -2.0, "initial_log_temperature": args.init_log_temperature,
        "max_grad_norm": 1.0, "steps_per_task": args.steps_per_task,
        "start_training": args.start_training, "batch_size": args.batch_size,
        "buffer_size": args.buffer_size, "updates_per_step_after_warmup": 1,
        "env_randomization": "random_init_all", "training_reward_scale": 1.0 / 200,
        "evaluation_reward_scale": 1.0, "random_exploration_each_task": True,
        "clear_replay_each_task": True, "reset_actor_optimizer_each_task": bool(args.reset_actor_optimizer),
        "reset_critic_each_task": bool(args.reset_critic),
        "reset_temperature_each_task": args.temperature_mode == "shared_reset",
        "temperature_mode": args.temperature_mode,
        "critic_warmup_updates": args.critic_warmup_updates,
        "early_eval_updates": list(args.early_eval_updates),
        "protocol_tag": args.protocol_tag,
        "source_variant": "cw_v2_protocol_ablation",
        "importance_mode": args.importance_mode, "importance_beta": args.importance_beta,
        "scale_floor": args.scale_floor, "importance_samples": args.importance_samples,
        "importance_accumulate": args.importance_accumulate,
        "anchor_mode": args.anchor_mode, "anchor_coef": args.anchor_coef,
        "anchor_states_per_task": args.anchor_states_per_task, "anchor_batch": args.anchor_batch,
        "actor_lr_ramp_updates": args.actor_lr_ramp_updates,
        "task_condition": args.task_condition, "anchor_budget": args.anchor_budget,
        "anchor_target": args.anchor_target, "log_temperature_min": args.log_temperature_min,
        "task_code_scale": args.task_code_scale,
        "condition_ids": condition_ids_for(task_names, args.task_condition),
        "population_size": 1, "distillation": False, "evolution": False,
        "forward_mask": False, "hard_freezing": False,
        "decay_factor": args.decay_factor, "mask_fraction": args.mask_fraction,
        "refresh_interval_actor_steps": args.refresh_interval,
        "mask_scope": "each_actor_dense_kernel_including_both_policy_heads",
        "selection_rule": "per_kernel_top_abs_ceil_fraction_fixed_index_ties",
        "count_rule": "once_per_task_on_first_training_time_selection_including_task1",
        "history_rule": "training_time_union_not_task_end_mask",
        "lr_multiplier": "decay_factor ** persistent_selection_count",
        "mask_exit_restores_lr": False, "minimum_lr_multiplier": None,
        "decay_factor_status": "first_run_hypothesis_not_tuned_optimum",
        "eval_interval": args.eval_interval, "eval_episodes": args.eval_episodes,
        "eval_policy": "deterministic_all_seen_tasks_episode_max_success",
        "forgetting_reference": "task_end_evaluations_previous_tasks_mean_allow_negative",
        "checkpoint_semantics": "episode_boundary_no_inflight_simulator_state",
    }
    # Leave legacy defaults byte-for-byte compatible with completed runs.
    if args.actor_variant != "legacy" or args.zero_grad_patience or args.retain_task_checkpoints:
        config.update({
            "actor_variant": args.actor_variant,
            "head_feature_normalization": "affine_free_layer_norm" if args.actor_variant == "stable" else None,
            "mean_parameterization": "linear" if args.actor_variant == "stable" else "inner_tanh",
            "std_parameterization": "exp_clipped_log_std" if args.actor_variant == "stable" else "softplus_bounded_tanh",
            "stable_std_range_matches_legacy": args.actor_variant == "stable",
            "zero_grad_patience": args.zero_grad_patience,
            "gradient_guard_semantics": "every_update_exact_raw_gradient_nonzero_count_before_clip_and_adam",
            "retain_task_checkpoints": args.retain_task_checkpoints,
            "task_limit": args.task_limit or len(task_names),
            "planned_env_steps": (args.task_limit or len(task_names)) * args.steps_per_task,
            "failure_checkpoint_semantics": "separate_mid_episode_diagnostic_only_pre_and_post_update_states_and_batch",
        })
    return config


def main(argv=None):
    args = parse_args(argv)
    segment_start = time.monotonic()
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / ".trainer.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another trainer is already using " + str(root))

    import jax
    from continual_world import TASK_SEQS, get_single_env

    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("--require-gpu requested, but JAX has no GPU device")
    if args.env_name not in TASK_SEQS:
        raise ValueError("Unknown task sequence: " + args.env_name)
    sequence = TASK_SEQS[args.env_name]
    task_names = [item["task"] if isinstance(item, dict) else item for item in sequence]
    limit = args.task_limit or len(task_names)
    if limit > len(task_names):
        raise ValueError("task-limit exceeds task sequence length")
    config = build_config(args, task_names)
    latest = root / "checkpoints" / "latest.pkl"
    resume_path = latest if args.resume == "auto" else (
        None if args.resume == "none" else Path(args.resume).expanduser().resolve())
    restore = resume_path is not None and resume_path.is_file()
    if args.resume not in ("auto", "none") and not restore:
        raise FileNotFoundError(resume_path)
    existing_outputs = [root / name for name in LOG_FILES + ("config.json", "status.json")]
    if (root / "failure.json").exists():
        raise RuntimeError("This run has a saved failure; inspect failure.json before selecting a new run directory")
    if not restore and any(path.exists() for path in existing_outputs):
        raise FileExistsError("Run has existing results but no selected checkpoint: " + str(root))

    np.random.seed(args.seed)
    random.seed(args.seed)
    learner_path = Path(__file__).resolve().parent / "jaxrl/agents/sac/cumulative_learner.py"
    spec = importlib.util.spec_from_file_location("cumulative_sac_learner", learner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    temporary_env = get_single_env(task_names[0], args.seed,
                                   randomization="random_init_all")
    observation_shape = temporary_env.observation_space.shape
    action_shape = temporary_env.action_space.shape
    if observation_shape != (12,) or action_shape != (4,):
        raise ValueError("Expected benchmark raw obs/action (12,)/(4,), got " +
                         str((observation_shape, action_shape)))
    agent = module.CumulativeSACLearner(
        seed=args.seed, observations=np.zeros((1,) + observation_shape, np.float32),
        actions=np.zeros((1,) + action_shape, np.float32), num_tasks=len(task_names),
        actor_hidden_dims=tuple(config["actor_hidden_dims"]),
        critic_hidden_dims=tuple(config["critic_hidden_dims"]), learning_rate=3e-4,
        decay_factor=args.decay_factor, mask_fraction=args.mask_fraction,
        refresh_interval=args.refresh_interval, target_entropy=-2.0,
        init_temperature=args.init_log_temperature,
        reset_actor_optimizer=bool(args.reset_actor_optimizer),
        reset_critic=bool(args.reset_critic),
        reset_temperature=args.temperature_mode == "shared_reset",
        actor_variant=args.actor_variant,
        temperature_mode=args.temperature_mode,
        critic_warmup_updates=args.critic_warmup_updates,
        importance_mode=args.importance_mode, importance_beta=args.importance_beta,
        scale_floor=args.scale_floor, importance_samples=args.importance_samples,
        importance_accumulate=args.importance_accumulate,
        anchor_mode=args.anchor_mode, anchor_coef=args.anchor_coef,
        anchor_states_per_task=args.anchor_states_per_task, anchor_batch=args.anchor_batch,
        actor_lr_ramp_updates=args.actor_lr_ramp_updates,
        condition_ids=tuple(condition_ids_for(task_names, args.task_condition)),
        anchor_budget=args.anchor_budget, anchor_target=args.anchor_target,
        log_temperature_min=args.log_temperature_min, task_code_scale=args.task_code_scale)
    temporary_env.close()
    eval_envs = [get_single_env(name, args.seed, randomization="random_init_all",
                                normalize_reward=False) for name in task_names]
    progress = {"task_index": 0, "task_steps": 0, "total_env_steps": 0,
                "actor_updates": 0, "episodes": 0, "task_episodes": 0,
                "task_active": False, "next_eval_step": args.eval_interval,
                "next_log_step": args.log_interval,
                "next_checkpoint_step": args.checkpoint_interval,
                "endpoint_evaluations": [], "success_history": [],
                "last_train": None, "diagnostics": {}, "elapsed_seconds": 0.0,
                "zero_grad_streak": 0, "max_zero_grad_streak": 0,
                "zero_grad_update_count": 0, "checked_actor_updates": 0,
                "early_evals_done": []}
    restored = None
    if restore:
        with resume_path.open("rb") as handle:
            restored = pickle.load(handle)
        if restored["config"] != config:
            raise ValueError("Resume algorithm/evaluation configuration differs from checkpoint")
        agent.load_state_dict(restored["learner"])
        progress = restored["progress"]
        for name in ("zero_grad_streak", "max_zero_grad_streak",
                     "zero_grad_update_count", "checked_actor_updates"):
            progress.setdefault(name, 0)
        progress.setdefault("early_evals_done", [])
        for env, state in zip(eval_envs, restored["eval_rngs"]):
            restore_env_rng(env, state)
        for name in LOG_FILES:
            path = root / name
            offset = restored["log_offsets"].get(name, 0)
            if offset and (not path.exists() or path.stat().st_size < offset):
                raise RuntimeError("Checkpoint log is missing/truncated: " + str(path))
            if path.exists():
                with path.open("r+b") as handle:
                    handle.truncate(offset)
        np.random.set_state(restored["numpy_rng"])
        random.setstate(restored["python_rng"])
        print("CUMULATIVE_RESUME " + json.dumps({
            key: progress[key] for key in ("task_index", "task_steps", "actor_updates",
                                           "total_env_steps", "task_active")}), flush=True)
    else:
        atomic_json(root / "config.json", config)
    write_matrix(root, progress, task_names)
    runtime = {"devices": [str(device) for device in jax.devices()],
               "jax_version": jax.__version__, "numpy_version": np.__version__,
               "argv": sys.argv if argv is None else argv,
               "walltime_seconds": args.walltime_seconds, "task_limit": limit,
               "checkpoint_interval": args.checkpoint_interval,
               "log_interval": args.log_interval, "started_at": time.time()}
    atomic_json(root / "runtime.json", runtime)

    env, replay, latest_update = None, None, {}
    last_episode_seconds, last_eval_seconds = 0.0, 0.0
    previous_elapsed = progress["elapsed_seconds"]

    def status(phase):
        value = {key: progress[key] for key in (
            "task_index", "task_steps", "total_env_steps", "actor_updates",
            "episodes", "task_active", "last_train", "diagnostics")}
        value.update({"phase": phase, "status": phase,
                      "completed_tasks": len(progress["success_history"]),
                      "task_limit": limit, "num_tasks": len(task_names),
                      "elapsed_seconds": previous_elapsed + time.monotonic() - segment_start,
                      "segment_seconds": time.monotonic() - segment_start,
                      "updated_at": time.time(), "latest_checkpoint": str(latest),
                      "checkpoint_boundary": "episode_boundary",
                      "zero_grad_patience": args.zero_grad_patience,
                      **{name: progress[name] for name in ("zero_grad_streak", "max_zero_grad_streak",
                          "zero_grad_update_count", "checked_actor_updates")},
                      "run_dir": str(root)})
        atomic_json(root / "status.json", value)
        return value

    def checkpoint(phase):
        latest.parent.mkdir(parents=True, exist_ok=True)
        offsets = {}
        for name in LOG_FILES:
            path = root / name
            if path.exists():
                with path.open("ab") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                offsets[name] = path.stat().st_size
        progress["elapsed_seconds"] = previous_elapsed + time.monotonic() - segment_start
        payload = {"version": 1, "config": config,
                   "learner": jax.device_get(agent.state_dict()),
                   "progress": copy.deepcopy(progress),
                   "replay": replay.state_dict() if replay is not None and
                   progress["task_steps"] < args.steps_per_task else None,
                   "train_env_rng": env_rng_state(env) if env is not None else None,
                   "eval_rngs": [env_rng_state(item) for item in eval_envs],
                   "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                   "log_offsets": offsets}
        temporary = latest.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, latest)
        status(phase)
        print("CUMULATIVE_CHECKPOINT " + json.dumps({
            "phase": phase, "task_index": progress["task_index"],
            "task_steps": progress["task_steps"],
            "total_env_steps": progress["total_env_steps"],
            "actor_updates": progress["actor_updates"]}), flush=True)

    def archive_checkpoint(label):
        if not args.retain_task_checkpoints:
            return
        archive = latest.parent / (label + ".pkl")
        # latest is replaced atomically, never modified in place, so this link
        # remains an immutable copy without duplicating current checkpoint IO.
        os.link(latest, archive)
        print("CUMULATIVE_CHECKPOINT_ARCHIVE " + str(archive), flush=True)

    def save_update_failure(reason, batch, before_state, before_counters):
        failure_dir = root / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        artifact = failure_dir / ("task_{:02d}_step_{:09d}_{}.pkl".format(
            progress["task_index"], progress["task_steps"], reason))
        before = {"format_version": 1, "config": copy.deepcopy(agent.config),
                  **before_counters,
                  "learner_state": jax.device_get(module.serialization.to_state_dict(before_state))}
        failure_metrics = {}
        for name, value in jax.device_get(latest_update).items():
            number = float(np.asarray(value).reshape(()))
            failure_metrics[name] = number if math.isfinite(number) else None
        metadata = {"reason": reason, "time": time.time(),
                    "task_index": progress["task_index"], "task_steps": progress["task_steps"],
                    "total_env_steps": progress["total_env_steps"],
                    "actor_updates": progress["actor_updates"],
                    "zero_grad_streak": progress["zero_grad_streak"],
                    "max_zero_grad_streak": progress["max_zero_grad_streak"],
                    "checked_actor_updates": progress["checked_actor_updates"],
                    "metrics": failure_metrics, "artifact": str(artifact),
                    "diagnostic_only": True, "resume_allowed": False,
                    "checkpoint_boundary": "mid_episode_no_simulator_state",
                    "last_resumable_checkpoint": str(latest),
                    "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "learner_sha256": hashlib.sha256(learner_path.read_bytes()).hexdigest()}
        payload = {"version": 1, "diagnostic_only": True, "resume_allowed": False,
                   "metadata": metadata, "config": config,
                   "pre_update_learner": before,
                   "post_update_learner": jax.device_get(agent.state_dict()),
                   "trigger_batch": copy.deepcopy(batch),
                   "progress": copy.deepcopy(progress), "replay": replay.state_dict(),
                   "train_env_rng": env_rng_state(env),
                   "eval_rngs": [env_rng_state(item) for item in eval_envs],
                   "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        temporary = artifact.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, artifact)
        atomic_json(root / "failure.json", metadata)
        value = status("failed_" + reason)
        value.update(error=reason, failure_artifact=str(artifact))
        atomic_json(root / "status.json", value)
        print("CUMULATIVE_UPDATE_FAILURE " + json.dumps(metadata, allow_nan=False), flush=True)

    def guard():
        margin = max(180.0, 3 * last_episode_seconds + 30,
                     1.5 * last_eval_seconds + 60)
        return STOP_REQUESTED or (
            time.monotonic() - segment_start >= args.walltime_seconds - margin)

    def log_training(episode):
        metrics = scalar_metrics(latest_update, jax)
        diagnostics = agent.diagnostic_stats()
        row = {"task_index": progress["task_index"], "task_steps": progress["task_steps"],
               "total_env_steps": progress["total_env_steps"],
               "actor_updates": progress["actor_updates"], "episodes": progress["episodes"],
               "episode_return": episode["return"], "episode_success": episode["success"],
               "metrics": metrics, "diagnostics": diagnostics, "time": time.time()}
        append_json(root / "train.jsonl", row)
        append_json(root / "diagnostics.jsonl", {
            key: row[key] for key in ("task_index", "task_steps", "total_env_steps",
                                      "actor_updates", "diagnostics", "time")})
        csv_path = root / "training.csv"
        fields = ("task_index", "task_steps", "total_env_steps", "actor_updates", "episodes",
                  "episode_return", "episode_success", "metrics")
        needs_header = not csv_path.exists() or not csv_path.stat().st_size
        with csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if needs_header:
                writer.writeheader()
            writer.writerow({key: json.dumps(row[key]) if key == "metrics" else row[key]
                             for key in fields})
        progress["last_train"], progress["diagnostics"] = row, diagnostics
        status("running")
        print("CUMULATIVE_TRAIN " + json.dumps(row, allow_nan=False), flush=True)

    def early_eval(task_id, updates_in_task):
        """Dense post-switch evaluation of every seen task at a fixed actor-update count."""
        row = evaluate(agent, eval_envs, task_names, task_id, args.eval_episodes,
                       progress["success_history"])
        row.update({"phase": "early", "task_index": task_id, "updates_in_task": updates_in_task,
                    "actor_trained_so_far": int(updates_in_task) if not agent.in_critic_warmup else 0,
                    "task_steps": progress["task_steps"], "total_env_steps": progress["total_env_steps"]})
        append_json(root / "evaluations.jsonl", row)
        print("CUMULATIVE_EVAL " + json.dumps(row), flush=True)
        progress["early_evals_done"].append(int(updates_in_task))

    def request_stop(signum, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    status("initializing")
    try:
        while progress["task_index"] < limit:
            task_id = progress["task_index"]
            env = get_single_env(task_names[task_id], args.seed,
                                 randomization="random_init_all", normalize_reward=True)
            replay = Replay(args.buffer_size, observation_shape, action_shape)
            if progress["task_active"]:
                if restored is None:
                    raise RuntimeError("Active task has no restoring checkpoint")
                if restored["replay"] is not None:
                    replay.load_state_dict(restored["replay"])
                if restored["train_env_rng"] is not None:
                    restore_env_rng(env, restored["train_env_rng"])
                # Environment construction may consume global RNGs; restore after it.
                np.random.set_state(restored["numpy_rng"])
                random.setstate(restored["python_rng"])
                restored = None
            else:
                agent.begin_task(task_id)
                progress["task_active"] = True
                progress["task_steps"], progress["task_episodes"] = 0, 0
                progress["next_eval_step"] = args.eval_interval
                progress["zero_grad_streak"] = 0
                progress["early_evals_done"] = []
                latest_update = {}
                checkpoint("task_started")
                archive_checkpoint("task_{:02d}_start".format(task_id))
                if task_id > 0 and 0 in args.early_eval_updates:
                    early_eval(task_id, 0)
            status("running")
            while progress["task_steps"] < args.steps_per_task:
                if guard():
                    checkpoint("continuation_required")
                    return 75
                # A walltime exit can occur after an episode but before its due
                # evaluation. Complete that evaluation before collecting new data.
                if progress["task_steps"] >= progress["next_eval_step"]:
                    eval_start = time.monotonic()
                    row = evaluate(agent, eval_envs, task_names, task_id,
                                   args.eval_episodes, progress["success_history"])
                    row.update({"phase": "periodic", "task_index": task_id,
                                "task_steps": progress["task_steps"],
                                "total_env_steps": progress["total_env_steps"]})
                    append_json(root / "evaluations.jsonl", row)
                    print("CUMULATIVE_EVAL " + json.dumps(row), flush=True)
                    last_eval_seconds = time.monotonic() - eval_start
                    progress["next_eval_step"] = (
                        progress["task_steps"] // args.eval_interval + 1) * args.eval_interval
                episode_start = time.monotonic()
                observation = reset_env(env)
                ep_return, ep_success, ep_length, done = 0.0, 0.0, 0, False
                while not done and progress["task_steps"] < args.steps_per_task:
                    if progress["task_steps"] < args.start_training:
                        action = np.asarray(env.action_space.sample(), np.float32)
                    else:
                        action = np.asarray(agent.sample_actions(
                            observation, task_id=task_id, temperature=1.0),
                            np.float32).reshape(action_shape)
                    next_observation, reward, done, info, mask = step_env(env, action)
                    replay.insert(observation, action, reward, mask, next_observation)
                    observation = next_observation
                    ep_return += reward
                    ep_success = max(ep_success, float(info.get("success", 0.0)))
                    ep_length += 1
                    progress["task_steps"] += 1
                    progress["total_env_steps"] += 1
                    if (progress["task_steps"] >= args.start_training and
                            replay.size >= args.batch_size):
                        training_batch = replay.sample(args.batch_size)
                        if args.zero_grad_patience:
                            before_state = agent._state
                            before_counters = {name: getattr(agent, name) for name in (
                                "current_task", "total_updates", "task_updates", "last_refresh_update")}
                        latest_update = agent.update(training_batch)
                        progress["actor_updates"] += 1
                        if task_id > 0 and args.early_eval_updates:
                            updates_in_task = agent.task_actor_updates
                            if (updates_in_task in args.early_eval_updates
                                    and updates_in_task not in progress["early_evals_done"]):
                                early_eval(task_id, updates_in_task)
                        if args.zero_grad_patience:
                            health = jax.device_get(tuple(latest_update[name] for name in (
                                "nonfinite", "actor_grad_nonzero_count", "actor_grad_max_abs")))
                            reason = record_update_health(*health, progress, args.zero_grad_patience)
                            if reason is not None:
                                save_update_failure(reason, training_batch, before_state, before_counters)
                                return 2
                progress["episodes"] += 1
                progress["task_episodes"] += 1
                last_episode_seconds = time.monotonic() - episode_start
                episode = {"task_index": task_id, "task_name": task_names[task_id],
                           "task_steps": progress["task_steps"],
                           "total_env_steps": progress["total_env_steps"],
                           "actor_updates": progress["actor_updates"],
                           "episode": progress["episodes"], "return": ep_return,
                           "success": ep_success, "length": ep_length,
                           "budget_truncated": not done, "seconds": last_episode_seconds}
                append_json(root / "episodes.jsonl", episode)
                task_end = progress["task_steps"] >= args.steps_per_task
                if progress["total_env_steps"] >= progress["next_log_step"] or task_end:
                    log_training(episode)
                    progress["next_log_step"] = (
                        progress["total_env_steps"] // args.log_interval + 1) * args.log_interval
                if guard():
                    checkpoint("continuation_required")
                    return 75
                if (progress["task_steps"] >= progress["next_eval_step"] and not task_end):
                    eval_start = time.monotonic()
                    row = evaluate(agent, eval_envs, task_names, task_id,
                                   args.eval_episodes, progress["success_history"])
                    row.update({"phase": "periodic", "task_index": task_id,
                                "task_steps": progress["task_steps"],
                                "total_env_steps": progress["total_env_steps"]})
                    append_json(root / "evaluations.jsonl", row)
                    print("CUMULATIVE_EVAL " + json.dumps(row), flush=True)
                    last_eval_seconds = time.monotonic() - eval_start
                    progress["next_eval_step"] = (
                        progress["task_steps"] // args.eval_interval + 1) * args.eval_interval
                if progress["total_env_steps"] >= progress["next_checkpoint_step"] and not task_end:
                    progress["next_checkpoint_step"] = (
                        progress["total_env_steps"] // args.checkpoint_interval + 1) * args.checkpoint_interval
                    checkpoint("running")
            if guard():
                checkpoint("continuation_required")
                return 75
            eval_start = time.monotonic()
            row = evaluate(agent, eval_envs, task_names, task_id,
                           args.eval_episodes, progress["success_history"])
            row.update({"phase": "task_end", "task_index": task_id,
                        "task_steps": progress["task_steps"],
                        "total_env_steps": progress["total_env_steps"]})
            append_json(root / "evaluations.jsonl", row)
            print("CUMULATIVE_EVAL " + json.dumps(row), flush=True)
            last_eval_seconds = time.monotonic() - eval_start
            progress["endpoint_evaluations"].append(row)
            progress["success_history"].append([item["success"] for item in row["results"]])
            write_matrix(root, progress, task_names)
            if args.importance_mode != "magnitude" or args.anchor_mode != "none":
                # E1/E2: task-end importance and anchors from this task's own replay, before the
                # boundary checkpoint so they are persisted with the learner state.
                sample_count = max(args.importance_samples, args.anchor_states_per_task)
                finish_rng = np.random.RandomState(args.seed * 1000 + task_id)
                finish_info = agent.finish_task(replay.sample_observations(sample_count, finish_rng))
                append_json(root / "diagnostics.jsonl", {
                    "phase": "finish_task", "task_index": task_id,
                    "total_env_steps": progress["total_env_steps"], **finish_info,
                    "diagnostics": agent.diagnostic_stats(), "time": time.time()})
                print("CUMULATIVE_FINISH_TASK " + json.dumps({"task_index": task_id, **finish_info}), flush=True)
            env.close()
            env, replay = None, None
            progress["task_index"] += 1
            progress["task_steps"], progress["task_episodes"] = 0, 0
            progress["task_active"] = False
            progress["next_checkpoint_step"] = (
                progress["total_env_steps"] // args.checkpoint_interval + 1) * args.checkpoint_interval
            checkpoint("completed" if progress["task_index"] == len(task_names)
                       else "task_boundary")
            archive_checkpoint("task_{:02d}_end".format(task_id))
        status("completed" if progress["task_index"] == len(task_names)
               else "task_limit_reached")
        return 0
    except Exception as error:
        value = status("failed")
        value["error"] = repr(error)
        atomic_json(root / "status.json", value)
        raise
    finally:
        if env is not None:
            env.close()
        for item in eval_envs:
            item.close()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
