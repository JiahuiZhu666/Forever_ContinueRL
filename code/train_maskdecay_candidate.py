#!/usr/bin/env python3
"""Train ONE mask-decay candidate on ONE task from a common start state, then evaluate it.

Protocol (method document v1.1): dense actor, full gradients; actor increment = scales * Adam(clip(g));
scales = lambda ** max(h + M - 1, 0) from the candidate's weight mask M and the history counts h, fixed
for the whole task. Task boundary: inherit actor / critics / targets / temperature of the start state,
re-initialise every optimizer, empty replay, 10k random steps. Final selection score: mean return on
the current task over E paired initial conditions (same seed for every candidate).
"""
import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import jax  # noqa: E402
from flax.core import unfreeze  # noqa: E402

from continual_world import TASK_SEQS, get_single_env  # noqa: E402
from train_cumulative import Replay, evaluate, reset_env, step_env, append_json, atomic_json  # noqa: E402
import maskdecay as md  # noqa: E402


def load_learner_class():
    spec = importlib.util.spec_from_file_location(
        "cumulative_sac_learner", os.path.join(HERE, "jaxrl", "agents", "sac", "cumulative_learner.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CumulativeSACLearner


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--env-name", default="cw10")
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--start-state", default="none", help="pickle of the previous winner's learner state_dict, or none")
    p.add_argument("--neuron-mask", required=True, help="npz with layer_i arrays (candidate neuron mask)")
    p.add_argument("--history-counts", default="none", help="npz tree of per-weight history counts h, or none")
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=110)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--start-training", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--eval-interval", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-seed", type=int, default=20_000)
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--actor-width", type=int, default=1024)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--critic-width", type=int, default=256)
    p.add_argument("--actor-variant", choices=("legacy", "stable"), default="stable")
    p.add_argument("--init-log-temperature", type=float, default=1.0)
    p.add_argument("--forward-masking", type=int, choices=(0, 1), default=1,
                   help="1: Nevo-style masked forward (task's policy = its sub-network); 0: dense actor (doc v1.1)")
    p.add_argument("--include-biases", type=int, choices=(0, 1), default=1,
                   help="1: hidden-layer biases follow their neuron mask in selection/counting/decay")
    p.add_argument("--anchor-coef", type=float, default=0.0,
                   help=">0: add anchor_coef * KL(pi_old || pi_theta) on stored old-task states to the actor loss")
    p.add_argument("--anchor-states-per-task", type=int, default=4096)
    p.add_argument("--anchor-batch", type=int, default=256)
    p.add_argument("--actor-optimizer", choices=("adam", "sgd"), default="adam",
                   help="actor optimizer (critics/temperature stay Adam); sgd = plain SGD without momentum")
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--early-stop-success", type=float, default=None,
                   help="if set: when a periodic eval reaches this current-task success, run a confirmation eval "
                        "(--confirm-episodes, fresh seeds); if that also reaches it, stop training (--steps is the cap)")
    p.add_argument("--confirm-episodes", type=int, default=50)
    p.add_argument("--confirm-seed", type=int, default=30_000)
    p.add_argument("--stop-file", default="none",
                   help="per-task file shared by the K candidates: the first to confirm creates it (atomically) and "
                        "wins; the others stop as soon as they see it")
    p.add_argument("--require-gpu", action="store_true")
    return p.parse_args(argv)


def evaluate_current(agent, env, task_id, episodes, seed):
    """Deterministic success/return on the current task only, with a fresh seed sequence (confirmation eval)."""
    env.seed(seed + task_id)
    env.action_space.seed(seed + task_id)
    returns, successes = [], []
    for _ in range(episodes):
        observation = reset_env(env)
        total_return, success, done = 0.0, 0.0, False
        while not done:
            action = np.asarray(agent.sample_actions(observation, task_id=task_id, temperature=0.0),
                                np.float32).reshape(-1)
            observation, reward, done, info, _ = step_env(env, action)
            total_return += reward
            success = max(success, float(info.get("success", 0.0)))
        returns.append(total_return)
        successes.append(success)
    return {"success": float(np.mean(successes)), "return": float(np.mean(returns)), "episodes": episodes, "seed": seed}


def claim_stop_file(path, owner):
    """Atomically create the shared stop file. True if this candidate created it (it is the winner)."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"owner": owner, "time": time.time()}, f)
    return True


def evaluate_paired(agent, eval_envs, task_names, current_task, episodes, seed):
    """Same initial-condition sequence for every candidate: re-seed each eval env first."""
    results = []
    for task_id in range(current_task + 1):
        env = eval_envs[task_id]
        env.seed(seed + task_id)
        env.action_space.seed(seed + task_id)
        returns, successes = [], []
        for _ in range(episodes):
            observation = reset_env(env)
            total_return, success, done = 0.0, 0.0, False
            while not done:
                action = np.asarray(agent.sample_actions(observation, task_id=task_id, temperature=0.0),
                                    np.float32).reshape(-1)
                observation, reward, done, info, _ = step_env(env, action)
                total_return += reward
                success = max(success, float(info.get("success", 0.0)))
            returns.append(total_return)
            successes.append(success)
        results.append({"eval_task_index": task_id, "task_name": task_names[task_id],
                        "success": float(np.mean(successes)), "return": float(np.mean(returns)),
                        "return_std": float(np.std(returns)), "episodes": episodes})
    return results


def main(argv=None):
    args = parse_args(argv)
    root = args.run_dir
    os.makedirs(root, exist_ok=True)
    t0 = time.monotonic()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU backend required")
    seq = TASK_SEQS[args.env_name]
    task_names = [d["task"] if isinstance(d, dict) else d for d in seq]
    t = args.task_index
    if not 0 <= t < len(task_names):
        raise ValueError("task index out of range")
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_env = get_single_env(task_names[t], args.seed, randomization="random_init_all", normalize_reward=True)
    eval_envs = [get_single_env(name, args.seed, randomization="random_init_all", normalize_reward=False)
                 for name in task_names[:t + 1]]
    observation_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape

    Learner = load_learner_class()
    agent = Learner(
        seed=args.seed, observations=np.zeros((1,) + observation_shape, np.float32),
        actions=np.zeros((1,) + action_shape, np.float32), num_tasks=len(task_names),
        actor_hidden_dims=(args.actor_width,) * args.hidden_layers,
        critic_hidden_dims=(args.critic_width,) * args.hidden_layers,
        learning_rate=3e-4, decay_factor=1.0, mask_fraction=0.1, refresh_interval=10_000,
        reset_critic=False, reset_actor_optimizer=True, reset_temperature=False,
        target_entropy=-2.0, init_temperature=args.init_log_temperature, actor_variant=args.actor_variant,
        temperature_mode="shared_keep", critic_warmup_updates=0, importance_mode="external",
        anchor_mode="kl" if args.anchor_coef > 0 else "none", anchor_coef=max(args.anchor_coef, 1e-12),
        anchor_states_per_task=args.anchor_states_per_task, anchor_batch=args.anchor_batch, anchor_budget=0,
        reset_all_optimizers=True, forward_masking=bool(args.forward_masking),
        actor_optimizer=args.actor_optimizer, actor_learning_rate=args.actor_lr)
    if args.start_state != "none":
        with open(args.start_state, "rb") as f:
            agent.load_state_dict(pickle.load(f))
    agent.begin_task(t)  # inherits parameters, restarts all optimizers (reset_all_optimizers)

    # Candidate weight mask, history counts, multipliers.
    actor_tree = md.to_plain_tree(jax.device_get(unfreeze(agent._state.actor_params)))
    neuron_mask = md.load_neuron_mask(args.neuron_mask)
    if args.forward_masking:
        agent.set_task_mask(t, np.stack(neuron_mask))  # old tasks' masks come with the start state
    specs = md.kernel_specs_for(args.hidden_layers, include_biases=bool(args.include_biases))
    weight_mask = md.build_weight_mask(actor_tree, neuron_mask, specs)
    counts = md.load_tree_npz(args.history_counts) if args.history_counts != "none" else md.init_counts(actor_tree)
    scales = md.candidate_scales(counts, weight_mask, args.lam)
    agent.set_scales(scales)
    stats = md.scale_statistics(weight_mask, counts, scales, specs)
    config = {"args": vars(args), "task_name": task_names[t], "learner_config": agent.config,
              "neuron_mask_density": [float(np.mean(a)) for a in neuron_mask],
              "neuron_mask_counts": [int(np.sum(a)) for a in neuron_mask],
              "weight_mask_stats": stats, "jax_backend": jax.default_backend()}
    atomic_json(os.path.join(root, "config.json"), config)
    print("MASKDECAY_CANDIDATE_START " + json.dumps({"task": t, "lam": args.lam, "seed": args.seed,
          "density": config["neuron_mask_density"], "selected_fraction": stats["total"]["selected_fraction"],
          "scaled_fraction": stats["total"]["scaled_fraction"]}), flush=True)

    replay = Replay(args.buffer_size, observation_shape, action_shape)
    steps, episodes, updates = 0, 0, 0
    next_eval, next_log = args.eval_interval, args.log_interval
    latest = {}
    stop_file = None if args.stop_file == "none" else args.stop_file
    stop_reason, confirmations = "cap", []
    while steps < args.steps:
        if stop_file is not None and os.path.exists(stop_file):
            # Another candidate of this task confirmed first: stop without final eval / state.
            with open(stop_file) as f:
                owner = json.load(f).get("owner")
            result = {"status": "stopped_by_peer", "task_index": t, "task_name": task_names[t], "stopped_by": owner,
                      "env_steps": steps, "actor_updates": updates, "episodes": episodes, "seed": args.seed,
                      "confirmations": confirmations, "seconds": time.monotonic() - t0}
            atomic_json(os.path.join(root, "result.json"), result)
            print("MASKDECAY_CANDIDATE_STOPPED " + json.dumps({"task_index": t, "env_steps": steps, "by": owner}), flush=True)
            for env in eval_envs + [train_env]:
                env.close()
            return 0
        observation = reset_env(train_env)
        ep_return, ep_success, done = 0.0, 0.0, False
        while not done and steps < args.steps:
            if steps < args.start_training:
                action = np.asarray(train_env.action_space.sample(), np.float32)
            else:
                action = np.asarray(agent.sample_actions(observation, task_id=t, temperature=1.0),
                                    np.float32).reshape(action_shape)
            next_observation, reward, done, info, mask = step_env(train_env, action)
            replay.insert(observation, action, reward, mask, next_observation)
            observation = next_observation
            ep_return += reward
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            steps += 1
            if steps >= args.start_training and replay.size >= args.batch_size:
                latest = agent.update(replay.sample(args.batch_size))
                updates += 1
        episodes += 1
        if steps >= next_log or steps >= args.steps:
            metrics = {}
            if latest:
                keys = ("actor_loss", "critic_loss", "temperature", "entropy", "q1", "anchor_loss",
                        "actor_update_norm_ratio", "actor_raw_update_norm", "actor_scaled_update_norm", "nonfinite",
                        "actor_grad_norm")
                values = jax.device_get(tuple(latest[k] for k in keys if k in latest))
                metrics = {k: float(np.asarray(v)) for k, v in zip([k for k in keys if k in latest], values)}
            row = {"task_index": t, "task_steps": steps, "actor_updates": updates, "episodes": episodes,
                   "episode_return": ep_return, "episode_success": ep_success,
                   "elapsed_seconds": time.monotonic() - t0, "metrics": metrics}
            append_json(os.path.join(root, "train.jsonl"), row)
            next_log = (steps // args.log_interval + 1) * args.log_interval
            if metrics.get("nonfinite", 0.0) > 0:
                raise FloatingPointError("non-finite SAC statistics")
        if steps >= next_eval and steps < args.steps:
            row = evaluate(agent, eval_envs, task_names, t, args.eval_episodes, [])
            row.update({"phase": "periodic", "task_index": t, "task_steps": steps})
            append_json(os.path.join(root, "evaluations.jsonl"), row)
            print("MASKDECAY_EVAL " + json.dumps({"task_steps": steps, "results": [
                (r["task_name"], r["success"]) for r in row["results"]]}), flush=True)
            next_eval = (steps // args.eval_interval + 1) * args.eval_interval
            current = [r for r in row["results"] if r["eval_task_index"] == t][0]["success"]
            if args.early_stop_success is not None and current >= args.early_stop_success:
                # Confirmation with fresh seeds (a different sequence each time it is attempted).
                check = evaluate_current(agent, eval_envs[t], t, args.confirm_episodes,
                                         args.confirm_seed + 1000 * len(confirmations))
                check.update({"task_steps": steps, "periodic_success": current})
                confirmations.append(check)
                append_json(os.path.join(root, "confirmations.jsonl"), check)
                print("MASKDECAY_CONFIRM " + json.dumps(check), flush=True)
                if check["success"] >= args.early_stop_success:
                    if stop_file is None or claim_stop_file(stop_file, root):
                        stop_reason = "confirmed"
                        break
                    # another candidate claimed first: the check at the top of the loop stops this one

    # Final paired evaluation on every seen task; the selection score is the current task's mean return.
    final = evaluate_paired(agent, eval_envs, task_names, t, args.final_eval_episodes, args.final_eval_seed)
    score = final[t]["return"]
    # Task end: store anchor states of this task (kl mode) from its own replay; no-op otherwise.
    finish_rng = np.random.RandomState(args.seed + 9973)
    finish_info = agent.finish_task(replay.sample_observations(min(args.anchor_states_per_task, replay.size), finish_rng))
    with open(os.path.join(root, "final_state.pkl"), "wb") as f:
        pickle.dump(jax.device_get(agent.state_dict()), f, protocol=pickle.HIGHEST_PROTOCOL)
    result = {"status": "ok" if math.isfinite(score) else "nonfinite", "task_index": t, "task_name": task_names[t],
              "score_mean_return": score, "success_current": final[t]["success"], "final_eval": final,
              "env_steps": steps, "actor_updates": updates, "episodes": episodes, "seed": args.seed,
              "lam": args.lam, "anchor_coef": args.anchor_coef, "finish_info": finish_info,
              "stop_reason": stop_reason, "confirmed": stop_reason == "confirmed", "confirmations": confirmations,
              "seconds": time.monotonic() - t0}
    atomic_json(os.path.join(root, "result.json"), result)
    print("MASKDECAY_CANDIDATE_DONE " + json.dumps({k: result[k] for k in ("task_index", "score_mean_return",
          "success_current", "env_steps", "stop_reason", "seconds")}), flush=True)
    for env in eval_envs + [train_env]:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # record the failure for the orchestrator, then fail loudly
        traceback.print_exc()
        try:
            run_dir = sys.argv[sys.argv.index("--run-dir") + 1]
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "result.json"), "w") as f:
                json.dump({"status": "failed", "error": repr(error)}, f)
        except Exception:
            pass
        sys.exit(1)
