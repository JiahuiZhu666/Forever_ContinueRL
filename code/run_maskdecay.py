#!/usr/bin/env python3
"""Task-level orchestrator for the fixed-capacity mask-decay method (document v1.1).

For every task: embed the task description, generate K candidate neuron masks with the Nevo rule,
train the K candidates in parallel processes from the same start state, pick the winner by paired
mean return on the current task, and commit exactly once: learner state, neuron-mask history, task
embedding, per-weight counts. Resumable at task granularity (ledger.json) and candidate granularity
(result.json).
"""
import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from continual_world import TASK_SEQS  # noqa: E402
import maskdecay as md  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--env-name", default="cw10")
    p.add_argument("--tasks", type=int, default=3)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--lam", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=110, help="base training seed; candidate seed = seed + 1000*t + k")
    p.add_argument("--mask-seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--start-training", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--eval-interval", type=int, default=20_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--final-eval-episodes", type=int, default=50)
    p.add_argument("--final-eval-seed", type=int, default=20_000)
    p.add_argument("--actor-width", type=int, default=1024)
    p.add_argument("--hidden-layers", type=int, default=4)
    p.add_argument("--critic-width", type=int, default=256)
    p.add_argument("--actor-variant", default="stable")
    p.add_argument("--init-log-temperature", type=float, default=1.0)
    p.add_argument("--sample-ratio", type=float, default=0.25)
    p.add_argument("--min-ratio", type=float, default=0.01)
    p.add_argument("--top-history", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--forward-masking", type=int, choices=(0, 1), default=1)
    p.add_argument("--include-biases", type=int, choices=(0, 1), default=1)
    p.add_argument("--no-new-neurons", action="store_true",
                   help="tasks > 0 only inherit the parent mask (new ratio 0); task 0 still samples sample_ratio")
    p.add_argument("--anchor-coef", type=float, default=0.0, help=">0: KL anchor on stored old-task states")
    p.add_argument("--anchor-states-per-task", type=int, default=4096)
    p.add_argument("--anchor-batch", type=int, default=256)
    p.add_argument("--mask-rule", choices=("nevo", "random"), default="nevo",
                   help="nevo: inherit the most similar history mask + new free neurons; random: EVERY task "
                        "draws independent random masks (sample_ratio per layer, no inheritance, no history "
                        "pool); overlaps with earlier winners are what the counts/decay act on")
    p.add_argument("--fixed-new-ratio", type=float, default=None,
                   help="tasks > 0 add round(ratio * width) (at least 1) never-used neurons per layer. nevo rule: "
                        "added to the inherited parent mask instead of the similarity-dependent ratio (parent "
                        "selection unchanged). random rule: added on top of each candidate's random draw, from "
                        "neurons no earlier winner used and not already drawn (separate RNG stream)")
    p.add_argument("--early-stop-success", type=float, default=None,
                   help="if set: a candidate stops once a periodic eval AND a confirmation eval reach this current-task "
                        "success; the first to confirm wins the task and stops the others (--steps is the cap). "
                        "If none confirms, all train to --steps and the winner is chosen by return as before")
    p.add_argument("--confirm-episodes", type=int, default=50)
    p.add_argument("--release", type=int, choices=(0, 1), default=0,
                   help="1: before each task > 0, release unimportant neurons from old tasks' masks (neuron_release.py)")
    p.add_argument("--release-kl-step", type=float, default=0.01)
    p.add_argument("--release-kl-cum", type=float, default=0.03)
    p.add_argument("--release-frac", type=float, default=0.05)
    p.add_argument("--release-max-tries", type=int, default=16)
    p.add_argument("--mask-priority", choices=("none", "released"), default="none",
                   help="random rule only: 'released' = candidates first take released / unreferenced neurons")
    p.add_argument("--actor-optimizer", choices=("adam", "sgd"), default="adam")
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args(argv)


def load_ledger(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"committed": 0, "history_mask_files": [], "history_task_indices": [], "counts_file": None,
            "start_state": None, "tasks": []}


def main(argv=None):
    args = parse_args(argv)
    exp = os.path.abspath(args.exp_dir)
    os.makedirs(exp, exist_ok=True)
    ledger_path = os.path.join(exp, "ledger.json")
    ledger = load_ledger(ledger_path)
    seq = TASK_SEQS[args.env_name][:args.tasks]
    names = [d["task"] for d in seq]
    hints = [d["hint"] for d in seq]
    widths = (args.actor_width,) * args.hidden_layers
    specs = md.kernel_specs_for(args.hidden_layers)
    md.dump_json(os.path.join(exp, "config.json"), {"args": vars(args), "tasks": names, "hints": hints})

    emb_path = os.path.join(exp, "task_embeddings.npy")
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else None
    if embeddings is None or embeddings.shape[0] < len(hints):  # e.g. resumed with more tasks
        embeddings = md.embed_task_hints(hints)
        np.save(emb_path, embeddings)
    print("task embeddings:", embeddings.shape, flush=True)

    for t in range(args.tasks):
        if t < ledger["committed"]:
            print(f"task {t} already committed", flush=True)
            continue
        task_start = time.time()
        tdir = os.path.join(exp, f"task{t:02d}")
        os.makedirs(tdir, exist_ok=True)
        release_pool = None
        if args.release and t > 0:
            # Release unimportant neurons from old tasks' masks (KL-checked) before generating this task's candidates.
            rdir = os.path.join(exp, "release")
            rjson = os.path.join(rdir, f"release_for{t:02d}.json")
            if not os.path.exists(rjson):
                rcmd = [args.python, os.path.join(HERE, "neuron_release.py"), "--state", ledger["start_state"],
                        "--task", str(t), "--history", ",".join(ledger["history_mask_files"]), "--out-dir", rdir,
                        "--kl-step", str(args.release_kl_step), "--kl-cum", str(args.release_kl_cum),
                        "--frac", str(args.release_frac), "--max-tries", str(args.release_max_tries),
                        "--hidden-layers", str(args.hidden_layers), "--include-biases", str(args.include_biases)]
                with open(os.path.join(exp, "release.log"), "a") as rlog:
                    rc = subprocess.run(rcmd, cwd=HERE, stdout=rlog, stderr=subprocess.STDOUT).returncode
                if rc != 0 or not os.path.exists(rjson):
                    print(f"MASKDECAY_TASK_FAILED task={t}: neuron release failed (rc={rc})", flush=True)
                    return 4
            rel = json.load(open(rjson))
            s = rel["summary"]
            release_pool = rel["pool"]
            if ledger.get("released_for", -1) != t:
                ledger["start_state"], ledger["history_mask_files"], ledger["counts_file"] = (
                    s["state_out"], s["mask_files"], s["counts_out"])
                ledger.setdefault("releases", []).append(s)
                ledger["released_for"] = t
                md.dump_json(ledger_path + ".tmp", ledger)
                os.replace(ledger_path + ".tmp", ledger_path)
            print("MASKDECAY_RELEASE " + json.dumps({"task": t, "released_pairs": s["released_pairs"],
                  "recycled": s["recycled_neurons"], "pool_sizes": s["pool_sizes"]}), flush=True)
        history_masks = [md.load_neuron_mask(p) for p in ledger["history_mask_files"]]
        history_emb = [embeddings[i] for i in ledger["history_task_indices"]]
        rng = np.random.default_rng(args.mask_seed * 1000 + t)
        no_new = bool(args.no_new_neurons and t > 0)
        if args.mask_rule == "random" and args.mask_priority == "released" and release_pool is not None:
            # Same seed stream; each candidate first takes neurons from the released / unreferenced pool.
            population, records = md.priority_population(
                widths, history_masks, release_pool, rng, candidate_count=args.candidates, sample_ratio=args.sample_ratio)
        elif args.mask_rule == "random":
            # Independent random masks for every task and candidate: floor(sample_ratio * N) per layer,
            # drawn uniformly from ALL positions. History enters only through the weight counts.
            # --fixed-new-ratio: tasks > 0 also add round(ratio * N) never-used neurons per layer (full speed).
            extra = None
            if args.fixed_new_ratio is not None:
                extra = [max(1, int(round(args.fixed_new_ratio * w))) for w in widths]
            population, records = md.random_population(
                widths, history_masks, rng, candidate_count=args.candidates, sample_ratio=args.sample_ratio,
                extra_new_counts=extra, extra_rng=np.random.default_rng(args.mask_seed * 1000 + t + 500_000))
        else:
            fixed = None
            if args.fixed_new_ratio is not None and not no_new:
                fixed = [max(1, int(round(args.fixed_new_ratio * w))) for w in widths]
            population, records = md.nevo_population(
                widths, history_masks, history_emb, embeddings[t], rng, candidate_count=args.candidates,
                sample_ratio=0.0 if no_new else args.sample_ratio, min_ratio=0.0 if no_new else args.min_ratio,
                semantic_top_k=args.top_history, temperature=args.temperature, fixed_new_counts=fixed)
        # Launch candidates (skip those already finished).
        procs = {}
        pending = []
        for k in range(args.candidates):
            cdir = os.path.join(tdir, f"cand{k}")
            os.makedirs(cdir, exist_ok=True)
            mask_path = os.path.join(cdir, "neuron_mask.npz")
            if not os.path.exists(mask_path):
                md.save_neuron_mask(population[k], mask_path)
                md.dump_json(os.path.join(cdir, "mask_record.json"), records[k])
            result_path = os.path.join(cdir, "result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    if json.load(f).get("status") in ("ok", "stopped_by_peer"):
                        continue
            cmd = [args.python, os.path.join(HERE, "train_maskdecay_candidate.py"),
                   "--run-dir", cdir, "--env-name", args.env_name, "--task-index", str(t),
                   "--start-state", ledger["start_state"] or "none", "--neuron-mask", mask_path,
                   "--history-counts", ledger["counts_file"] or "none", "--lam", str(args.lam),
                   "--seed", str(args.seed + 1000 * t + k), "--steps", str(args.steps),
                   "--start-training", str(args.start_training), "--buffer-size", str(args.buffer_size),
                   "--eval-interval", str(args.eval_interval), "--eval-episodes", str(args.eval_episodes),
                   "--final-eval-episodes", str(args.final_eval_episodes), "--final-eval-seed", str(args.final_eval_seed),
                   "--actor-width", str(args.actor_width), "--hidden-layers", str(args.hidden_layers),
                   "--critic-width", str(args.critic_width), "--actor-variant", args.actor_variant,
                   "--init-log-temperature", str(args.init_log_temperature),
                   "--forward-masking", str(args.forward_masking), "--include-biases", str(args.include_biases),
                   "--anchor-coef", str(args.anchor_coef), "--anchor-states-per-task", str(args.anchor_states_per_task),
                   "--anchor-batch", str(args.anchor_batch),
                   "--actor-optimizer", args.actor_optimizer, "--actor-lr", str(args.actor_lr), "--require-gpu"]
            if args.early_stop_success is not None:
                cmd += ["--early-stop-success", str(args.early_stop_success),
                        "--confirm-episodes", str(args.confirm_episodes),
                        "--stop-file", os.path.join(tdir, "STOP.json")]
            pending.append((k, cdir, cmd))
        print(f"[task {t} {names[t]}] launching {len(pending)} candidates (parallel {args.parallel})", flush=True)
        while pending or procs:
            while pending and len(procs) < args.parallel:
                k, cdir, cmd = pending.pop(0)
                log = open(os.path.join(cdir, "train.log"), "a")
                procs[k] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE), log)
                print(f"  started candidate {k} (pid {procs[k][0].pid})", flush=True)
            time.sleep(30)
            for k in list(procs):
                proc, log = procs[k]
                if proc.poll() is not None:
                    log.close()
                    print(f"  candidate {k} exited rc={proc.returncode}", flush=True)
                    del procs[k]
        # Collect and select.
        scores = []
        for k in range(args.candidates):
            result_path = os.path.join(tdir, f"cand{k}", "result.json")
            r = json.load(open(result_path)) if os.path.exists(result_path) else {"status": "missing"}
            valid = r.get("status") == "ok" and math.isfinite(float(r.get("score_mean_return", float("nan"))))
            scores.append((k, valid, r))
        winner = None
        confirmed = [k for k, valid, r in scores if valid and r.get("confirmed")]
        if confirmed:
            # Early stop: the candidate that created the stop file (first to confirm) wins.
            stop_path = os.path.join(tdir, "STOP.json")
            owner = json.load(open(stop_path)).get("owner") if os.path.exists(stop_path) else None
            owners = [k for k in confirmed if os.path.join(tdir, f"cand{k}") == owner]
            winner = owners[0] if owners else confirmed[0]
        else:
            for k, valid, r in scores:
                if valid and (winner is None or r["score_mean_return"] > scores[winner][2]["score_mean_return"]):
                    winner = k
        table = [{"candidate": k, "status": r.get("status"), "score_mean_return": r.get("score_mean_return"),
                  "success_current": r.get("success_current"), "seconds": r.get("seconds"),
                  "env_steps": r.get("env_steps"), "stop_reason": r.get("stop_reason"),
                  "confirmed": bool(r.get("confirmed")), "mask_record": records[k]} for k, valid, r in scores]
        env_steps_total = int(sum(int(r.get("env_steps") or 0) for k, valid, r in scores))
        if winner is None:
            md.dump_json(os.path.join(tdir, "selection.json"), {"winner": None, "candidates": table})
            print(f"MASKDECAY_TASK_FAILED task={t}: no valid candidate", flush=True)
            return 3
        wdir = os.path.join(tdir, f"cand{winner}")
        with open(os.path.join(wdir, "final_state.pkl"), "rb") as f:
            winner_state = pickle.load(f)
        actor_tree = md.to_plain_tree(winner_state["learner_state"]["actor_params"])
        counts = md.load_tree_npz(ledger["counts_file"]) if ledger["counts_file"] else md.init_counts(actor_tree)
        winner_mask = md.load_neuron_mask(os.path.join(wdir, "neuron_mask.npz"))
        specs = md.kernel_specs_for(args.hidden_layers, include_biases=bool(args.include_biases))
        weight_mask = md.build_weight_mask(actor_tree, winner_mask, specs)
        new_counts = md.commit_counts(counts, weight_mask)
        counts_path = os.path.join(exp, f"counts_after_task{t:02d}.npz")
        md.save_tree_npz(new_counts, counts_path)
        start_path = os.path.join(exp, f"start_task{t + 1:02d}.pkl")
        shutil.copyfile(os.path.join(wdir, "final_state.pkl"), start_path)
        mask_hist_path = os.path.join(exp, f"winner_mask_task{t:02d}.npz")
        md.save_neuron_mask(winner_mask, mask_hist_path)
        union = [np.zeros(w, dtype=np.uint8) for w in widths]
        for p in ledger["history_mask_files"] + [mask_hist_path]:
            for layer, a in enumerate(md.load_neuron_mask(p)):
                union[layer] |= a
        flat_counts = md.flatten_tree(new_counts)
        count_summary = {key: {str(int(c)): int(n) for c, n in zip(*np.unique(v, return_counts=True))}
                         for key, v in flat_counts.items() if key.endswith("/kernel")}
        final_eval = scores[winner][2]["final_eval"]
        entry = {"task_index": t, "task_name": names[t], "winner": winner, "candidates": table,
                 "winner_final_eval": final_eval, "winner_success_row": [r["success"] for r in final_eval],
                 "winner_return_row": [r["return"] for r in final_eval],
                 "neuron_union_density": [float(u.mean()) for u in union],
                 "winner_mask_density": [float(a.mean()) for a in winner_mask],
                 "weight_count_histograms": count_summary, "seconds": time.time() - task_start,
                 "winner_confirmed": bool(scores[winner][2].get("confirmed")),
                 "winner_env_steps": scores[winner][2].get("env_steps"), "env_steps_total": env_steps_total}
        md.dump_json(os.path.join(tdir, "selection.json"), entry)
        ledger["history_mask_files"].append(mask_hist_path)
        ledger["history_task_indices"].append(t)
        ledger["counts_file"] = counts_path
        ledger["start_state"] = start_path
        ledger["tasks"].append(entry)
        ledger["committed"] = t + 1
        tmp = ledger_path + ".tmp"
        md.dump_json(tmp, ledger)
        os.replace(tmp, ledger_path)
        # Success matrix across committed tasks.
        with open(os.path.join(exp, "task_success_matrix.csv"), "w") as f:
            f.write("trained_task_index," + ",".join(names) + ",avg_seen\n")
            for e in ledger["tasks"]:
                row = e["winner_success_row"] + [""] * (len(names) - len(e["winner_success_row"]))
                f.write(f"{e['task_index']}," + ",".join(str(v) for v in row) + f",{np.mean(e['winner_success_row']):.4f}\n")
        print("MASKDECAY_TASK_DONE " + json.dumps({"task": t, "winner": winner,
              "scores": [round(c["score_mean_return"], 1) if c["score_mean_return"] is not None else None for c in table],
              "success_row": entry["winner_success_row"], "union_density": entry["neuron_union_density"],
              "winner_confirmed": entry["winner_confirmed"], "winner_env_steps": entry["winner_env_steps"],
              "env_steps_total": env_steps_total}), flush=True)
    print("MASKDECAY_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
