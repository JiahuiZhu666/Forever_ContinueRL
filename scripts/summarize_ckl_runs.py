"""Summarize scheme C + KL CW10 runs from their ledgers: success when each task was learned (own-end), the final row
after task 10, mean forgetting over the first 9 tasks, number of confirmed winners and env steps.
Success = the runs' own definition (success at ANY step of the episode; Nevo-CRL's evaluate_cl uses the LAST step).
usage: python3 summarize_ckl_runs.py name1=/path/to/run1 [name2=/path/to/run2 ...]"""
import json
import os
import sys

SHORT = ["hammer", "push-w", "faucet", "push-b", "stick", "handle", "push", "shelf", "window", "peg"]


def main(items):
    print(f"{'run':20s} " + " ".join(f"{s:>6s}" for s in SHORT) + "    avg  no-shelf  forget  confirmed  env-steps")
    for item in items:
        name, d = item.split("=", 1)
        p = os.path.join(d, "ledger.json")
        if not os.path.exists(p):
            print(f"{name:20s} no ledger")
            continue
        tasks = json.load(open(p))["tasks"]
        own = [t["winner_final_eval"][t["task_index"]]["success"] for t in tasks]
        final = [x["success"] for x in tasks[-1]["winner_final_eval"]]
        n = len(tasks)
        cfg = json.load(open(os.path.join(d, "config.json"))).get("args", {})
        fixed = cfg.get("candidates", 5) * cfg.get("steps", 0)   # fixed-budget runs predate env_steps_total
        steps = sum(t.get("env_steps_total", fixed) for t in tasks)
        conf = sum(bool(t.get("winner_confirmed")) for t in tasks)
        forget = sum(own[i] - final[i] for i in range(n - 1)) / max(n - 1, 1)
        tag = "" if n == 10 else f"  (only {n} tasks)"
        no_shelf = (sum(final) - final[7]) / (n - 1) if n > 7 else float("nan")
        print(f"{name:20s} " + " ".join(f"{x:6.2f}" for x in final) + "       " * (10 - n) +
              f"  {sum(final) / n:5.3f}  {no_shelf:8.3f}  {forget:+6.3f}  {conf:>9d}  {steps / 1e6:8.2f}M{tag}")
        print(f"{'  (when learned)':20s} " + " ".join(f"{x:6.2f}" for x in own) + "       " * (10 - n) + f"  {sum(own) / n:5.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
