#!/usr/bin/env python3
"""Checks for neuron_release.py and maskdecay.priority_population on a real scheme C + KL checkpoint
(start of task 3: three old tasks, 4x256)."""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import maskdecay as md  # noqa: E402
from neuron_release import Releaser, build_agent, kl_gauss  # noqa: E402

# A finished (or partly finished) scheme C + KL run directory that contains start_task03.pkl and winner_mask_task0{0,1,2}.npz.
EXP = os.environ.get("CKL_TEST_EXP", "/home/jerry/nevo_runs/new_runs/maskdecay_random_kl30_lam0.01_w256_cw10")
if not os.path.exists(f"{EXP}/start_task03.pkl"):
    raise SystemExit(f"test_release.py needs a run with start_task03.pkl; set CKL_TEST_EXP (now {EXP})")
T = 3
state = pickle.load(open(f"{EXP}/start_task{T:02d}.pkl", "rb"))
OBS, ACT = np.asarray(state["anchors"]["obs"]).shape[1], np.asarray(state["anchors"]["mu"]).shape[1]
hist = [md.load_neuron_mask(f"{EXP}/winner_mask_task{j:02d}.npz") for j in range(T)]
results = {}


def check(name, cond):
    results[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def counts_from(r):
    tree = md.to_plain_tree(jax.device_get(r.params))
    specs = md.kernel_specs_for(4, include_biases=True)
    c = md.init_counts(tree)
    for j in range(T):
        c = md.commit_counts(c, md.build_weight_mask(tree, tuple(r.masks[j][l].astype(np.uint8) for l in range(r.L)), specs))
    return md.flatten_tree(c)


r0 = Releaser(state, OBS, ACT)
check("mask table equals the ledger's winner masks", all(np.array_equal(np.stack(hist[j]).astype(np.float32), r0.masks[j]) for j in range(T)))
check("activation forward reproduces the learner's mean (< 1e-4)", max(r0.check_forward(j) for j in range(T)) < 1e-4)

# 1. nothing allowed -> masks unchanged and recomputed counts equal the run's counts file exactly
r1 = Releaser(state, OBS, ACT)
res1 = r1.release(T, kl_step=-1.0, kl_cum=-1.0, frac=0.05, max_tries=16)
orig = md.flatten_tree(md.load_tree_npz(f"{EXP}/counts_after_task{T - 1:02d}.npz"))
mine = counts_from(r1)
check("no release: masks unchanged", np.array_equal(r1.masks[:T], r0.masks[:T]) and not res1["released_pairs"])
check("no release: recomputed counts == counts_after_task02.npz", set(orig) == set(mine) and all(np.array_equal(orig[k], mine[k]) for k in orig))

# 2. everything allowed -> exactly floor(0.05*64)=3 per task per layer, never empty, counts follow the masks
r2 = Releaser(state, OBS, ACT)
res2 = r2.release(T, kl_step=1e9, kl_cum=1e9, frac=0.05, max_tries=16)
per = {}
for j, l, i in res2["released_pairs"]:
    per[(j, l)] = per.get((j, l), 0) + 1
check("unlimited thresholds: exactly 3 released per (task, layer)", all(per.get((j, l), 0) == 3 for j in range(T) for l in range(4)))
check("unlimited thresholds: masks lost exactly those neurons", all(
    r0.masks[j, l, i] == 1 and r2.masks[j, l, i] == 0 for j, l, i in res2["released_pairs"]) and
    int((r0.masks[:T] - r2.masks[:T]).sum()) == len(res2["released_pairs"]))
c2 = counts_from(r2)
tree2 = md.to_plain_tree(jax.device_get(r2.params))
k1 = np.zeros_like(c2["backbone_1/kernel"])
for j in range(T):
    k1 += np.outer(r2.masks[j, 0], r2.masks[j, 1]).astype(np.int64)
check("released counts: hidden kernel count = sum over tasks of outer(mask_l-1, mask_l)", np.array_equal(c2["backbone_1/kernel"], k1))
check("released counts: never larger than before", all(np.all(c2[k] <= orig[k]) for k in orig))

# 3. default thresholds: every accepted deletion within both limits, and final cumulative KL within the limit
r3 = Releaser(state, OBS, ACT)
res3 = r3.release(T, kl_step=0.01, kl_cum=0.03, frac=0.05, max_tries=16)
acc_rows = [c for e in res3["log"] for ly in e.get("layers", []) for c in ly["checks"] if c["accepted"]]
rej_rows = [c for e in res3["log"] for ly in e.get("layers", []) for c in ly["checks"] if not c["accepted"]]
check("default: accepted checks satisfy step<=0.01 and cum<=0.03", all(c["step_kl"] <= 0.01 and c["cum_kl"] <= 0.03 for c in acc_rows))
check("default: rejected checks violate a limit", all(c["step_kl"] > 0.01 or c["cum_kl"] > 0.03 for c in rej_rows))
print("  default thresholds: released", len(res3["released_pairs"]), "pairs; recycled", len(res3["recycled"]),
      "; per task cum KL", [(e["task"], round(e["cum_kl_start"], 4), round(e.get("cum_kl_end") or -1, 4)) for e in res3["log"]])
print("  step KL of all checks: median", float(np.median([c["step_kl"] for c in acc_rows + rej_rows])) if acc_rows + rej_rows else None)

# 4. recycling never changes an old task's policy (recycled neurons are referenced by no old task)
r4a, r4b = Releaser(state, OBS, ACT), Releaser(state, OBS, ACT)
ra = r4a.release(T, kl_step=1e9, kl_cum=1e9, frac=0.05, max_tries=16, do_recycle=False)
rb = r4b.release(T, kl_step=1e9, kl_cum=1e9, frac=0.05, max_tries=16, do_recycle=True)
same = True
for j in range(T):
    x, _, _ = r4a.inputs(j)
    m1, s1 = r4a.dist(x, r4a.masks[j])
    m2, s2 = r4b.dist(x, r4b.masks[j])
    same &= bool(np.allclose(np.asarray(m1), np.asarray(m2), atol=1e-6) and np.allclose(np.asarray(s1), np.asarray(s2), atol=1e-6))
check("recycling leaves every old task's (mu, sigma) unchanged", same and np.array_equal(r4a.masks, r4b.masks))
if rb["recycled"]:
    ok_out, ok_in = True, True
    for l, i in rb["recycled"]:
        outs = ([r4b.params[f"backbone_{l + 1}"]["kernel"][i]] if l < 3 else
                [r4b.params["mean"]["kernel"][i], r4b.params["std"]["kernel"][i]])
        ok_out &= all(float(jnp.max(jnp.abs(o))) == 0.0 for o in outs)
        ok_in &= not np.allclose(np.asarray(r4b.params[f"backbone_{l}"]["kernel"][:, i]), np.asarray(r4a.params[f"backbone_{l}"]["kernel"][:, i]))
    check(f"all {len(rb['recycled'])} recycled neurons: outgoing rows zeroed (both heads at the last layer)", ok_out)
    check("all recycled neurons: incoming column re-initialised", ok_in)
else:
    check("recycled neuron: (none recycled with 3 old tasks; skipped)", True)

# 5. the new state loads into the learner and reproduces the releaser's policy
st5 = r3.new_state()
agent = build_agent(st5, OBS, ACT)
ok = True
for j in range(T):
    x, _, _ = r3.inputs(j)
    mu, _ = r3.dist(x[:128], r3.masks[j])
    a = np.asarray(agent.sample_actions(np.asarray(x[:128, :OBS]), task_id=j, temperature=0.0))
    ok &= bool(np.max(np.abs(a - np.tanh(np.asarray(mu)))) < 1e-4)
check("released state loads into the learner; deterministic actions match", ok)

# 6. priority population
pool = [list(range(0, 10)), list(range(100, 200)), [], list(range(256))]
pop, rec = md.priority_population((256,) * 4, hist, pool, np.random.default_rng(3), candidate_count=5)
check("priority: 64 per layer", all(int(a.sum()) == 64 for c in pop for a in c))
check("priority: small pool fully included (layer 0)", all(np.all(c[0][:10] == 1) for c in pop))
check("priority: large pool -> all 64 from the pool (layer 1)", all(int(c[1][100:200].sum()) == 64 for c in pop))
check("priority: empty pool -> plain random (layer 2)", all(int(c[2].sum()) == 64 for c in pop) and rec[0]["taken_from_pool"][2] == 0)
print("RELEASE_TESTS", {"passed": sum(results.values()), "total": len(results)})
