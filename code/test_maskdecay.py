#!/usr/bin/env python3
"""Acceptance checks for the mask-decay implementation (method document section 15)."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.core import unfreeze  # noqa: E402
import maskdecay as md  # noqa: E402

spec = importlib.util.spec_from_file_location("cumulative_sac_learner", os.path.join(HERE, "jaxrl/agents/sac/cumulative_learner.py"))
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)

results = {}


def check(name, condition):
    results[name] = bool(condition)
    print(("PASS " if condition else "FAIL ") + name, flush=True)


# ---- population generation ----------------------------------------------------------------------
widths = (1024, 1024, 1024, 1024)
rng = np.random.default_rng(0)
e0 = np.ones(8); e1 = np.array([1, 1, 1, 1, 0, 0, 0, 0], float); e2 = -np.ones(8)
pop1, rec1 = md.nevo_population(widths, [], [], e0, rng, candidate_count=5)
check("task1: 5 candidates, each layer floor(0.25*N)=256", len(pop1) == 5 and all(int(a.sum()) == 256 for c in pop1 for a in c))
check("task1: candidates may overlap and differ", any(np.any(pop1[0][0] != pop1[1][0]) for _ in [0]))
hist = [pop1[0]]
pop2, rec2 = md.nevo_population(widths, hist, [e0], e1, rng, candidate_count=3)
sim = float(np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1)))
r_expected = 0.01 + max(0.0, 1 - sim) * 0.24
check("task2: inherits parent mask", all(np.all(c[l][hist[0][l] == 1] == 1) for c in pop2 for l in range(4)))
check("task2: new positions outside the union; count = floor(r*N)",
      all(int(c[l].sum()) - 256 == int(r_expected * 1024) for c in pop2 for l in range(4)) and
      abs(rec2[0]["new_ratio"] - r_expected) < 1e-12)
pop_neg, rec_neg = md.nevo_population(widths, hist, [e0], e2, rng, candidate_count=1)
check("negative similarity: ratio not clipped to 0.25", rec_neg[0]["new_ratio"] > 0.25 and rec_neg[0]["actual_new_counts"][0] == int(rec_neg[0]["new_ratio"] * 1024))
full = [tuple(np.ones(w, np.uint8) for w in widths)]
pop_full, rec_full = md.nevo_population(widths, full, [e0], e1, rng, candidate_count=1)
check("free pool exhausted: no error, zero new, inherits only", rec_full[0]["actual_new_counts"] == [0, 0, 0, 0] and all(int(a.sum()) == 1024 for a in pop_full[0]))
many_hist = [pop1[i % 5] for i in range(7)]
many_e = [np.random.default_rng(i).normal(size=8) for i in range(7)]
_, rec_many = md.nevo_population(widths, many_hist, many_e, e0, rng, candidate_count=20)
sims = np.array([np.dot(e, e0) / (np.linalg.norm(e) * np.linalg.norm(e0)) for e in many_e])
top5 = set(np.argsort(sims)[::-1][:5].tolist())
check("parents only from the top-5 most similar history tasks", all(r["parent_task_index_zero_based"] in top5 for r in rec_many))
hist_before = [a.copy() for a in hist[0]]
check("generation does not modify history", all(np.array_equal(a, b) for a, b in zip(hist_before, hist[0])))

# ---- fixed new-neuron count (similarity picks the parent; always 1% new) ------------------------
w256 = (256, 256, 256, 256)
fixed = [max(1, int(round(0.01 * w))) for w in w256]
pf1, rf1 = md.nevo_population(w256, [], [], e0, rng, candidate_count=5, fixed_new_counts=fixed)
check("fixed: first task still samples 25% (64/layer)", all(int(a.sum()) == 64 for c in pf1 for a in c) and rf1[0]["new_count_rule"] == "nevo_ratio")
fh = [pf1[0], pf1[1], pf1[2]]
fe = [e0, e1, e2]
union = [np.zeros(256, bool) for _ in w256]
for m in fh:
    for l, a in enumerate(m):
        union[l] |= a.astype(bool)
pf2, rf2 = md.nevo_population(w256, fh, fe, e1, rng, candidate_count=20, fixed_new_counts=fixed)
check("fixed: 1% of 256 rounds to 3 per layer", fixed == [3, 3, 3, 3])
check("fixed: child = parent mask + exactly 3 never-used neurons per layer",
      all(np.all(c[l][fh[r["parent_task_index_zero_based"]][l] == 1] == 1)
          and int(c[l].sum()) - int(fh[r["parent_task_index_zero_based"]][l].sum()) == 3
          and not np.any(union[l][(c[l] == 1) & (fh[r["parent_task_index_zero_based"]][l] == 0)])
          for c, r in zip(pf2, rf2) for l in range(4)))
check("fixed: count independent of similarity; rule recorded",
      all(r["requested_new_counts"] == [3, 3, 3, 3] and r["new_ratio"] is None and r["new_count_rule"] == "fixed" for r in rf2))
fe_sims = np.array([np.dot(e, e1) / (np.linalg.norm(e) * np.linalg.norm(e1)) for e in fe])
fe_logits = np.exp(fe_sims / 0.5); fe_p = fe_logits / fe_logits.sum()
parents = [r["parent_task_index_zero_based"] for r in rf2]
check("fixed: parent still drawn by softmax(sim/0.5), not argmax",
      len(set(parents)) > 1 and all(abs(r["similarity"] - fe_sims[r["parent_task_index_zero_based"]]) < 1e-12 for r in rf2))
try:
    md.nevo_population(w256, fh, fe, e1, rng, fixed_new_counts=[3, 3, 3])
    check("fixed: wrong number of layers rejected", False)
except ValueError:
    check("fixed: wrong number of layers rejected", True)

# ---- weight mapping on a real FullPolicy tree ---------------------------------------------------
def make_learner(mode="external", reset_all=True, seed=5, **kw):
    return MODULE.CumulativeSACLearner(
        seed=seed, observations=np.zeros((1, 12), np.float32), actions=np.zeros((1, 4), np.float32), num_tasks=3,
        actor_hidden_dims=(8, 6), critic_hidden_dims=(16, 16), decay_factor=1.0, mask_fraction=0.25,
        refresh_interval=1000, target_entropy=-2.0, init_temperature=1.0, actor_variant="stable",
        reset_critic=False, reset_actor_optimizer=True, temperature_mode="shared_keep",
        importance_mode=mode, reset_all_optimizers=reset_all, **kw)


agent = make_learner()
tree = md.to_plain_tree(jax.device_get(unfreeze(agent._state.actor_params)))
specs = md.kernel_specs_for(2)
z = (np.array([1, 1, 0, 0, 0, 0, 0, 0], np.uint8), np.array([1, 0, 1, 0, 0, 0], np.uint8))
M = md.build_weight_mask(tree, z, specs)
k0, k1 = M["backbone_0"]["kernel"], M["backbone_1"]["kernel"]
check("input kernel: whole columns of selected neurons (all 15 input rows)", k0.shape == (15, 8) and np.array_equal(k0.sum(0), np.array([15, 15, 0, 0, 0, 0, 0, 0])))
check("hidden kernel: outer product z0 z1^T", np.array_equal(k1, np.outer(z[0], z[1])))
check("mean/std heads: whole rows of selected last-layer neurons", np.array_equal(M["mean"]["kernel"], np.outer(z[1], np.ones(4, np.uint8))) and np.array_equal(M["std"]["kernel"], np.outer(z[1], np.ones(4, np.uint8))))
check("biases get zero mask", np.all(M["backbone_0"]["bias"] == 0) and np.all(M["std"]["bias"] == 0))

# ---- counts and multipliers ---------------------------------------------------------------------
h0 = md.init_counts(tree)
s1 = md.candidate_scales(h0, M, 0.8)
check("task1: all multipliers are 1", all(np.all(np.asarray(v) == 1.0) for v in jax.tree_util.tree_leaves(s1)))
h1 = md.commit_counts(h0, M)
s2 = md.candidate_scales(h1, M, 0.8)
check("h=1, M=1 -> lambda", np.allclose(s2["backbone_1"]["kernel"][k1 == 1], 0.8))
zB = (np.array([0, 0, 1, 0, 0, 0, 0, 0], np.uint8), np.array([0, 1, 0, 0, 0, 0], np.uint8))
MB = md.build_weight_mask(tree, zB, specs)
h2 = md.commit_counts(h1, M)  # same mask selected twice -> h=2
s3 = md.candidate_scales(h2, MB, 0.8)
check("h=2, M=0 -> still lambda", np.allclose(s3["backbone_1"]["kernel"][(k1 == 1) & (MB["backbone_1"]["kernel"] == 0)], 0.8))
check("h=0, M=1 -> 1 (first selection free)", np.allclose(s3["backbone_1"]["kernel"][(k1 == 0) & (MB["backbone_1"]["kernel"] == 1)], 1.0))
hAB = md.commit_counts(h1, MB)  # A then B
union_outer = np.outer(z[0] | zB[0], z[1] | zB[1])
check("counts are the SUM of per-task masks, not the union outer product",
      np.array_equal(hAB["backbone_1"]["kernel"], k1.astype(np.int64) + MB["backbone_1"]["kernel"]) and
      np.any((union_outer == 1) & (hAB["backbone_1"]["kernel"] == 0)))
check("commit does not mutate the input counts", np.all(h1["backbone_1"]["kernel"] == k1))

# ---- scheme C random masks + fixed never-used extra neurons ---------------------------------------
def legacy_random(widths_, seed, K):
    r = np.random.default_rng(seed); out = []
    for _ in range(K):
        cand = []
        for w in widths_:
            zz = np.zeros(w, np.uint8); zz[r.choice(w, size=int(0.25 * w), replace=False)] = 1; cand.append(zz)
        out.append(tuple(cand))
    return out
leg0 = legacy_random(w256, 7000, 5)
rp0, rr0 = md.random_population(w256, [], np.random.default_rng(7000), candidate_count=5,
                                extra_new_counts=[3] * 4, extra_rng=np.random.default_rng(1))
check("random rule: draws identical to the original inline implementation; no extras without history",
      all(np.array_equal(a, b) for c, d in zip(rp0, leg0) for a, b in zip(c, d)) and rr0[0]["rule"] == "random")
hist_r = [leg0[3], leg0[1]]
used_r = [hist_r[0][l].astype(bool) | hist_r[1][l].astype(bool) for l in range(4)]
leg1 = legacy_random(w256, 7001, 5)
rp1, rr1 = md.random_population(w256, hist_r, np.random.default_rng(7001), candidate_count=5,
                                extra_new_counts=[3] * 4, extra_rng=np.random.default_rng(2))
check("random+extra: the random 64 are unchanged; exactly 3 extra per layer, never used and not already drawn",
      all(int(c[l].sum()) == 67 and np.all(c[l][d[l] == 1] == 1)
          and not np.any(used_r[l][(c[l] == 1) & (d[l] == 0)]) and int(((c[l] == 1) & (d[l] == 0)).sum()) == 3
          for c, d in zip(rp1, leg1) for l in range(4))
      and all(r["actual_extra_new_counts"] == [3] * 4 and r["rule"] == "random+extra_new" for r in rr1))
nearly_full = [tuple(np.concatenate([np.ones(254, np.uint8), np.zeros(2, np.uint8)]) for _ in w256)]
rp2, rr2 = md.random_population(w256, nearly_full, np.random.default_rng(3), candidate_count=3,
                                extra_new_counts=[3] * 4, extra_rng=np.random.default_rng(4))
check("random+extra: small never-used pool -> takes what is left, no error",
      all(n <= 2 for r in rr2 for n in r["actual_extra_new_counts"]) and all(r["requested_extra_new_counts"] == [3] * 4 for r in rr2))
specs_b = md.kernel_specs_for(2, include_biases=True)
zH = (np.array([1, 1, 1, 1, 0, 0, 0, 0], np.uint8), np.array([1, 1, 1, 0, 0, 0], np.uint8))
hH = md.commit_counts(md.init_counts(tree), md.build_weight_mask(tree, zH, specs_b))
rp3, rr3 = md.random_population((8, 6), [zH], np.random.default_rng(5), candidate_count=1, sample_ratio=0.5,
                                extra_new_counts=[1, 1], extra_rng=np.random.default_rng(6))
c3 = rp3[0]
s_c3 = md.candidate_scales(hH, md.build_weight_mask(tree, c3, specs_b), 0.01)
ex0 = [i for i in np.flatnonzero(c3[0]) if zH[0][i] == 0]
ex1 = [i for i in np.flatnonzero(c3[1]) if zH[1][i] == 0]
check("random+extra: every weight and bias touching a never-used neuron has multiplier 1 (full speed)",
      all(np.all(s_c3["backbone_0"]["kernel"][:, i] == 1.0) and s_c3["backbone_0"]["bias"][i] == 1.0
          and np.all(s_c3["backbone_1"]["kernel"][i, :] == 1.0) for i in ex0)
      and all(np.all(s_c3["backbone_1"]["kernel"][:, j] == 1.0) and s_c3["backbone_1"]["bias"][j] == 1.0
              and np.all(s_c3["mean"]["kernel"][j, :] == 1.0) and np.all(s_c3["std"]["kernel"][j, :] == 1.0) for j in ex1)
      and len(ex0) >= 1 and len(ex1) >= 1)
check("random+extra: weights inside the old winner's block that the candidate reuses get lambda",
      all(s_c3["backbone_1"]["kernel"][i, j] == np.float32(0.01)
          for i in np.flatnonzero(c3[0] & zH[0]) for j in np.flatnonzero(c3[1] & zH[1])))

# ---- learner external mode ----------------------------------------------------------------------
batch = {"observations": np.random.RandomState(3).normal(size=(8, 12)).astype(np.float32),
         "actions": np.tanh(np.random.RandomState(4).normal(size=(8, 4))).astype(np.float32),
         "rewards": np.random.RandomState(5).normal(size=8).astype(np.float32),
         "masks": np.ones(8, np.float32),
         "next_observations": np.random.RandomState(6).normal(size=(8, 12)).astype(np.float32)}
a1, a2 = make_learner(seed=9), make_learner(seed=9)
a1.begin_task(0); a2.begin_task(0)
ones = md.candidate_scales(h0, M, 1.0)
half = jax.tree_util.tree_map(lambda s: s * 0.5, s1)  # every coordinate 0.5
p0 = jax.device_get(unfreeze(a1._state.actor_params))
a1.set_scales(ones); a2.set_scales(half)
a1.update(batch); a2.update(batch)
d1 = jax.tree_util.tree_map(lambda p, q: np.asarray(p) - np.asarray(q), jax.device_get(unfreeze(a1._state.actor_params)), p0)
d2 = jax.tree_util.tree_map(lambda p, q: np.asarray(p) - np.asarray(q), jax.device_get(unfreeze(a2._state.actor_params)), p0)
ratio_ok = all(np.allclose(np.asarray(b), 0.5 * np.asarray(a), atol=1e-7) for a, b in zip(jax.tree_util.tree_leaves(d1), jax.tree_util.tree_leaves(d2)))
check("external scales multiply the post-Adam increment exactly", ratio_ok)
try:
    a1.set_scales(jax.tree_util.tree_map(lambda s: s * 2.0, s1)); check("set_scales rejects >1", False)
except ValueError:
    check("set_scales rejects >1", True)
# boundary: parameters inherited, all optimizers reset, temperature kept
a1.update(batch)
params_before = jax.device_get(unfreeze(a1._state.critic_params))
temp_before = float(a1._state.log_temperature)
a1.begin_task(1)
critic_same = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(
    jax.tree_util.tree_leaves(params_before), jax.tree_util.tree_leaves(jax.device_get(unfreeze(a1._state.critic_params)))))
adam_count = int(jax.tree_util.tree_leaves(a1._state.critic_opt_state)[0]) if jax.tree_util.tree_leaves(a1._state.critic_opt_state) else 0
check("begin_task keeps critic params and temperature", critic_same and float(a1._state.log_temperature) == temp_before)
check("begin_task resets every optimizer (critic Adam count back to 0)", adam_count == 0)
# state round trip in external mode with arbitrary scales
a1.set_scales(s2)
state = jax.device_get(a1.state_dict())
a3 = make_learner(seed=1)
a3.load_state_dict(state)
check("state_dict/load_state_dict round trip keeps external scales",
      all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree_util.tree_leaves(a3._state.scales), jax.tree_util.tree_leaves(a1._state.scales))))
check("finish_task in external mode is a no-op returning {}", a1.finish_task(np.zeros((1, 12), np.float32)) == {})

# ---- actor optimizer: SGD (no momentum) instead of Adam ------------------------------------------
def first_delta(learner, scales):
    learner.begin_task(0); learner.set_scales(scales)
    before = jax.device_get(unfreeze(learner._state.actor_params))
    learner.update(batch)
    after = jax.device_get(unfreeze(learner._state.actor_params))
    return np.concatenate([(np.asarray(p) - np.asarray(q)).ravel() for p, q in
                           zip(jax.tree_util.tree_leaves(after), jax.tree_util.tree_leaves(before))])
check("default actor optimizer is Adam at learning_rate",
      a1.config["actor_optimizer"] == "adam" and a1.config["actor_learning_rate"] == 3e-4)
dA = first_delta(make_learner(seed=9), ones)
dS1 = first_delta(make_learner(seed=9, actor_optimizer="sgd", actor_learning_rate=1e-2), ones)
dS2 = first_delta(make_learner(seed=9, actor_optimizer="sgd", actor_learning_rate=2e-2), ones)
dSh = first_delta(make_learner(seed=9, actor_optimizer="sgd", actor_learning_rate=1e-2), half)
nzA, nzS = np.abs(dA) > 0, np.abs(dS1) > 0
check("Adam first step: |delta| ~ lr on every moving coordinate (sign-like)",
      0.9 < np.median(np.abs(dA[nzA]) / 3e-4) <= 1.0 + 1e-3)
# float32: delta = p_after - p_before carries rounding error ~ |p| * 6e-8, so compare with an absolute
# tolerance of 2e-7 (deltas themselves reach ~1e-3).
print("  SGD max |d(2lr) - 2 d(lr)| =", float(np.max(np.abs(dS2 - 2 * dS1))), " max |d(lr)| =", float(np.max(np.abs(dS1))),
      " max |d(s=0.5) - 0.5 d(s=1)| =", float(np.max(np.abs(dSh - 0.5 * dS1))))
check("SGD: delta exactly linear in lr (no gradient normalisation)", np.allclose(dS2, 2 * dS1, rtol=1e-4, atol=2e-7))
check("SGD: delta is lr * clipped gradient, not sign-like (magnitudes spread, global norm <= lr * clip)",
      np.percentile(np.abs(dS1[nzS]), 90) / max(np.percentile(np.abs(dS1[nzS]), 10), 1e-12) > 10
      and np.linalg.norm(dS1) <= 1e-2 * 1.0 + 1e-6)
check("SGD: post-optimizer scales still multiply the increment exactly", np.allclose(dSh, 0.5 * dS1, rtol=1e-4, atol=2e-7))
sg = make_learner(seed=9, actor_optimizer="sgd", actor_learning_rate=1e-2)
sg.begin_task(0); sg.set_scales(ones); sg.update(batch); sg.update(batch)
sg_state = jax.device_get(sg.state_dict())
sg2 = make_learner(seed=1, actor_optimizer="sgd", actor_learning_rate=1e-2)
sg2.load_state_dict(sg_state)
check("SGD learner state round trip", all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(
    jax.tree_util.tree_leaves(sg2._state.actor_params), jax.tree_util.tree_leaves(sg._state.actor_params))))
try:
    make_learner(seed=1).load_state_dict(sg_state); check("SGD checkpoint refused by an Adam learner", False)
except ValueError:
    check("SGD checkpoint refused by an Adam learner", True)
old_state = jax.device_get(a1.state_dict())
old_state["config"] = {k: v for k, v in old_state["config"].items() if k not in ("actor_optimizer", "actor_learning_rate")}
a4 = make_learner(seed=1)
a4.load_state_dict(old_state)
check("checkpoints written before the actor_optimizer option load as Adam", a4.config["actor_optimizer"] == "adam")
# npz round trips
tmp = "/tmp/maskdecay_test"
os.makedirs(tmp, exist_ok=True)
md.save_tree_npz(h1, os.path.join(tmp, "c.npz")); md.save_neuron_mask(z, os.path.join(tmp, "m.npz"))
check("npz round trip (counts, masks)", np.array_equal(md.load_tree_npz(os.path.join(tmp, "c.npz"))["backbone_1"]["kernel"], h1["backbone_1"]["kernel"])
      and all(np.array_equal(a, b) for a, b in zip(md.load_neuron_mask(os.path.join(tmp, "m.npz")), z)))

# ---- forward masking (Nevo structure) --------------------------------------------------------------
def make_masked(seed=21, mode="external", forward_masking=True):
    return MODULE.CumulativeSACLearner(
        seed=seed, observations=np.zeros((1, 12), np.float32), actions=np.zeros((1, 4), np.float32), num_tasks=3,
        actor_hidden_dims=(8, 8), critic_hidden_dims=(16, 16), decay_factor=1.0, mask_fraction=0.25,
        refresh_interval=1000, target_entropy=-2.0, init_temperature=1.0, actor_variant="stable",
        reset_critic=False, reset_actor_optimizer=True, temperature_mode="shared_keep",
        importance_mode=mode, reset_all_optimizers=True, forward_masking=forward_masking)


m1 = make_masked()
m1.begin_task(0)
obs = np.random.RandomState(11).normal(size=(5, 12)).astype(np.float32)
dense = make_masked(seed=21, forward_masking=False)  # same seed and widths -> same initial params, dense actor
dense.begin_task(0)
dense_params_equal = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
    jax.tree_util.tree_leaves(dense._state.actor_params), jax.tree_util.tree_leaves(m1._state.actor_params)))
m1.set_task_mask(0, np.ones((2, 8), np.float32))
a_masked = m1.sample_actions(obs, task_id=0, temperature=0.0)
a_dense = dense.sample_actions(obs, task_id=0, temperature=0.0)
check("forward masking with all-ones mask equals the dense actor", dense_params_equal and np.allclose(a_masked, a_dense, atol=1e-6))
zA = np.array([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]], np.float32)
m1.set_task_mask(0, zA)
base = m1.sample_actions(obs, task_id=0, temperature=0.0)
# perturb every weight OUTSIDE task 0's sub-block (and biases of inactive neurons): output must not change
params = unfreeze(jax.device_get(m1._state.actor_params))
specs8 = md.kernel_specs_for(2, include_biases=True)
MA = md.build_weight_mask(md.to_plain_tree(params), tuple(zA.astype(np.uint8)), specs8)
rs = np.random.RandomState(5)
def perturb(path, p, m):
    p = np.array(p); m = np.asarray(m)
    if path[-1] in ("kernel", "bias") and path[0] != "std":  # keep std bias (shared) untouched
        p = p + rs.normal(size=p.shape).astype(p.dtype) * (1 - m)
    return p
pert = md.map_params(perturb, md.to_plain_tree(params), MA)
from flax.core import freeze as _freeze
m1._state = m1._state.replace(actor_params=_freeze(jax.tree_util.tree_map(jnp.asarray, pert)))
after = m1.sample_actions(obs, task_id=0, temperature=0.0)
check("task output depends only on its own sub-block (outside weights/biases perturbed, output unchanged)", np.allclose(base, after, atol=1e-6))
# perturbing an in-block weight does change the output
pert2 = jax.tree_util.tree_map(np.array, pert)
pert2["backbone_1"]["kernel"][0, 0] += 1.0  # neuron 0 (layer 0) -> neuron 0 (layer 1), both active
m1._state = m1._state.replace(actor_params=_freeze(jax.tree_util.tree_map(jnp.asarray, pert2)))
check("in-block weight does change the output", not np.allclose(base, m1.sample_actions(obs, task_id=0, temperature=0.0), atol=1e-6))
# gradients vanish outside the sub-block
m2 = make_masked(seed=3)
m2.begin_task(0); m2.set_task_mask(0, zA); m2.set_scales(md.candidate_scales(md.init_counts(md.to_plain_tree(params)), MA, 1.0))
p_before = jax.device_get(unfreeze(m2._state.actor_params))
m2.update(batch)
p_after = jax.device_get(unfreeze(m2._state.actor_params))
moved_outside = False
for key in ("backbone_0", "backbone_1", "mean", "std"):
    for leaf in ("kernel", "bias"):
        if leaf not in p_before[key] or (key == "std" and leaf == "bias"):
            continue
        d = np.abs(np.asarray(p_after[key][leaf]) - np.asarray(p_before[key][leaf]))
        if np.any(d[np.asarray(MA[key][leaf]) == 0] > 0):
            moved_outside = True
check("with forward masking, weights outside the sub-block receive no update", not moved_outside)
# bias mapping
check("include_biases: hidden biases get their layer's neuron mask, std bias stays 0",
      np.array_equal(MA["backbone_0"]["bias"], zA[0].astype(np.uint8)) and np.array_equal(MA["backbone_1"]["bias"], zA[1].astype(np.uint8)) and np.all(MA["std"]["bias"] == 0))
# masks survive state_dict round trip and are used per task
m1.set_task_mask(1, np.array([[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]], np.float32))
sd = jax.device_get(m1.state_dict())
m3 = make_masked(seed=99)
m3.load_state_dict(sd)
check("neuron mask table saved/restored with the learner state", np.array_equal(np.asarray(m3.neuron_mask_table), np.asarray(m1.neuron_mask_table)))
check("sample_actions uses the requested task's mask", not np.allclose(m3.sample_actions(obs, task_id=0, temperature=0.0), m3.sample_actions(obs, task_id=1, temperature=0.0)))
try:
    make_masked(mode="mas"); check("forward_masking rejects importance modes mas/fisher", False)
except ValueError:
    check("forward_masking rejects importance modes mas/fisher", True)

# ---- KL anchor on top of forward masking (scheme C + KL) ---------------------------------------
def make_masked_kl(seed=31, coef=30.0):
    return MODULE.CumulativeSACLearner(
        seed=seed, observations=np.zeros((1, 12), np.float32), actions=np.zeros((1, 4), np.float32), num_tasks=3,
        actor_hidden_dims=(8, 8), critic_hidden_dims=(16, 16), decay_factor=1.0, mask_fraction=0.25,
        refresh_interval=1000, target_entropy=-2.0, init_temperature=1.0, actor_variant="stable",
        reset_critic=False, reset_actor_optimizer=True, temperature_mode="shared_keep",
        importance_mode="external", reset_all_optimizers=True, forward_masking=True,
        anchor_mode="kl", anchor_coef=coef, anchor_states_per_task=64, anchor_batch=16)


mk = make_masked_kl()
mk.begin_task(0); mk.set_task_mask(0, zA)
tree_k = md.to_plain_tree(jax.device_get(unfreeze(mk._state.actor_params)))
MAk = md.build_weight_mask(tree_k, tuple(zA.astype(np.uint8)), specs8)
mk.set_scales(md.candidate_scales(md.init_counts(tree_k), MAk, 1.0))
obs_pool = np.random.RandomState(12).normal(size=(100, 12)).astype(np.float32)
info0 = mk.finish_task(obs_pool)
check("finish_task stores anchors under the task's mask", info0.get("anchor_states_total") == 64 and mk.anchor_cond.tolist() == [0] * 64)
old_actions = mk.sample_actions(mk.anchor_obs, task_id=0, temperature=0.0)
zB2 = np.array([[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]], np.float32)
mk.begin_task(1); mk.set_task_mask(1, zB2)
MBk = md.build_weight_mask(tree_k, tuple(zB2.astype(np.uint8)), specs8)
hk = md.commit_counts(md.init_counts(tree_k), MAk)
mk.set_scales(md.candidate_scales(hk, MBk, 0.01))
pb = jax.device_get(unfreeze(mk._state.actor_params))
first = mk.update(batch)
first_anchor = float(jax.device_get(first["anchor_loss"]))
for _ in range(5):
    last = mk.update(batch)
pa = jax.device_get(unfreeze(mk._state.actor_params))
check("KL anchor with per-sample masks: anchor loss is 0 before any change and finite after",
      first_anchor < 1e-5 and np.isfinite(float(jax.device_get(last["anchor_loss"]))))
moved = False
for key in ("backbone_0", "backbone_1", "mean", "std"):
    for leaf in ("kernel", "bias"):
        if leaf not in pb[key] or (key == "std" and leaf == "bias"):
            continue
        inside = (np.asarray(MAk[key][leaf]) == 1) | (np.asarray(MBk[key][leaf]) == 1)
        d = np.abs(np.asarray(pa[key][leaf]) - np.asarray(pb[key][leaf]))
        if np.any(d[~inside] > 0):
            moved = True
check("with KL anchor, only the union of the two tasks' blocks moves", not moved)
new_actions = mk.sample_actions(mk.anchor_obs, task_id=0, temperature=0.0)
check("task-0 actions on its anchor states stay close under lambda=0.01 + KL", np.max(np.abs(new_actions - old_actions)) < 0.05)
sd_k = jax.device_get(mk.state_dict())
mk2 = make_masked_kl(seed=1)
mk2.load_state_dict(sd_k)
check("anchors + masks survive the state round trip", mk2.anchor_obs.shape[0] == 64 and np.array_equal(np.asarray(mk2.neuron_mask_table), np.asarray(mk.neuron_mask_table)))

status = "PASS" if all(results.values()) else "FAIL"
print("MASKDECAY_TESTS", {"test_status": status, "passed": sum(results.values()), "total": len(results)})
sys.exit(0 if status == "PASS" else 1)
