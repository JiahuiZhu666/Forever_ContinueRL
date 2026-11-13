#!/usr/bin/env python3
"""Scheme C + KL + neuron release: at the boundary before task t, release UNIMPORTANT neurons from old tasks' masks,
checked with KL (advisor's guide, section 3, with KL in place of action MSE).

For every old task j < t and every hidden layer:
  1. Order j's active neurons by the guide's contribution U_{j,i} = E_o|h_{j,i}(o)| * sum_{k active downstream}|W_ik|
     (h = the output actually passed on; last hidden layer: after the head LayerNorm, both heads), lowest first.
     U only sets the checking order; it never authorises a deletion.
  2. Try one neuron at a time: remove it from task j's mask and rerun the full actor (mask-dependent LayerNorms
     recomputed) on j's stored anchor states. Accept only if
       step KL  = KL(pi_j before this deletion || pi_j after)          <= kl_step
       cum  KL  = KL(pi_j reference stored at learning || pi_j after) <= kl_cum
     (the same per-state KL, summed over action dims and averaged over states, as the training KL anchor).
  3. Stop a layer after floor(frac * active) acceptances or max_tries tries; never empty a layer; skip a task whose
     current cumulative KL already exceeds kl_cum.
Afterwards a neuron referenced by no old task any more is recycled (guide): incoming kernel column and bias
re-initialised, outgoing row (next layer, or both heads) zeroed. Per-weight history counts are recomputed from the
revised masks (counts = sum over old tasks of each task's weight block), so released connections decay less.

usage: neuron_release.py --state start_taskT.pkl --task T --history m0.npz,m1.npz,... --out-dir DIR [options]
Writes: DIR/start_task{T}_released.pkl, DIR/released_mask_task{j}_for{T}.npz, DIR/counts_released_for{T}.npz,
        DIR/release_for{T}.json (log + priority pool per layer).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax.core import freeze, unfreeze  # noqa: E402

import maskdecay as md  # noqa: E402
from train_maskdecay_candidate import load_learner_class  # noqa: E402

KEYS = ("num_tasks", "actor_hidden_dims", "critic_hidden_dims", "learning_rate", "decay_factor", "mask_fraction",
        "refresh_interval", "reset_critic", "reset_actor_optimizer", "reset_temperature", "discount", "tau",
        "target_update_period", "target_entropy", "init_temperature", "max_grad_norm", "final_fc_init_scale",
        "clip_mean", "log_std_min", "log_std_max", "actor_variant", "temperature_mode", "critic_warmup_updates",
        "importance_mode", "importance_beta", "scale_floor", "importance_samples", "importance_accumulate",
        "anchor_mode", "anchor_coef", "anchor_states_per_task", "anchor_batch", "actor_lr_ramp_updates",
        "condition_ids", "anchor_budget", "anchor_target", "log_temperature_min", "task_code_scale",
        "reset_all_optimizers", "forward_masking", "scale_position", "actor_optimizer", "actor_learning_rate")
EPS = 1e-6


def build_agent(state, obs_dim, act_dim):
    cfg = state["config"]
    kw = {k: (tuple(cfg[k]) if k in ("condition_ids", "actor_hidden_dims", "critic_hidden_dims") else cfg[k])
          for k in KEYS if k in cfg}
    agent = load_learner_class()(seed=1, observations=np.zeros((1, obs_dim), np.float32),
                                 actions=np.zeros((1, act_dim), np.float32), **kw)
    agent.load_state_dict(state)
    return agent


def kl_gauss(mu_o, s_o, mu_n, s_n):
    """Mean over states of sum over dims of KL(N(mu_o, s_o) || N(mu_n, s_n)) (same formula as the KL anchor)."""
    kl = jnp.log(s_n / s_o) + (s_o ** 2 + (mu_o - mu_n) ** 2) / (2.0 * s_n ** 2) - 0.5
    return float(jnp.mean(jnp.sum(kl, axis=-1)))


class Releaser:
    def __init__(self, state, obs_dim, act_dim):
        self.state = state
        self.agent = build_agent(state, obs_dim, act_dim)
        cfg = state["config"]
        self.cfg = cfg
        raw = state["learner_state"]["actor_params"]
        raw = unfreeze(raw) if type(raw).__name__ == "FrozenDict" else raw
        self.params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), md.to_plain_tree(raw))
        self.masks = np.asarray(state["neuron_masks"], np.float32).copy()  # [num_tasks, L, W]
        self.L, self.W = self.masks.shape[1], self.masks.shape[2]
        a = state["anchors"]
        self.a_obs, self.a_cond = np.asarray(a["obs"], np.float32), np.asarray(a["cond"])
        self.a_mu, self.a_sigma = np.asarray(a["mu"], np.float32), np.asarray(a["sigma"], np.float32)
        self.num_conditions = int(cfg["num_conditions"])
        self.code_scale = float(cfg.get("task_code_scale", 1.0))
        actor = self.agent.actor
        self._dist = jax.jit(lambda p, x, m: actor.apply({"params": p}, x, neuron_masks=m))
        self._acts = jax.jit(self._activations)

    # -- inputs and policies --------------------------------------------------------------------------------
    def inputs(self, j):
        sel = self.a_cond == j
        obs = jnp.asarray(self.a_obs[sel])
        code = jax.nn.one_hot(jnp.full((obs.shape[0],), j), self.num_conditions, dtype=obs.dtype) * self.code_scale
        return jnp.concatenate((obs, code), axis=-1), jnp.asarray(self.a_mu[sel]), jnp.asarray(self.a_sigma[sel])

    def dist(self, x, mask, params=None):
        return self._dist(self.params if params is None else params, x, jnp.asarray(mask))

    def _activations(self, p, x, mask):
        """Per-layer outputs actually passed on (stable FullPolicy): L0 masked LN + tanh, L1.. leaky_relu,
        last layer additionally the head masked LayerNorm."""
        acts = []
        for i in range(self.L):
            k, b = p[f"backbone_{i}"]["kernel"], p[f"backbone_{i}"]["bias"]
            m = mask[i]
            x = (x @ k + b) * m
            if i == 0:
                cnt = jnp.maximum(jnp.sum(m), 1.0)
                mu = jnp.sum(x * m, -1, keepdims=True) / cnt
                c = (x - mu) * m
                x = jnp.tanh(c * jax.lax.rsqrt(jnp.sum(c * c, -1, keepdims=True) / cnt + EPS))
            else:
                x = jax.nn.leaky_relu(x)
            acts.append(x)
        m = mask[self.L - 1]
        cnt = jnp.maximum(jnp.sum(m), 1.0)
        mu = jnp.sum(x * m, -1, keepdims=True) / cnt
        c = (x - mu) * m
        acts[-1] = c * jax.lax.rsqrt(jnp.sum(c * c, -1, keepdims=True) / cnt + EPS)
        return acts

    def contributions(self, x, mask):
        acts = self._acts(self.params, x, jnp.asarray(mask))
        U = []
        for i in range(self.L):
            e_abs = np.asarray(jnp.mean(jnp.abs(acts[i]), 0))
            if i < self.L - 1:
                out = np.asarray(jnp.sum(jnp.abs(self.params[f"backbone_{i + 1}"]["kernel"]) * mask[i + 1][None, :], 1))
            else:
                out = np.asarray(jnp.sum(jnp.abs(self.params["mean"]["kernel"]), 1) +
                                 jnp.sum(jnp.abs(self.params["std"]["kernel"]), 1))
            U.append(np.where(mask[i] > 0, e_abs * out, np.inf))
        return U

    def check_forward(self, j):
        """The activation forward must reproduce the learner's mean (sanity check)."""
        x, _, _ = self.inputs(j)
        mu, _ = self.dist(x[:256], self.masks[j])
        h = self._acts(self.params, x[:256], jnp.asarray(self.masks[j]))[-1]
        mu2 = (h @ self.params["mean"]["kernel"]) * float(self.cfg["clip_mean"])
        return float(jnp.max(jnp.abs(mu - mu2)))

    # -- release --------------------------------------------------------------------------------------------
    def release(self, t, kl_step, kl_cum, frac, max_tries, do_recycle=True):
        log, released_pairs = [], []
        before = self.masks[:t].copy()
        for j in range(t):
            x, ref_mu, ref_s = self.inputs(j)
            mask = self.masks[j].copy()
            mu_c, s_c = self.dist(x, mask)
            cum0 = kl_gauss(ref_mu, ref_s, mu_c, s_c)
            entry = {"task": j, "cum_kl_start": cum0, "layers": []}
            if cum0 > kl_cum:
                entry["skipped"] = "cumulative KL already above the limit"
                log.append(entry)
                continue
            U = self.contributions(x, mask)
            for l in range(self.L):
                active = np.flatnonzero(mask[l] > 0)
                cap = int(np.floor(frac * active.size))
                order = active[np.argsort(U[l][active])]
                tries, acc, rows = 0, 0, []
                for i in order:
                    if acc >= cap or tries >= max_tries or mask[l].sum() <= 1:
                        break
                    tries += 1
                    trial = mask.copy()
                    trial[l, i] = 0.0
                    mu_t, s_t = self.dist(x, trial)
                    ks = kl_gauss(mu_c, s_c, mu_t, s_t)
                    kc = kl_gauss(ref_mu, ref_s, mu_t, s_t)
                    ok = ks <= kl_step and kc <= kl_cum
                    rows.append({"neuron": int(i), "U": float(U[l][i]), "step_kl": ks, "cum_kl": kc, "accepted": bool(ok)})
                    if ok:
                        mask, mu_c, s_c = trial, mu_t, s_t
                        acc += 1
                        released_pairs.append((j, l, int(i)))
                entry["layers"].append({"layer": l, "active_before": int(active.size), "cap": cap, "tries": tries,
                                        "released": acc, "checks": rows})
            entry["cum_kl_end"] = kl_gauss(ref_mu, ref_s, mu_c, s_c)
            self.masks[j] = mask
            log.append(entry)
        # recycle neurons that no old task references any more
        refs_before = before.sum(0)            # [L, W]
        refs_after = self.masks[:t].sum(0)
        recycled = [(l, int(i)) for l in range(self.L) for i in np.flatnonzero((refs_before[l] > 0) & (refs_after[l] == 0))]
        if do_recycle:
            self.recycle(recycled, seed=int(1000 + t))
        released_neurons = sorted({(l, i) for _, l, i in released_pairs})
        # Priority pool (variant B): only neurons released at THIS boundary (from at least one old task; includes
        # those recycled because no old task references them any more). Never-used neurons are not "released"
        # and are deliberately excluded, otherwise early tasks would simply pick brand-new neurons.
        pool = []
        for l in range(self.L):
            pool.append(sorted({i for (ll, i) in released_neurons if ll == l}))
        return {"log": log, "released_pairs": released_pairs, "recycled": recycled, "pool": pool,
                "refs_before": refs_before.tolist(), "refs_after": refs_after.tolist()}

    def recycle(self, neurons, seed):
        """Guide: re-initialise incoming kernel column and bias, zero the outgoing row (next layer, or both heads)."""
        if not neurons:
            return
        p = jax.tree_util.tree_map(lambda a: np.array(a, np.float32), self.params)
        rng = np.random.default_rng(seed)
        # Pass 1: re-initialise every recycled neuron's incoming column and bias.
        for l, i in neurons:
            k = p[f"backbone_{l}"]["kernel"]
            fan_in = k.shape[0]
            # variance_scaling(2.0, fan_in, truncated_normal): std = sqrt(2/fan_in)/0.87962566, truncated at 2 std
            std = np.sqrt(2.0 / fan_in) / 0.87962566103423978
            z = rng.standard_normal(fan_in)
            while np.any(np.abs(z) > 2):
                bad = np.abs(z) > 2
                z[bad] = rng.standard_normal(int(bad.sum()))
            k[:, i] = z * std
            p[f"backbone_{l}"]["bias"][i] = 0.0
        # Pass 2: zero every recycled neuron's outgoing row (after pass 1, so a re-initialised incoming column of a
        # recycled neuron in the next layer cannot overwrite the zeroed outgoing row of one in this layer).
        for l, i in neurons:
            if l < self.L - 1:
                p[f"backbone_{l + 1}"]["kernel"][i, :] = 0.0
            else:
                p["mean"]["kernel"][i, :] = 0.0
                p["std"]["kernel"][i, :] = 0.0
        self.params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float32), p)

    def new_state(self):
        st = dict(self.state)
        ls = dict(st["learner_state"])
        orig = st["learner_state"]["actor_params"]
        p = jax.tree_util.tree_map(lambda a: np.asarray(a, np.float32), self.params)
        ls["actor_params"] = freeze(p) if type(orig).__name__ == "FrozenDict" else p
        st["learner_state"] = ls
        st["neuron_masks"] = self.masks.copy()
        return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--task", type=int, required=True, help="index of the task about to be trained (old tasks: 0..T-1)")
    ap.add_argument("--history", required=True, help="comma-separated winner neuron-mask npz files of tasks 0..T-1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kl-step", type=float, default=0.01)
    ap.add_argument("--kl-cum", type=float, default=0.03)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--max-tries", type=int, default=16)
    ap.add_argument("--hidden-layers", type=int, default=4)
    ap.add_argument("--include-biases", type=int, default=1)
    a = ap.parse_args()
    t = a.task
    state = pickle.load(open(a.state, "rb"))
    hist = [md.load_neuron_mask(p) for p in a.history.split(",")]
    assert len(hist) == t, "need one winner mask per old task"
    obs_dim = np.asarray(state["anchors"]["obs"]).shape[1]
    act_dim = np.asarray(state["anchors"]["mu"]).shape[1]
    r = Releaser(state, obs_dim, act_dim)
    # the learner's mask table and the ledger's winner masks must agree for old tasks
    for j in range(t):
        assert np.array_equal(np.stack(hist[j]).astype(np.float32), r.masks[j]), f"mask table differs for task {j}"
    diff = max(r.check_forward(j) for j in range(t))
    assert diff < 1e-4, f"activation forward does not reproduce the learner (max diff {diff})"
    res = r.release(t, a.kl_step, a.kl_cum, a.frac, a.max_tries)
    os.makedirs(a.out_dir, exist_ok=True)
    st = r.new_state()
    with open(os.path.join(a.out_dir, f"start_task{t:02d}_released.pkl"), "wb") as f:
        pickle.dump(st, f, protocol=pickle.HIGHEST_PROTOCOL)
    files = []
    for j in range(t):
        pth = os.path.join(a.out_dir, f"released_mask_task{j:02d}_for{t:02d}.npz")
        md.save_neuron_mask(tuple(r.masks[j][l].astype(np.uint8) for l in range(r.L)), pth)
        files.append(pth)
    tree = md.to_plain_tree(jax.device_get(r.params))
    specs = md.kernel_specs_for(a.hidden_layers, include_biases=bool(a.include_biases))
    counts = md.init_counts(tree)
    for j in range(t):
        counts = md.commit_counts(counts, md.build_weight_mask(tree, tuple(r.masks[j][l].astype(np.uint8) for l in range(r.L)), specs))
    counts_path = os.path.join(a.out_dir, f"counts_released_for{t:02d}.npz")
    md.save_tree_npz(counts, counts_path)
    summary = {
        "task": t, "kl_step": a.kl_step, "kl_cum": a.kl_cum, "frac": a.frac, "max_tries": a.max_tries,
        "released_pairs": len(res["released_pairs"]), "recycled_neurons": len(res["recycled"]),
        "released_per_task_layer": {str(e["task"]): [ly["released"] for ly in e.get("layers", [])] for e in res["log"]},
        "cum_kl": {str(e["task"]): [e["cum_kl_start"], e.get("cum_kl_end")] for e in res["log"]},
        "pool_sizes": [len(p) for p in res["pool"]],
        "state_out": os.path.join(a.out_dir, f"start_task{t:02d}_released.pkl"), "mask_files": files,
        "counts_out": counts_path,
    }
    json.dump({"summary": summary, **res}, open(os.path.join(a.out_dir, f"release_for{t:02d}.json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("RELEASE_DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
