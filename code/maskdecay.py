"""Fixed-capacity Nevo-style mask generation with cross-task update decay (method document v1.1).

Host-side (NumPy) pieces: candidate neuron-mask generation with the Nevo initialisation rule,
neuron-mask -> weight-mask mapping by adjacent-layer outer products, per-weight history counts,
candidate multipliers s = lambda ** max(h + M - 1, 0), and winner commitment. Nothing here touches
the forward pass: the actor stays dense; the multipliers scale the post-Adam actor increments.

The NumPy core follows the method document verbatim (section 11); the helpers at the end adapt it
to the cw_v2 FullPolicy parameter tree and to npz files.
"""
from collections.abc import Mapping
import json
import numpy as np


# --------------------------------------------------------------------------------------------------
# Method document section 11: NumPy core
# --------------------------------------------------------------------------------------------------
def map_params(fn, *trees, path=()):
    """Map aligned dict/list/tuple parameter trees."""
    if not trees:
        raise ValueError("at least one tree is required")
    first = trees[0]
    if isinstance(first, Mapping):
        if any(not isinstance(t, Mapping) or set(t) != set(first) for t in trees[1:]):
            raise ValueError("parameter tree structures differ")
        return type(first)({k: map_params(fn, *(t[k] for t in trees), path=path + (k,)) for k in first})
    if isinstance(first, (list, tuple)):
        if any(type(t) is not type(first) or len(t) != len(first) for t in trees[1:]):
            raise ValueError("parameter tree structures differ")
        return type(first)([map_params(fn, *(t[i] for t in trees), path=path + (i,)) for i in range(len(first))])
    if any(isinstance(t, (Mapping, list, tuple)) for t in trees[1:]):
        raise ValueError("parameter tree leaf structures differ")
    return fn(path, *trees)


def as_embedding(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("task embedding must be finite and nonempty")
    if np.linalg.norm(value) == 0:
        raise ValueError("task embedding must have nonzero norm")
    return value


def validate_neuron_mask(mask, widths):
    if len(mask) != len(widths):
        raise ValueError("wrong number of hidden-layer masks")
    result = []
    for layer_mask, width in zip(mask, widths):
        layer_mask = np.asarray(layer_mask)
        if layer_mask.shape != (width,):
            raise ValueError("hidden-layer mask shape mismatch")
        if not np.all((layer_mask == 0) | (layer_mask == 1)):
            raise ValueError("neuron mask must be binary")
        result.append(layer_mask.astype(np.uint8, copy=True))
    return tuple(result)


def nevo_population(widths, history_masks, history_embeddings, current_embedding, rng,
                    candidate_count=5, sample_ratio=0.25, min_ratio=0.01,
                    semantic_top_k=5, temperature=0.5, fixed_new_counts=None):
    """Nevo initialization rule; does not change any supplied history.
    fixed_new_counts: optional per-layer neuron counts. When given, every task after the first adds exactly
    that many never-used neurons per layer to the inherited parent mask, instead of Nevo's
    similarity-dependent ratio (parent selection is unchanged). The first task still samples sample_ratio."""
    widths = tuple(int(w) for w in widths)
    if not widths or any(w <= 0 for w in widths):
        raise ValueError("hidden-layer widths must be positive")
    if candidate_count < 1 or semantic_top_k < 1 or temperature <= 0:
        raise ValueError("invalid population or similarity configuration")
    if not 0 <= min_ratio <= sample_ratio <= 1:
        raise ValueError("invalid sampling ratios")
    if fixed_new_counts is not None:
        fixed_new_counts = tuple(int(c) for c in fixed_new_counts)
        if len(fixed_new_counts) != len(widths) or any(not 0 <= c <= w for c, w in zip(fixed_new_counts, widths)):
            raise ValueError("fixed_new_counts must give one count in [0, width] per layer")
    if len(history_masks) != len(history_embeddings):
        raise ValueError("mask and task-embedding histories differ")

    current_e = as_embedding(current_embedding)
    history = [validate_neuron_mask(a, widths) for a in history_masks]
    embeddings = [as_embedding(e) for e in history_embeddings]
    if any(e.shape != current_e.shape for e in embeddings):
        raise ValueError("task embedding dimensions differ")

    occupied = [np.zeros(w, dtype=bool) for w in widths]
    for task_mask in history:
        for layer, a in enumerate(task_mask):
            occupied[layer] |= a.astype(bool)
    free = [np.flatnonzero(~u) for u in occupied]

    if history:
        similarities = np.asarray([np.dot(e, current_e) / (np.linalg.norm(e) * np.linalg.norm(current_e))
                                   for e in embeddings])
        top = np.argsort(similarities)[::-1][:semantic_top_k]
        logits = similarities[top] / temperature
        weights = np.exp(logits - np.max(logits))
        probabilities = weights / weights.sum()

    population, records = [], []
    for _ in range(candidate_count):
        if history:
            parent = int(rng.choice(top, p=probabilities))
            similarity = float(similarities[parent])
            ratio = min_ratio + max(0.0, 1.0 - similarity) * (sample_ratio - min_ratio)
            candidate = [a.copy() for a in history[parent]]
        else:
            parent, similarity, ratio = None, None, sample_ratio
            candidate = [np.zeros(w, dtype=np.uint8) for w in widths]

        # All candidates use the SAME historical free pools.
        requested, added = [], []
        if history and fixed_new_counts is not None:
            ratio = None
        for layer, width in enumerate(widths):
            want = fixed_new_counts[layer] if ratio is None else int(ratio * width)
            take = min(want, free[layer].size)
            if take:
                chosen = rng.choice(free[layer], size=take, replace=False)
                candidate[layer][chosen] = 1
            requested.append(want)
            added.append(take)

        population.append(tuple(candidate))
        records.append({
            "parent_task_index_zero_based": parent,
            "similarity": similarity,
            "new_ratio": ratio,
            "new_count_rule": "fixed" if (history and fixed_new_counts is not None) else "nevo_ratio",
            "requested_new_counts": requested,
            "actual_new_counts": added,
            "selected_counts": [int(a.sum()) for a in candidate],
            "free_counts_at_task_start": [int(f.size) for f in free],
        })
    return population, records


def random_population(widths, history_masks, rng, candidate_count=5, sample_ratio=0.25,
                      extra_new_counts=None, extra_rng=None):
    """Scheme C rule: every candidate draws an independent random floor(sample_ratio * N) neurons per layer
    from ALL positions (draw order identical to the original inline implementation in run_maskdecay.py).
    extra_new_counts (optional, per layer): when there is history, additionally add that many neurons that
    no earlier winner has used and that are not already in this candidate's random draw, drawn with
    extra_rng so the random part stays identical to the run without extras. All weights touching such a
    neuron have count 0 and therefore multiplier 1 (full speed)."""
    widths = tuple(int(w) for w in widths)
    history = [validate_neuron_mask(a, widths) for a in history_masks]
    use_extra = extra_new_counts is not None and len(history) > 0
    if extra_new_counts is not None:
        extra_new_counts = tuple(int(c) for c in extra_new_counts)
        if len(extra_new_counts) != len(widths) or any(not 0 <= c <= w for c, w in zip(extra_new_counts, widths)):
            raise ValueError("extra_new_counts must give one count in [0, width] per layer")
        if extra_rng is None:
            raise ValueError("extra_new_counts needs its own extra_rng")
    used = [np.zeros(w, dtype=bool) for w in widths]
    for task_mask in history:
        for layer, a in enumerate(task_mask):
            used[layer] |= a.astype(bool)
    population, records = [], []
    for _ in range(candidate_count):
        cand = []
        for w in widths:
            z = np.zeros(w, np.uint8)
            z[rng.choice(w, size=int(sample_ratio * w), replace=False)] = 1
            cand.append(z)
        requested, added, free_counts = [], [], []
        if use_extra:
            for layer, z in enumerate(cand):
                free = np.flatnonzero(~used[layer] & (z == 0))
                take = min(extra_new_counts[layer], free.size)
                if take:
                    z[extra_rng.choice(free, size=take, replace=False)] = 1
                requested.append(extra_new_counts[layer])
                added.append(int(take))
                free_counts.append(int((~used[layer]).sum()))
        overlap = [int((z & h).sum()) for z, h in zip(cand, history[-1])] if history else None
        population.append(tuple(cand))
        records.append({"parent_task_index_zero_based": None, "similarity": None, "new_ratio": sample_ratio,
                        "rule": "random+extra_new" if use_extra else "random",
                        "random_counts": [int(sample_ratio * w) for w in widths],
                        "requested_extra_new_counts": requested if use_extra else None,
                        "actual_extra_new_counts": added if use_extra else None,
                        "never_used_counts_at_task_start": free_counts if use_extra else [int((~u).sum()) for u in used],
                        "requested_new_counts": [int(z.sum()) for z in cand],
                        "actual_new_counts": [int(z.sum()) for z in cand],
                        "selected_counts": [int(z.sum()) for z in cand],
                        "overlap_with_previous_winner": overlap})
    return population, records


def priority_population(widths, history_masks, pool, rng, candidate_count=5, sample_ratio=0.25):
    """Scheme C with priority for released neurons: each candidate first takes, per layer, neurons from the priority
    pool (neurons released from old tasks at this boundary) uniformly at random, up to floor(sample_ratio * N),
    then fills the rest uniformly from the remaining neurons."""
    widths = tuple(int(w) for w in widths)
    history = [validate_neuron_mask(a, widths) for a in history_masks]
    if len(pool) != len(widths):
        raise ValueError("pool must give one index list per layer")
    population, records = [], []
    for _ in range(candidate_count):
        cand, from_pool = [], []
        for layer, w in enumerate(widths):
            n = int(sample_ratio * w)
            p = np.asarray(sorted(set(int(i) for i in pool[layer])), dtype=np.int64)
            if p.size and (p.min() < 0 or p.max() >= w):
                raise ValueError("pool index out of range")
            k = min(n, p.size)
            chosen = rng.choice(p, size=k, replace=False) if k else np.zeros(0, np.int64)
            rest = np.setdiff1d(np.arange(w), chosen)
            fill = rng.choice(rest, size=n - k, replace=False) if n - k > 0 else np.zeros(0, np.int64)
            z = np.zeros(w, np.uint8)
            z[chosen] = 1
            z[fill] = 1
            cand.append(z)
            from_pool.append(int(k))
        overlap = [int((z & h).sum()) for z, h in zip(cand, history[-1])] if history else None
        population.append(tuple(cand))
        records.append({"parent_task_index_zero_based": None, "similarity": None, "new_ratio": sample_ratio,
                        "rule": "random+priority_released", "pool_sizes": [len(set(p)) for p in pool],
                        "taken_from_pool": from_pool,
                        "requested_new_counts": [int(z.sum()) for z in cand],
                        "actual_new_counts": [int(z.sum()) for z in cand],
                        "selected_counts": [int(z.sum()) for z in cand],
                        "overlap_with_previous_winner": overlap})
    return population, records


def build_weight_mask(actor_params, neuron_mask, kernel_specs):
    """Each spec is (kernel_path, pre_hidden_index, post_hidden_index).
    None means all input/action coordinates on that side are selected.
    Neuron layer indices here start at zero.
    """
    masks = tuple(np.asarray(a, dtype=np.uint8) for a in neuron_mask)
    for a in masks:
        if a.ndim != 1 or not np.all((a == 0) | (a == 1)):
            raise ValueError("invalid neuron mask")
    specs = {}
    for path, pre_index, post_index in kernel_specs:
        path = tuple(path)
        if path in specs or not path or path[-1] not in ("kernel", "bias"):
            raise ValueError("invalid or duplicate kernel specification")
        for index in (pre_index, post_index):
            if index is not None and not 0 <= index < len(masks):
                raise ValueError("hidden-layer index out of range")
        specs[path] = (pre_index, post_index)

    seen = set()

    def lift(path, value):
        value = np.asarray(value)
        if path not in specs:
            if path and path[-1] == "kernel":
                raise ValueError("unregistered Actor kernel: " + str(path))
            return np.zeros(value.shape, dtype=np.uint8)
        pre_index, post_index = specs[path]
        if path[-1] == "bias":  # registered bias vector: mask = its layer's neuron mask
            if value.ndim != 1 or post_index is None or value.shape != masks[post_index].shape:
                raise ValueError("registered bias must be a vector matching its layer mask")
            seen.add(path)
            return masks[post_index].astype(np.uint8).copy()
        if value.ndim != 2 or value.dtype.kind != "f":
            raise ValueError("registered kernel must be a floating matrix")
        pre = np.ones(value.shape[0], dtype=np.uint8) if pre_index is None else masks[pre_index]
        post = np.ones(value.shape[1], dtype=np.uint8) if post_index is None else masks[post_index]
        if (pre.size, post.size) != value.shape:
            raise ValueError("neuron masks and kernel shape differ")
        seen.add(path)
        return np.outer(pre, post).astype(np.uint8)

    result = map_params(lift, actor_params)
    if seen != set(specs):
        raise ValueError("registered kernel path missing from Actor")
    return result


def init_counts(actor_params):
    return map_params(lambda path, value: np.zeros(np.asarray(value).shape, dtype=np.int64), actor_params)


def validate_count_and_mask(history, mask):
    def validate(path, h, m):
        h, m = np.asarray(h), np.asarray(m)
        if h.shape != m.shape:
            raise ValueError("count and weight mask shapes differ")
        if h.dtype.kind not in "iu" or np.any(h < 0):
            raise ValueError("history must contain nonnegative integers")
        if not np.all((m == 0) | (m == 1)):
            raise ValueError("weight mask must be binary")
        if h.size and np.max(h) >= np.iinfo(np.int64).max:
            raise OverflowError("history counter exhausted")
        return None
    map_params(validate, history, mask)


def candidate_scales(history, weight_mask, decay):
    if not 0 < decay <= 1:  # decay=1 permits a no-decay baseline.
        raise ValueError("decay must be in (0, 1]")
    validate_count_and_mask(history, weight_mask)
    return map_params(
        lambda path, h, m: np.power(
            decay, np.maximum(np.asarray(h, dtype=np.int64) + np.asarray(m, dtype=np.int64) - 1, 0)
        ).astype(np.float32),
        history, weight_mask,
    )


def commit_counts(history, winner_weight_mask):
    """Compute new counts without mutating the old history."""
    validate_count_and_mask(history, winner_weight_mask)
    return map_params(lambda path, h, m: (np.asarray(h, dtype=np.int64) + np.asarray(m, dtype=np.int64)),
                      history, winner_weight_mask)


# --------------------------------------------------------------------------------------------------
# Adapters for the cw_v2 FullPolicy actor and for files
# --------------------------------------------------------------------------------------------------
def kernel_specs_for(num_hidden_layers, include_biases=False):
    """FullPolicy kernels: backbone_0 (input x hidden_0), backbone_i (hidden_{i-1} x hidden_i),
    mean / std heads (hidden_{L-1} x action). LayerNorms carry no parameters.
    include_biases: also register the hidden-layer biases (mask = that layer's neuron mask), the
    forward-masking variant's analogue of Nevo freezing old neurons' biases. The std head bias is
    shared by every task and stays at multiplier 1."""
    L = int(num_hidden_layers)
    specs = [(("backbone_0", "kernel"), None, 0)]
    for i in range(1, L):
        specs.append((("backbone_{}".format(i), "kernel"), i - 1, i))
    specs.append((("mean", "kernel"), L - 1, None))
    specs.append((("std", "kernel"), L - 1, None))
    if include_biases:
        for i in range(L):
            specs.append((("backbone_{}".format(i), "bias"), None, i))
    return specs


def to_plain_tree(tree):
    """FrozenDict / dict of arrays -> nested plain dicts of numpy arrays."""
    if isinstance(tree, Mapping):
        return {k: to_plain_tree(v) for k, v in tree.items()}
    return np.asarray(tree)


def flatten_tree(tree, prefix=()):
    out = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            out.update(flatten_tree(v, prefix + (str(k),)))
    else:
        out["/".join(prefix)] = np.asarray(tree)
    return out


def unflatten_tree(flat):
    tree = {}
    for key, value in flat.items():
        node = tree
        parts = key.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = np.asarray(value)
    return tree


def save_tree_npz(tree, path):
    np.savez_compressed(path, **flatten_tree(tree))


def load_tree_npz(path):
    with np.load(path) as data:
        return unflatten_tree({k: data[k] for k in data.files})


def save_neuron_mask(mask, path):
    np.savez_compressed(path, **{"layer_{}".format(i): np.asarray(a, dtype=np.uint8) for i, a in enumerate(mask)})


def load_neuron_mask(path):
    with np.load(path) as data:
        return tuple(data["layer_{}".format(i)] for i in range(len(data.files)))


def scale_statistics(weight_mask, counts, scales, kernel_specs):
    """Per-kernel selection ratio, count distribution and multiplier distribution (for the log)."""
    flat_m, flat_h, flat_s = flatten_tree(weight_mask), flatten_tree(counts), flatten_tree(scales)
    per_kernel = {}
    total = {"weights": 0, "selected": 0, "scaled": 0}
    for path, _, _ in kernel_specs:
        key = "/".join(path)
        m, h, s = flat_m[key], flat_h[key], flat_s[key]
        per_kernel[key] = {
            "weights": int(m.size), "selected_fraction": float(m.mean()),
            "count_histogram": {str(int(c)): int(n) for c, n in zip(*np.unique(h, return_counts=True))},
            "scale_lt1_fraction": float(np.mean(s < 1.0)), "scale_min": float(s.min()), "scale_mean": float(s.mean()),
        }
        total["weights"] += int(m.size); total["selected"] += int(m.sum()); total["scaled"] += int(np.sum(s < 1.0))
    total["selected_fraction"] = total["selected"] / max(total["weights"], 1)
    total["scaled_fraction"] = total["scaled"] / max(total["weights"], 1)
    return {"per_kernel": per_kernel, "total": total}


def embed_task_hints(hints, model_name="all-MiniLM-L12-v2"):
    """Fixed task encoder (same model as Nevo-CRL); imported lazily so trainers never load torch."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return np.asarray([as_embedding(model.encode(h)) for h in hints], dtype=np.float64)


def dump_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
