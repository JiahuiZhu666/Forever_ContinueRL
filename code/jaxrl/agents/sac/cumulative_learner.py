"""Full-network SAC with permanent, task-counted per-weight Adam scaling.

The count rule is N_i += 1 on the first selection of coordinate i during a
task. The multiplier is lambda ** N_i, including the first selected task.
Refreshing a task mask never counts a coordinate twice, and leaving a mask
never restores its learning rate. Biases and normalization parameters are not
selected. There is no multiplier floor, original neuron mask, population,
distillation, or implicit learning-rate schedule.

Only JAX, Flax, Optax and NumPy are imported. This module is independent of the
old package initializer that imports language models and dictionary learners.
The default legacy policy retains the CW MetaPolicy mean/std parameterization.
The opt-in stable policy normalizes the features entering the heads, uses a
linear Gaussian mean, and exponentiates clipped log standard deviations. Both
use every neuron and the same permanent count rule. Their tanh-normal density
uses the exact, stable change of
variables rather than the old TFP distribution's clipped-tail approximation.
"""

import copy
import functools
import math
from typing import Any, Sequence

import flax.linen as nn
from flax import serialization, struct
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _initializer(scale=2.0):
    # Matches CW/jaxrl/networks/common.py:default_init.
    return nn.initializers.variance_scaling(scale, "fan_in", "truncated_normal")


def _stable_std_constants(log_std_min, log_std_max):
    """Use the legacy policy's sigma endpoints, expressed as log sigma."""
    def softplus(value):
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))
    minimum, maximum = softplus(log_std_min), softplus(log_std_max)
    if minimum <= 0.0 or not math.isfinite(maximum):
        raise ValueError("stable actor standard-deviation bounds are not representable")
    return math.log(minimum), math.log(maximum), math.log(math.log(2.0))


def _gradient_coordinate_stats(gradients):
    """Inspect actual coordinates without squaring tiny float32 gradients."""
    leaves = jax.tree_util.tree_leaves(gradients)
    maximum = jnp.max(jnp.stack([jnp.max(jnp.abs(gradient)) for gradient in leaves]))
    nonzero_count = sum(jnp.count_nonzero(gradient) for gradient in leaves)
    return maximum, nonzero_count


def _masked_layer_norm(x, mask, epsilon):
    """Parameter-free LayerNorm over the active neurons only (inactive outputs stay exactly 0).
    With an all-ones mask this equals flax's LayerNorm(use_bias=False, use_scale=False)."""
    mask = mask.astype(x.dtype)
    count = jnp.maximum(jnp.sum(mask, axis=-1, keepdims=True), 1.0)  # per row (masks may be per sample)
    mean = jnp.sum(x * mask, axis=-1, keepdims=True) / count
    centered = (x - mean) * mask
    variance = jnp.sum(centered * centered, axis=-1, keepdims=True) / count
    return centered * jax.lax.rsqrt(variance + epsilon)


def _layer_mask(neuron_masks, index):
    """Mask of one hidden layer: neuron_masks is [layers, width] (one task) or
    [batch, layers, width] (one task per sample, used for anchor batches)."""
    if neuron_masks is None:
        return None
    if neuron_masks.ndim == 3:
        return neuron_masks[:, index, :]
    return neuron_masks[index]


class FullPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-4
    clip_mean: float = 1.0
    log_std_min: float = -2.0
    log_std_max: float = 2.0
    actor_variant: str = "legacy"

    @nn.compact
    def __call__(self, observations, return_diagnostics=False, neuron_masks=None):
        # neuron_masks: optional [num_hidden_layers, width] binary array (Nevo-style forward
        # masking). Every hidden pre-activation is multiplied by its layer's mask and the
        # parameter-free LayerNorms normalise over the active neurons only, so a task's output
        # depends on its own sub-network alone. None reproduces the dense actor exactly.
        x = observations
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, kernel_init=_initializer(), name="backbone_{}".format(index))(x)
            mask = _layer_mask(neuron_masks, index)
            if mask is not None:
                x = x * mask.astype(x.dtype)
            if index == 0:
                if mask is None:
                    x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                                     name="first_layer_norm")(x)
                else:
                    x = _masked_layer_norm(x, mask, 1e-6)
                x = jnp.tanh(x)
            else:
                x = jax.nn.leaky_relu(x)
        head_hidden = x
        last_mask = _layer_mask(neuron_masks, len(self.hidden_dims) - 1)
        if self.actor_variant == "stable":
            if last_mask is None:
                x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6,
                                 name="head_layer_norm")(x)
            else:
                x = _masked_layer_norm(x, last_mask, 1e-6)
        raw_mean = nn.Dense(self.action_dim, use_bias=False,
                         kernel_init=_initializer(self.final_fc_init_scale), name="mean")(x)
        if self.actor_variant == "stable":
            minimum, maximum, initial_bias = _stable_std_constants(self.log_std_min, self.log_std_max)
            raw_std = nn.Dense(self.action_dim, kernel_init=_initializer(self.final_fc_init_scale),
                               bias_init=nn.initializers.constant(initial_bias), name="std")(x)
            means = raw_mean * self.clip_mean
            std = jnp.exp(jnp.clip(raw_std, minimum, maximum))
            lower_clip = jnp.mean((raw_std <= minimum).astype(jnp.float32))
            upper_clip = jnp.mean((raw_std >= maximum).astype(jnp.float32))
            mean_saturation = jnp.asarray(0.0, jnp.float32)
            std_saturation = jnp.asarray(0.0, jnp.float32)
        else:
            means = jnp.tanh(raw_mean) * self.clip_mean
            raw_std = nn.Dense(self.action_dim, kernel_init=_initializer(self.final_fc_init_scale),
                               name="std")(x)
            bounded_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                jnp.tanh(raw_std) + 1.0)
            std = jax.nn.softplus(bounded_std)
            lower_clip = upper_clip = jnp.asarray(0.0, jnp.float32)
            mean_saturation = jnp.mean((jnp.abs(jnp.tanh(raw_mean)) == 1.0).astype(jnp.float32))
            std_saturation = jnp.mean((jnp.abs(jnp.tanh(raw_std)) == 1.0).astype(jnp.float32))
        if return_diagnostics:
            return means, std, {
                "mean_raw_absmax": jnp.max(jnp.abs(raw_mean)),
                "std_raw_absmax": jnp.max(jnp.abs(raw_std)),
                "mean_absmax": jnp.max(jnp.abs(means)),
                "std_min": jnp.min(std), "std_max": jnp.max(std),
                "std_clip_fraction": lower_clip + upper_clip,
                "std_lower_clip_fraction": lower_clip, "std_upper_clip_fraction": upper_clip,
                "mean_tanh_saturated_fraction": mean_saturation,
                "std_tanh_saturated_fraction": std_saturation,
                "head_hidden_absmax": jnp.max(jnp.abs(head_hidden)),
                "head_input_absmax": jnp.max(jnp.abs(x)),
                "head_input_rms": jnp.sqrt(jnp.mean(x * x)),
            }
        return means, std


class TwinCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate((observations, actions), axis=-1)
        values = []
        for critic_index in range(2):
            x = inputs
            for index, width in enumerate(self.hidden_dims):
                name = "q{}_hidden_{}".format(critic_index, index)
                x = nn.Dense(width, kernel_init=_initializer(), name=name)(x)
                x = nn.LayerNorm(name=name + "_norm")(x)
                x = jax.nn.leaky_relu(x)
            x = nn.Dense(1, kernel_init=_initializer(), name="q{}_output".format(critic_index))(x)
            values.append(jnp.squeeze(x, axis=-1))
        return jnp.stack(values, axis=0)


def _condition(observations, task_id, num_tasks):
    code = jax.nn.one_hot(task_id, num_tasks, dtype=observations.dtype)
    code = jnp.broadcast_to(code, observations.shape[:-1] + (num_tasks,))
    return jnp.concatenate((observations, code), axis=-1)


def _sample_and_log_prob(policy, params, observations, key, return_diagnostics=False, neuron_masks=None):
    if return_diagnostics:
        mean, std, diagnostics = policy.apply({"params": params}, observations, return_diagnostics=True,
                                              neuron_masks=neuron_masks)
    else:
        mean, std = policy.apply({"params": params}, observations, neuron_masks=neuron_masks)
    noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    pre_tanh = mean + std * noise
    actions = jnp.tanh(pre_tanh)
    # log Normal(z;mean,std), using the reparameterized noise; all paths retain
    # their dependence on params. The stable tanh Jacobian avoids log(1-a*a).
    normal_log_prob = -0.5 * noise ** 2 - jnp.log(std) - 0.5 * math.log(2.0 * math.pi)
    log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
    log_prob = jnp.sum(normal_log_prob - log_jacobian, axis=-1)
    if return_diagnostics:
        return actions, log_prob, diagnostics
    return actions, log_prob


def _scale_tree(params, counts, decay_factor):
    return jax.tree_util.tree_map(
        lambda parameter, count: jnp.power(jnp.asarray(decay_factor, dtype=parameter.dtype), count),
        params, counts)


@functools.partial(jax.jit, static_argnums=(3, 4))
def _refresh_counted_mask(params, counts, seen, mask_fraction, decay_factor):
    """Stable per-kernel TopK and task-local deduplication, entirely on device."""
    flat_params = flatten_dict(unfreeze(params))
    flat_counts = flatten_dict(unfreeze(counts))
    flat_seen = flatten_dict(unfreeze(seen))
    next_counts, next_seen, masks = {}, {}, {}
    for path, parameter in flat_params.items():
        if path[-1] == "kernel":
            size = parameter.size
            count = int(math.ceil(mask_fraction * size))
            # Stable ascending sort of -abs picks the lowest flattened index
            # first for tied magnitudes, including all-zero initial kernels.
            order = jnp.argsort(-jnp.abs(parameter).reshape(-1), kind="stable")
            mask = jnp.zeros((size,), dtype=jnp.bool_).at[order[:count]].set(True).reshape(parameter.shape)
        else:
            mask = jnp.zeros_like(parameter, dtype=jnp.bool_)
        newly_selected = jnp.logical_and(mask, jnp.logical_not(flat_seen[path]))
        next_counts[path] = flat_counts[path] + newly_selected.astype(jnp.int32)
        next_seen[path] = jnp.logical_or(flat_seen[path], mask)
        masks[path] = mask
    next_counts = freeze(unflatten_dict(next_counts))
    next_seen = freeze(unflatten_dict(next_seen))
    masks = freeze(unflatten_dict(masks))
    return next_counts, next_seen, masks, _scale_tree(params, next_counts, decay_factor)


@struct.dataclass
class _LearnerState:
    rng: Any
    actor_params: Any
    actor_opt_state: Any
    critic_params: Any
    critic_opt_state: Any
    target_critic_params: Any
    log_temperature: Any
    temperature_opt_state: Any
    counts: Any
    seen: Any
    current_mask: Any
    scales: Any
    importance: Any      # float32 tree, accumulated normalized importance (mas/fisher modes)
    anchor_params: Any   # actor parameters at the end of the previous task (ewc anchor)


class CumulativeSACLearner:
    """Single actor, two critics, fixed task one-hot, and cumulative LR counts.

    observations/actions are example raw observations and actions; their final
    dimensions define the spaces. Replay batches must contain raw observations,
    actions, rewards, masks (bootstrap continuation), and next_observations.
    init_temperature follows the old CW configuration: it initializes log alpha,
    so init_temperature=1.0 means the initial alpha is exp(1), not 1.

    All updates retain complete gradients. Actor kernel masks are measured at
    task start (before current-task data) and every refresh_interval actor
    updates. During a task each coordinate is counted at most once. Adam states
    may be reset at task boundaries, but counts never are. lambda=1 is an
    ordinary-Adam control with the same task conditioning and reset policies.
    """

    def __init__(self, seed, observations, actions, num_tasks,
                 actor_hidden_dims=(1024,) * 4, critic_hidden_dims=(256,) * 4,
                 learning_rate=3e-4, decay_factor=0.5, mask_fraction=0.1,
                 refresh_interval=10000, reset_critic=True,
                 reset_actor_optimizer=True, reset_temperature=True,
                 discount=0.99, tau=0.005, target_update_period=1,
                 target_entropy=None, init_temperature=1.0, max_grad_norm=1.0,
                 final_fc_init_scale=1e-4, clip_mean=1.0,
                 log_std_min=-2.0, log_std_max=2.0, actor_variant="legacy",
                 temperature_mode="shared_reset", critic_warmup_updates=0,
                 importance_mode="magnitude", importance_beta=10.0, scale_floor=0.05,
                 importance_samples=2048, importance_accumulate="max",
                 anchor_mode="none", anchor_coef=1.0, anchor_states_per_task=4096, anchor_batch=256,
                 actor_lr_ramp_updates=0, condition_ids=None, anchor_budget=0,
                 anchor_target="full", log_temperature_min=None, task_code_scale=1.0,
                 reset_all_optimizers=False, forward_masking=False, scale_position="post_adam",
                 actor_optimizer="adam", actor_learning_rate=None):
        #   actor_optimizer: "adam" (default, unchanged behaviour) or "sgd" (plain SGD, no momentum),
        #   for the actor only; critics and temperature always use Adam. The pipeline stays
        #   clip_by_global_norm -> optimizer -> x post-optimizer scales. actor_learning_rate:
        #   None = learning_rate.
        #   forward_masking (mask-decay method, Nevo structure): the actor forward pass of task t
        #   is multiplied layer-wise by that task's binary neuron mask (set_task_mask), so each
        #   task's policy depends on its own sub-network only; old tasks keep their masks for
        #   evaluation. Only importance_mode magnitude/external and anchor_mode none are allowed.
        #   importance_mode "external" (mask-decay method): the post-Adam multipliers are supplied
        #   by the caller through set_scales() and stay fixed within a task; no in-task refresh,
        #   no task-end importance measurement. reset_all_optimizers: at every task boundary keep
        #   every parameter (actor, critics, targets, temperature) but re-initialise every optimizer.
        #   anchor_target (kl mode): "full" = Gaussian KL on (mu, sigma); "mean" = only the old
        #   mean action is anchored (weighted by 1/sigma_old^2), the current sigma stays free so
        #   the new task keeps its exploration noise.
        #   log_temperature_min: floor on log alpha after each temperature step (None = no floor),
        #   so alpha cannot collapse before the new task's reward has been discovered.
        #   task_code_scale: multiplier on the task one-hot fed to actor and critics (1 = frozen).
        #   condition_ids: tuple mapping each task position to a condition index; positions that
        #   share a condition (a revisited environment) share the one-hot the actor/critics see.
        #   Default: identity (one condition per position, the frozen behaviour).
        #   anchor_budget: total number of stored anchor states across all conditions (0 =
        #   unbounded). When exceeded, every condition keeps an equal random share.
        #   Revisit rule: while a condition is being trained its own anchors are excluded from the
        #   anchor batches, and finish_task replaces that condition's anchors with fresh ones.
        #   actor_lr_ramp_updates: for tasks > 0, after any critic warm-up, the post-Adam actor
        #   increment is additionally multiplied by min(1, k / N) for the k-th actor update of the
        #   task, so a freshly reset Adam cannot take full-size sign-like steps on every weight.
        # v2 protection options (E1/E2):
        #   importance_mode: "magnitude" (frozen behaviour: per-kernel TopK counts, lambda**N) |
        #   "mas" (task-end squared per-sample gradients of ||tanh(mu)||^2) | "fisher" (task-end
        #   squared per-sample score of log pi). For mas/fisher the importance Omega is normalized
        #   by its global 99th percentile, accumulated over tasks (max or sum) over EVERY actor
        #   parameter (biases and heads included), and the post-Adam multiplier becomes
        #   s_i = max(scale_floor, 1 / (1 + importance_beta * Omega_i)); no in-task refresh.
        #   anchor_mode: "none" | "kl" (function-space anchor: KL from stored old-task policy
        #   (mu, sigma) on stored old-task states, added to the actor loss) | "ewc" (parameter
        #   anchor: importance-weighted mean squared drift from the previous task-end parameters).
        # v2 protocol options (E0 ablation):
        #   temperature_mode: "shared_reset" (frozen behaviour: one log alpha, reset to
        #   init_temperature at every task start), "shared_keep" (one log alpha, never reset),
        #   "per_task" (a log alpha per task id, Continual World style; each starts at init).
        #   critic_warmup_updates: for tasks > 0, the first N updates train only the critics;
        #   the actor and temperature are frozen so the actor never follows a random critic.
        if type(num_tasks) is not int or num_tasks <= 0:
            raise ValueError("num_tasks must be a positive integer")
        if not 0.0 < float(decay_factor) <= 1.0:
            raise ValueError("decay_factor must be in (0,1], with 1 an Adam control")
        if not 0.0 < float(mask_fraction) <= 1.0:
            raise ValueError("mask_fraction must be in (0,1]")
        if type(refresh_interval) is not int or refresh_interval <= 0:
            raise ValueError("refresh_interval must be a positive integer")
        if type(target_update_period) is not int or target_update_period <= 0:
            raise ValueError("target_update_period must be a positive integer")
        if not 0.0 <= float(discount) <= 1.0 or not 0.0 < float(tau) <= 1.0:
            raise ValueError("discount/tau are outside their valid ranges")
        for value in (learning_rate, max_grad_norm, final_fc_init_scale, clip_mean):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError("learning rate, norm and initializer/mean scales must be finite and positive")
        if not all(math.isfinite(float(value)) for value in (init_temperature, log_std_min, log_std_max)):
            raise ValueError("temperature and standard-deviation bounds must be finite")
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be below log_std_max")
        if actor_variant not in ("legacy", "stable"):
            raise ValueError("actor_variant must be legacy or stable")
        if actor_variant == "stable":
            _stable_std_constants(log_std_min, log_std_max)
        if temperature_mode not in ("shared_reset", "shared_keep", "per_task"):
            raise ValueError("temperature_mode must be shared_reset, shared_keep or per_task")
        if type(critic_warmup_updates) is not int or critic_warmup_updates < 0:
            raise ValueError("critic_warmup_updates must be a nonnegative integer")
        # The temperature reset flag follows the mode; per-task entries are never reset.
        reset_temperature = temperature_mode == "shared_reset"
        if importance_mode not in ("magnitude", "mas", "fisher", "external"):
            raise ValueError("importance_mode must be magnitude, mas, fisher or external")
        if importance_accumulate not in ("max", "sum"):
            raise ValueError("importance_accumulate must be max or sum")
        if anchor_mode not in ("none", "kl", "ewc"):
            raise ValueError("anchor_mode must be none, kl or ewc")
        if not 0.0 < float(scale_floor) <= 1.0 or float(importance_beta) < 0 or float(anchor_coef) < 0:
            raise ValueError("scale_floor in (0,1], importance_beta >= 0, anchor_coef >= 0")
        for value in (importance_samples, anchor_states_per_task, anchor_batch):
            if type(value) is not int or value <= 0:
                raise ValueError("importance/anchor sample counts must be positive integers")
        if type(actor_lr_ramp_updates) is not int or actor_lr_ramp_updates < 0:
            raise ValueError("actor_lr_ramp_updates must be a nonnegative integer")
        if condition_ids is None:
            condition_ids = tuple(range(num_tasks))
        condition_ids = tuple(int(c) for c in condition_ids)
        if len(condition_ids) != num_tasks or min(condition_ids) < 0:
            raise ValueError("condition_ids must give one nonnegative condition per task position")
        num_conditions = max(condition_ids) + 1
        if sorted(set(condition_ids)) != list(range(num_conditions)):
            raise ValueError("condition_ids must use consecutive indices starting at 0")
        if type(anchor_budget) is not int or anchor_budget < 0:
            raise ValueError("anchor_budget must be a nonnegative integer (0 = unbounded)")
        if anchor_target not in ("full", "mean"):
            raise ValueError("anchor_target must be full or mean")
        if log_temperature_min is not None and not math.isfinite(float(log_temperature_min)):
            raise ValueError("log_temperature_min must be finite or None")
        if not math.isfinite(float(task_code_scale)) or float(task_code_scale) <= 0:
            raise ValueError("task_code_scale must be finite and positive")
        actor_hidden_dims = tuple(actor_hidden_dims)
        critic_hidden_dims = tuple(critic_hidden_dims)
        if not actor_hidden_dims or not critic_hidden_dims or any(
                type(width) is not int or width <= 0 for width in actor_hidden_dims + critic_hidden_dims):
            raise ValueError("actor/critic hidden widths must be positive integers")
        forward_masking = bool(forward_masking)
        if forward_masking:
            if len(set(actor_hidden_dims)) != 1:
                raise ValueError("forward_masking needs uniform actor hidden widths")
            if anchor_mode not in ("none", "kl") or importance_mode in ("mas", "fisher"):
                raise ValueError("forward_masking supports only anchor_mode none/kl and importance_mode magnitude/external")
        if actor_optimizer not in ("adam", "sgd"):
            raise ValueError("actor_optimizer must be adam or sgd")
        actor_learning_rate = float(learning_rate if actor_learning_rate is None else actor_learning_rate)
        if not math.isfinite(actor_learning_rate) or actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be finite and positive")
        observations = jnp.asarray(observations, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if observations.ndim < 1 or actions.ndim < 1:
            raise ValueError("observations/actions need a final feature dimension")
        self.observation_dim = observations.shape[-1]
        self.action_dim = actions.shape[-1]
        self.num_tasks = num_tasks
        self.num_conditions = num_conditions
        self.condition_ids = condition_ids
        self.condition_table = jnp.asarray(condition_ids, dtype=jnp.int32)
        self.target_entropy = -self.action_dim / 2.0 if target_entropy is None else float(target_entropy)
        if not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        self.config = {
            "observation_dim": self.observation_dim, "action_dim": self.action_dim,
            "num_tasks": num_tasks, "actor_hidden_dims": list(actor_hidden_dims),
            "critic_hidden_dims": list(critic_hidden_dims), "learning_rate": float(learning_rate),
            "decay_factor": float(decay_factor), "mask_fraction": float(mask_fraction),
            "refresh_interval": refresh_interval, "reset_critic": bool(reset_critic),
            "reset_actor_optimizer": bool(reset_actor_optimizer), "reset_temperature": bool(reset_temperature),
            "discount": float(discount), "tau": float(tau), "target_update_period": target_update_period,
            "target_entropy": self.target_entropy, "init_temperature": float(init_temperature),
            "init_temperature_semantics": "initial log alpha; alpha=exp(init_temperature)",
            "max_grad_norm": float(max_grad_norm), "final_fc_init_scale": float(final_fc_init_scale),
            "clip_mean": float(clip_mean), "log_std_min": float(log_std_min), "log_std_max": float(log_std_max),
            "task_condition": "raw observations concatenated with fixed num_tasks one-hot for actor and critics",
            "count_rule": "one increment on first selection per task, including task 0; permanent lambda**N",
            "task_start_mask": "magnitude of shared parameters before current-task training data",
            "initializer": "variance_scaling(scale, fan_in, truncated_normal)",
            "actor_variant": actor_variant,
            "temperature_mode": temperature_mode,
            "critic_warmup_updates": critic_warmup_updates,
            "importance_mode": importance_mode, "importance_beta": float(importance_beta),
            "scale_floor": float(scale_floor), "importance_samples": importance_samples,
            "importance_accumulate": importance_accumulate,
            "anchor_mode": anchor_mode, "anchor_coef": float(anchor_coef),
            "anchor_states_per_task": anchor_states_per_task, "anchor_batch": anchor_batch,
            "actor_lr_ramp_updates": actor_lr_ramp_updates,
            "condition_ids": list(condition_ids), "num_conditions": num_conditions,
            "anchor_budget": anchor_budget,
            "anchor_target": anchor_target,
            "log_temperature_min": None if log_temperature_min is None else float(log_temperature_min),
            "task_code_scale": float(task_code_scale),
            "reset_all_optimizers": bool(reset_all_optimizers),
            "forward_masking": forward_masking,
            "scale_position": scale_position,
            "actor_optimizer": actor_optimizer,
            "actor_learning_rate": actor_learning_rate,
        }
        # Per-task neuron masks for forward masking (host-managed, saved with state_dict; all ones
        # until set_task_mask). Kept outside the jitted state so older checkpoints still load.
        self.neuron_mask_table = jnp.ones((num_tasks, len(actor_hidden_dims), actor_hidden_dims[0]), jnp.float32)
        self.actor = FullPolicy(actor_hidden_dims, self.action_dim, final_fc_init_scale,
                                clip_mean, log_std_min, log_std_max, actor_variant)
        self.critic = TwinCritic(critic_hidden_dims)
        actor_opt = (optax.adam(actor_learning_rate) if actor_optimizer == "adam"
                     else optax.sgd(actor_learning_rate, momentum=None))
        self.actor_tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), actor_opt)
        self.critic_tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate))
        self.temperature_tx = optax.adam(learning_rate)
        rng, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        self._dummy_obs = jnp.zeros((1, self.observation_dim + num_conditions), dtype=jnp.float32)
        self._dummy_actions = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        actor_params = freeze(self.actor.init(actor_key, self._dummy_obs)["params"])
        critic_params = freeze(self.critic.init(critic_key, self._dummy_obs, self._dummy_actions)["params"])
        counts = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.int32), actor_params)
        seen = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.bool_), actor_params)
        if temperature_mode == "per_task":
            log_temperature = jnp.full((num_tasks,), init_temperature, dtype=jnp.float32)
        else:
            log_temperature = jnp.asarray(init_temperature, dtype=jnp.float32)
        self._state = _LearnerState(
            rng=rng, actor_params=actor_params, actor_opt_state=self.actor_tx.init(actor_params),
            critic_params=critic_params, critic_opt_state=self.critic_tx.init(critic_params),
            target_critic_params=critic_params, log_temperature=log_temperature,
            temperature_opt_state=self.temperature_tx.init(log_temperature), counts=counts,
            seen=seen, current_mask=seen, scales=jax.tree_util.tree_map(jnp.ones_like, actor_params),
            importance=jax.tree_util.tree_map(jnp.zeros_like, actor_params),
            anchor_params=actor_params)
        self.current_task = None
        self.total_updates = 0
        self.task_updates = 0
        self.last_refresh_update = 0
        # Host-side function-space anchor set (kl mode): old-task states with the old policy's
        # pre-tanh Gaussian (mu, sigma) under that task's one-hot. Grows by anchor_states_per_task
        # per finished task; sampled with its own RNG so the frozen RNG streams are untouched.
        self.anchor_obs = np.zeros((0, self.observation_dim), np.float32)
        self.anchor_cond = np.zeros((0,), np.int32)
        self.anchor_mu = np.zeros((0, self.action_dim), np.float32)
        self.anchor_sigma = np.zeros((0, self.action_dim), np.float32)
        self.anchor_rng = np.random.RandomState(seed + 7919)
        self.finished_tasks = 0
        self._anchor_pool = np.zeros((0,), np.int64)

    def _cond(self, observations, task_id):
        """Concatenate the condition one-hot of a task position (shared across revisits)."""
        code = jax.nn.one_hot(self.condition_table[task_id], self.num_conditions, dtype=observations.dtype)
        code = code * self.config["task_code_scale"]
        code = jnp.broadcast_to(code, observations.shape[:-1] + (self.num_conditions,))
        return jnp.concatenate((observations, code), axis=-1)

    def current_condition(self):
        return None if self.current_task is None else self.condition_ids[self.current_task]

    def _rebuild_anchor_pool(self):
        if self.current_task is None:
            self._anchor_pool = np.zeros((0,), np.int64)
        else:
            self._anchor_pool = np.nonzero(self.anchor_cond != self.current_condition())[0]

    @property
    def actor_updates(self):
        return self.total_updates

    @property
    def task_actor_updates(self):
        return self.task_updates

    def _check_task(self, task_id):
        if isinstance(task_id, np.integer):
            task_id = int(task_id)
        if type(task_id) is not int or not 0 <= task_id < self.num_tasks:
            raise ValueError("task_id must be an integer in [0,num_tasks)")
        return task_id

    def begin_task(self, task_id):
        task_id = self._check_task(task_id)
        if self.current_task is not None and task_id < self.current_task:
            raise ValueError("begin_task cannot move backwards; specify historical IDs only for sampling")
        if task_id == self.current_task:
            return  # In-progress checkpoint resume must not clear saved seen.
        state = self._state
        if self.current_task is not None:
            if self.config["reset_actor_optimizer"]:
                state = state.replace(actor_opt_state=self.actor_tx.init(state.actor_params))
            if self.config["reset_critic"]:
                rng, key = jax.random.split(state.rng)
                params = freeze(self.critic.init(key, self._dummy_obs, self._dummy_actions)["params"])
                state = state.replace(rng=rng, critic_params=params, target_critic_params=params,
                                      critic_opt_state=self.critic_tx.init(params))
            if self.config["reset_temperature"]:
                log_temperature = jnp.asarray(self.config["init_temperature"], dtype=jnp.float32)
                state = state.replace(log_temperature=log_temperature,
                                      temperature_opt_state=self.temperature_tx.init(log_temperature))
            if self.config["reset_all_optimizers"]:
                # Mask-decay boundary protocol: inherit every parameter, restart every optimizer.
                state = state.replace(
                    actor_opt_state=self.actor_tx.init(state.actor_params),
                    critic_opt_state=self.critic_tx.init(state.critic_params),
                    temperature_opt_state=self.temperature_tx.init(state.log_temperature))
        self.current_task = task_id
        self.task_updates = 0
        self.last_refresh_update = 0
        self._state = state.replace(seen=jax.tree_util.tree_map(jnp.zeros_like, state.seen))
        self._rebuild_anchor_pool()
        self.refresh_mask()

    def refresh_mask(self):
        if self.current_task is None:
            raise ValueError("begin_task must precede mask refresh")
        if self.config["importance_mode"] != "magnitude":
            # Importance modes measure protection once per task in finish_task; nothing to refresh.
            self.last_refresh_update = self.task_updates
            return
        counts, seen, mask, scales = _refresh_counted_mask(
            self._state.actor_params, self._state.counts, self._state.seen,
            self.config["mask_fraction"], self.config["decay_factor"])
        self._state = self._state.replace(counts=counts, seen=seen, current_mask=mask, scales=scales)
        self.last_refresh_update = self.task_updates

    def set_scales(self, host_scales):
        """External mode: install per-coordinate post-Adam multipliers (same tree as the actor params).

        Values must be finite and in (0, 1]. They stay fixed until the next call; begin_task does
        not touch them, so the caller sets them once per task after begin_task.
        """
        if self.config["importance_mode"] != "external":
            raise ValueError("set_scales is only valid in importance_mode='external'")
        scales = freeze(jax.tree_util.tree_map(lambda s: jnp.asarray(s, dtype=jnp.float32), unfreeze(host_scales)))
        expected = jax.tree_util.tree_structure(self._state.actor_params)
        if jax.tree_util.tree_structure(scales) != expected:
            raise ValueError("scale tree structure does not match the actor parameters")
        for s, p in zip(jax.tree_util.tree_leaves(scales), jax.tree_util.tree_leaves(self._state.actor_params)):
            if s.shape != p.shape:
                raise ValueError("scale leaf shape does not match the actor parameter")
            host = np.asarray(s)
            if not np.all(np.isfinite(host)) or np.any(host <= 0.0) or np.any(host > 1.0):
                raise ValueError("scales must be finite and in (0, 1]")
        self._state = self._state.replace(scales=scales)

    def _masks_for(self, task_id):
        """Neuron masks of a task position for forward masking (None when masking is off)."""
        if not self.config["forward_masking"]:
            return None
        return self.neuron_mask_table[int(task_id)]

    def set_task_mask(self, task_id, masks):
        """Install the binary neuron masks (one vector per hidden layer) of a task position."""
        if not self.config["forward_masking"]:
            raise ValueError("set_task_mask requires forward_masking=True")
        task_id = self._check_task(task_id)
        masks = np.asarray(masks, dtype=np.float32)
        if masks.shape != self.neuron_mask_table.shape[1:]:
            raise ValueError("neuron masks must have shape [num_hidden_layers, width]")
        if not np.all((masks == 0) | (masks == 1)) or np.any(masks.sum(axis=1) == 0):
            raise ValueError("neuron masks must be binary with at least one active neuron per layer")
        self.neuron_mask_table = self.neuron_mask_table.at[task_id].set(jnp.asarray(masks))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _sample_actions_jit(self, rng, params, observations, task_id, temperature, neuron_masks):
        rng, key = jax.random.split(rng)
        conditioned = self._cond(observations, task_id)
        mean, std = self.actor.apply({"params": params}, conditioned, neuron_masks=neuron_masks)
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        return rng, jnp.tanh(mean + temperature * std * noise)

    def sample_actions(self, observations, task_id=None, temperature=1.0):
        task_id = self.current_task if task_id is None else task_id
        task_id = self._check_task(task_id)
        if not math.isfinite(float(temperature)) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.ndim < 1 or observations.shape[-1] != self.observation_dim:
            raise ValueError("sample_actions expects raw observations with the configured feature dimension")
        rng, actions = self._sample_actions_jit(self._state.rng, self._state.actor_params,
                                               observations, jnp.asarray(task_id),
                                               jnp.asarray(temperature, dtype=jnp.float32),
                                               self._masks_for(task_id))
        self._state = self._state.replace(rng=rng)
        return np.asarray(jax.device_get(actions))

    def _select_log_alpha(self, log_temperature, task_id):
        if self.config["temperature_mode"] == "per_task":
            return log_temperature[task_id]
        return log_temperature

    # ------------------------------------------------------------------
    # E1: task-end importance (mas / fisher) and smooth per-parameter multipliers
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=(0,))
    def _importance_chunk_jit(self, params, observations, task_id, key):
        """Sum over the chunk of squared per-sample gradients for every actor parameter."""
        conditioned = self._cond(observations, task_id)
        if self.config["importance_mode"] == "mas":
            def objective(p, single):
                mean, _ = self.actor.apply({"params": p}, single[None])
                return jnp.sum(jnp.tanh(mean) ** 2)
            grads = jax.vmap(lambda single: jax.grad(objective)(params, single))(conditioned)
        else:
            keys = jax.random.split(key, observations.shape[0])
            def objective(p, single, k):
                _, log_prob = _sample_and_log_prob(self.actor, p, single[None], k)
                return jnp.sum(log_prob)
            grads = jax.vmap(lambda single, k: jax.grad(objective)(params, single, k))(conditioned, keys)
        return jax.tree_util.tree_map(lambda g: jnp.sum(g ** 2, axis=0), grads)

    def _importance_scales(self, importance):
        beta, floor = self.config["importance_beta"], self.config["scale_floor"]
        return jax.tree_util.tree_map(lambda w: jnp.maximum(floor, 1.0 / (1.0 + beta * w)), importance)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _anchor_targets_jit(self, params, observations, task_id, neuron_masks=None):
        conditioned = self._cond(observations, task_id)
        return self.actor.apply({"params": params}, conditioned, neuron_masks=neuron_masks)

    def finish_task(self, observations):
        """Call once at the end of the current task with raw observations sampled from its replay.

        Updates accumulated importance and multipliers (mas/fisher) and extends the anchor set
        (kl) or anchor parameters (ewc). The actor itself is not modified.
        """
        if self.current_task is None:
            raise ValueError("begin_task must precede finish_task")
        if self.config["importance_mode"] in ("magnitude", "external") and self.config["anchor_mode"] == "none":
            self.finished_tasks = self.current_task + 1
            return {}
        observations = np.asarray(observations, np.float32)
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim or observations.shape[0] == 0:
            raise ValueError("finish_task expects [N, raw_observation_dim] observations")
        task_id = jnp.asarray(self.current_task)
        info = {}
        state = self._state
        if self.config["importance_mode"] in ("mas", "fisher"):
            n = min(self.config["importance_samples"], observations.shape[0])
            chunk = 32
            total = jax.tree_util.tree_map(jnp.zeros_like, state.actor_params)
            rng = state.rng
            for start in range(0, n, chunk):
                rng, key = jax.random.split(rng)
                part = self._importance_chunk_jit(state.actor_params, jnp.asarray(observations[start:start + chunk]),
                                                  task_id, key)
                total = jax.tree_util.tree_map(jnp.add, total, part)
            omega = jax.tree_util.tree_map(lambda t: t / n, total)
            flat = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(omega)])
            reference = jnp.maximum(jnp.percentile(flat, 99.0), 1e-12)
            normalized = jax.tree_util.tree_map(lambda w: jnp.clip(w / reference, 0.0, 1.0), omega)
            if self.config["importance_accumulate"] == "max":
                importance = jax.tree_util.tree_map(jnp.maximum, state.importance, normalized)
            else:
                importance = jax.tree_util.tree_map(jnp.add, state.importance, normalized)
            scales = self._importance_scales(importance)
            state = state.replace(rng=rng, importance=importance, scales=scales)
            info.update(importance_reference_q99=float(reference), importance_samples=int(n))
        if self.config["anchor_mode"] == "kl":
            cond = self.current_condition()
            m = min(self.config["anchor_states_per_task"], observations.shape[0])
            chosen = observations[self.anchor_rng.choice(observations.shape[0], m, replace=False)]
            mu, sigma = self._anchor_targets_jit(state.actor_params, jnp.asarray(chosen), task_id,
                                                 self._masks_for(self.current_task))
            # Revisit rule: a condition's anchors describe its latest policy only.
            keep = self.anchor_cond != cond
            replaced = int(np.sum(~keep))
            self.anchor_obs = np.concatenate((self.anchor_obs[keep], chosen), axis=0)
            self.anchor_cond = np.concatenate((self.anchor_cond[keep], np.full((m,), cond, np.int32)))
            self.anchor_mu = np.concatenate((self.anchor_mu[keep], np.asarray(jax.device_get(mu), np.float32)), axis=0)
            self.anchor_sigma = np.concatenate((self.anchor_sigma[keep], np.asarray(jax.device_get(sigma), np.float32)), axis=0)
            budget = self.config["anchor_budget"]
            if budget > 0 and self.anchor_obs.shape[0] > budget:
                # Fixed total memory: every stored condition keeps an equal random share.
                conditions = np.unique(self.anchor_cond)
                share = max(1, budget // len(conditions))
                kept = []
                for value in conditions:
                    members = np.nonzero(self.anchor_cond == value)[0]
                    if len(members) > share:
                        members = self.anchor_rng.choice(members, share, replace=False)
                    kept.append(members)
                kept = np.sort(np.concatenate(kept))
                self.anchor_obs, self.anchor_cond = self.anchor_obs[kept], self.anchor_cond[kept]
                self.anchor_mu, self.anchor_sigma = self.anchor_mu[kept], self.anchor_sigma[kept]
            self._rebuild_anchor_pool()
            info.update(anchor_states_total=int(self.anchor_obs.shape[0]), anchor_states_replaced=replaced,
                        anchor_conditions=int(len(np.unique(self.anchor_cond))))
        if self.config["anchor_mode"] == "ewc":
            state = state.replace(anchor_params=state.actor_params)
        self._state = state
        self.finished_tasks = self.current_task + 1
        return info

    def _ewc_weights(self, state):
        if self.config["importance_mode"] in ("magnitude", "external"):
            return jax.tree_util.tree_map(lambda c: c.astype(jnp.float32), state.counts)
        return state.importance

    def _sample_anchor_batch(self):
        if self.config["anchor_mode"] != "kl" or self._anchor_pool.shape[0] == 0:
            return None
        index = self._anchor_pool[self.anchor_rng.randint(self._anchor_pool.shape[0], size=self.config["anchor_batch"])]
        return (jnp.asarray(self.anchor_obs[index]), jnp.asarray(self.anchor_cond[index]),
                jnp.asarray(self.anchor_mu[index]), jnp.asarray(self.anchor_sigma[index]))

    @property
    def in_critic_warmup(self):
        return (self.current_task is not None and self.current_task > 0
                and self.task_updates < self.config["critic_warmup_updates"])

    def actor_lr_scale(self):
        """Ramp multiplier for the current actor update (1.0 on task 0 or when disabled)."""
        ramp = self.config["actor_lr_ramp_updates"]
        if ramp == 0 or self.current_task is None or self.current_task == 0:
            return 1.0
        k = self.task_updates - self.config["critic_warmup_updates"] + 1
        return float(min(1.0, max(0.0, k / ramp)))

    @functools.partial(jax.jit, static_argnums=(0, 5, 7))
    def _update_jit(self, state, batch, task_id, update_number, train_actor, anchor_batch, use_ewc, lr_scale,
                    neuron_masks=None, anchor_masks=None):
        observations, actions, rewards, masks, next_observations = batch
        observations = self._cond(observations, task_id)
        next_observations = self._cond(next_observations, task_id)
        rng, target_key, actor_key = jax.random.split(state.rng, 3)
        alpha = jnp.exp(self._select_log_alpha(state.log_temperature, task_id))
        next_actions, next_log_prob = _sample_and_log_prob(
            self.actor, state.actor_params, next_observations, target_key, neuron_masks=neuron_masks)
        next_q = self.critic.apply({"params": state.target_critic_params}, next_observations, next_actions)
        target_q = jax.lax.stop_gradient(rewards + self.config["discount"] * masks * (
            jnp.min(next_q, axis=0) - alpha * next_log_prob))

        def critic_loss_fn(params):
            qs = self.critic.apply({"params": params}, observations, actions)
            loss = jnp.mean(jnp.sum((qs - target_q[None, :]) ** 2, axis=0))
            return loss, (jnp.mean(qs[0]), jnp.mean(qs[1]))

        (critic_loss, (q1, q2)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(state.critic_params)
        critic_updates, critic_opt_state = self.critic_tx.update(
            critic_grads, state.critic_opt_state, state.critic_params)
        critic_params = optax.apply_updates(state.critic_params, critic_updates)
        target_critic_params = jax.lax.cond(
            update_number % self.config["target_update_period"] == 0,
            lambda _: jax.tree_util.tree_map(lambda q, target: self.config["tau"] * q + (
                1.0 - self.config["tau"]) * target, critic_params, state.target_critic_params),
            lambda _: state.target_critic_params, operand=None)

        def anchor_penalty(params):
            # E2 anchors. kl: Gaussian KL(old || current) in pre-tanh space on stored old-task states.
            # ewc: importance-weighted mean squared drift from the previous task-end parameters.
            penalty = jnp.asarray(0.0, jnp.float32)
            if anchor_batch is not None:
                anchor_obs, anchor_cond, mu_old, sigma_old = anchor_batch
                code = jax.nn.one_hot(anchor_cond, self.num_conditions, dtype=anchor_obs.dtype)
                code = code * self.config["task_code_scale"]
                # With forward masking every anchor state is evaluated under ITS task's neuron masks.
                mu_new, sigma_new = self.actor.apply({"params": params}, jnp.concatenate((anchor_obs, code), axis=-1),
                                                     neuron_masks=anchor_masks)
                if self.config["anchor_target"] == "mean":
                    # Mean-only anchor: KL with sigma_new held at sigma_old, i.e. only the mean
                    # action is pinned; the current standard deviation is unconstrained.
                    kl = (mu_old - mu_new) ** 2 / (2.0 * sigma_old ** 2)
                else:
                    kl = (jnp.log(sigma_new / sigma_old) + (sigma_old ** 2 + (mu_old - mu_new) ** 2)
                          / (2.0 * sigma_new ** 2) - 0.5)
                penalty = penalty + jnp.mean(jnp.sum(kl, axis=-1))
            if use_ewc:
                weights = self._ewc_weights(state)
                drift = jax.tree_util.tree_map(lambda p, a, w: jnp.sum(w * (p - a) ** 2),
                                               params, state.anchor_params, weights)
                mass = sum(jnp.sum(w) for w in jax.tree_util.tree_leaves(weights))
                penalty = penalty + sum(jax.tree_util.tree_leaves(drift)) / jnp.maximum(mass, 1e-12)
            return penalty

        def actor_loss_fn(params):
            sampled_actions, log_prob, head_before = _sample_and_log_prob(
                self.actor, params, observations, actor_key, return_diagnostics=True, neuron_masks=neuron_masks)
            qs = self.critic.apply({"params": critic_params}, observations, sampled_actions)
            sac_loss = jnp.mean(alpha * log_prob - jnp.min(qs, axis=0))
            penalty = anchor_penalty(params)
            return sac_loss + self.config["anchor_coef"] * penalty, (-jnp.mean(log_prob), head_before, penalty)

        (actor_loss, (entropy, head_before, anchor_loss)), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True)(state.actor_params)
        # These metrics refer to this loss's complete, pre-update gradients,
        # before clipping, Adam or lambda**N. Count/max do not square values.
        actor_grad_max_abs, actor_grad_nonzero_count = _gradient_coordinate_stats(actor_grads)
        # Complete gradients -> global clip -> Adam -> permanent coordinate
        # multiplier. Adam's returned moments are retained WITHOUT scaling.
        if train_actor and self.config["scale_position"] == "pre_adam":
            # Control arm only: multiply the GRADIENT by the scale before clip+Adam (Adam's moment
            # normalization is expected to cancel a constant per-coordinate gradient scaling).
            pre = jax.tree_util.tree_map(lambda g, scale: g * scale, actor_grads, state.scales)
            raw_updates, actor_opt_state = self.actor_tx.update(pre, state.actor_opt_state, state.actor_params)
            scaled_updates = jax.tree_util.tree_map(lambda update: update * lr_scale, raw_updates)
            actor_params = optax.apply_updates(state.actor_params, scaled_updates)
        elif train_actor:
            raw_updates, actor_opt_state = self.actor_tx.update(
                actor_grads, state.actor_opt_state, state.actor_params)
            scaled_updates = jax.tree_util.tree_map(lambda update, scale: update * scale * lr_scale,
                                                   raw_updates, state.scales)
            actor_params = optax.apply_updates(state.actor_params, scaled_updates)
        else:
            # Critic warm-up: gradients are still measured for monitoring, but neither the
            # actor parameters nor the actor Adam state move.
            raw_updates = jax.tree_util.tree_map(jnp.zeros_like, actor_grads)
            scaled_updates = raw_updates
            actor_opt_state = state.actor_opt_state
            actor_params = state.actor_params

        def temperature_loss_fn(log_temperature):
            return jnp.exp(self._select_log_alpha(log_temperature, task_id)) * jax.lax.stop_gradient(
                entropy - self.target_entropy)

        temperature_loss, temperature_grad = jax.value_and_grad(temperature_loss_fn)(state.log_temperature)
        if train_actor:
            temperature_update, temperature_opt_state = self.temperature_tx.update(
                temperature_grad, state.temperature_opt_state, state.log_temperature)
            if self.config["temperature_mode"] == "per_task":
                # Only the current task's entry may move; stale Adam momentum on other entries is masked.
                temperature_update = temperature_update * jax.nn.one_hot(
                    task_id, self.num_tasks, dtype=temperature_update.dtype)
            log_temperature = optax.apply_updates(state.log_temperature, temperature_update)
            if self.config["log_temperature_min"] is not None:
                log_temperature = jnp.maximum(log_temperature, self.config["log_temperature_min"])
        else:
            temperature_opt_state = state.temperature_opt_state
            log_temperature = state.log_temperature
        raw_norm = optax.global_norm(raw_updates)
        scaled_norm = optax.global_norm(scaled_updates)
        actor_grad_norm = optax.global_norm(actor_grads)
        critic_grad_norm = optax.global_norm(critic_grads)
        finite = jnp.all(jnp.isfinite(jnp.stack((actor_loss, critic_loss, temperature_loss,
            alpha, entropy, actor_grad_norm, critic_grad_norm, raw_norm, scaled_norm,
            actor_grad_max_abs) + tuple(head_before.values()))))
        next_state = state.replace(rng=rng, actor_params=actor_params, actor_opt_state=actor_opt_state,
            critic_params=critic_params, critic_opt_state=critic_opt_state, target_critic_params=target_critic_params,
            log_temperature=log_temperature, temperature_opt_state=temperature_opt_state)
        return next_state, {
            "actor_loss": actor_loss, "critic_loss": critic_loss, "temperature": alpha,
            "temperature_loss": temperature_loss, "entropy": entropy, "q1": q1, "q2": q2,
            "actor_grad_norm": actor_grad_norm, "critic_grad_norm": critic_grad_norm,
            "actor_grad_max_abs": actor_grad_max_abs,
            "actor_grad_nonzero_count": actor_grad_nonzero_count,
            "actor_raw_update_norm": raw_norm, "actor_scaled_update_norm": scaled_norm,
            "actor_update_norm_ratio": scaled_norm / jnp.maximum(raw_norm, 1e-30),
            "nonfinite": jnp.logical_not(finite).astype(jnp.float32),
            "actor_trained": jnp.asarray(1.0 if train_actor else 0.0, dtype=jnp.float32),
            "anchor_loss": anchor_loss,
            "actor_lr_scale": jnp.asarray(lr_scale, dtype=jnp.float32),
            **{"actor_" + key + "_before": value for key, value in head_before.items()},
        }

    def update(self, batch):
        if self.current_task is None:
            raise ValueError("begin_task must precede SAC updates")
        fields = ("observations", "actions", "rewards", "masks", "next_observations")
        arrays = tuple(jnp.asarray(batch[name] if isinstance(batch, dict) else getattr(batch, name),
                                   dtype=jnp.float32) for name in fields)
        observations, actions, rewards, masks, next_observations = arrays
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError("batch observations must have shape [batch, raw_observation_dim]")
        size = observations.shape[0]
        if actions.shape != (size, self.action_dim) or next_observations.shape != observations.shape:
            raise ValueError("batch action/next-observation shapes do not match")
        if rewards.shape not in ((size,), (size, 1)) or masks.shape not in ((size,), (size, 1)):
            raise ValueError("batch rewards/masks need one scalar per transition")
        if (self.task_updates > 0 and self.task_updates % self.config["refresh_interval"] == 0
                and self.last_refresh_update != self.task_updates):
            self.refresh_mask()
        arrays = (observations, actions, rewards.reshape(-1), masks.reshape(-1), next_observations)
        train_actor = not self.in_critic_warmup
        anchor_batch = self._sample_anchor_batch() if train_actor else None
        use_ewc = bool(self.config["anchor_mode"] == "ewc" and self.finished_tasks > 0 and train_actor)
        lr_scale = jnp.asarray(self.actor_lr_scale(), dtype=jnp.float32)
        anchor_masks = None
        if anchor_batch is not None and self.config["forward_masking"]:
            anchor_masks = self.neuron_mask_table[anchor_batch[1]]  # [batch, layers, width]
        self._state, info = self._update_jit(self._state, arrays, jnp.asarray(self.current_task),
                                             jnp.asarray(self.total_updates + 1, dtype=jnp.int32),
                                             train_actor, anchor_batch, use_ewc, lr_scale,
                                             self._masks_for(self.current_task), anchor_masks)
        self.total_updates += 1
        self.task_updates += 1
        return info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _diagnostic_stats_jit(self, state):
        counts = flatten_dict(unfreeze(state.counts))
        seen = flatten_dict(unfreeze(state.seen))
        masks = flatten_dict(unfreeze(state.current_mask))
        scales = flatten_dict(unfreeze(state.scales))
        paths = [path for path in counts if path[-1] == "kernel"]
        size = sum(counts[path].size for path in paths)
        count_mean = sum(jnp.sum(counts[path].astype(jnp.float32)) for path in paths) / size
        scale_mean = sum(jnp.sum(scales[path]) for path in paths) / size
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf))
                                  for leaf in jax.tree_util.tree_leaves(state)]))
        all_scales = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(state.scales)])
        all_importance = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(state.importance)])
        return {
            "scale_lt_0.1_fraction_all_params": jnp.mean(all_scales < 0.1),
            "scale_lt_0.5_fraction_all_params": jnp.mean(all_scales < 0.5),
            "scale_mean_all_params": jnp.mean(all_scales),
            "importance_mean_all_params": jnp.mean(all_importance),
            "importance_gt_0.1_fraction_all_params": jnp.mean(all_importance > 0.1),
            "count_min": jnp.min(jnp.stack([jnp.min(counts[path]) for path in paths])),
            "count_max": jnp.max(jnp.stack([jnp.max(counts[path]) for path in paths])),
            "count_mean": count_mean,
            "count_ge2_fraction": sum(jnp.sum(counts[path] >= 2) for path in paths) / size,
            "scale_min": jnp.min(jnp.stack([jnp.min(scales[path]) for path in paths])),
            "scale_mean": scale_mean,
            "scale_max": jnp.max(jnp.stack([jnp.max(scales[path]) for path in paths])),
            "mask_union_ratio": sum(jnp.sum(counts[path] > 0) for path in paths) / size,
            "task_seen_fraction": sum(jnp.sum(seen[path]) for path in paths) / size,
            "current_mask_ratio": sum(jnp.sum(masks[path]) for path in paths) / size,
            "all_finite": finite, "selected_kernel_parameters": jnp.asarray(size, dtype=jnp.int32),
        }

    def diagnostic_stats(self):
        """Compute full-tree reductions only when the caller requests a log."""
        values = jax.device_get(self._diagnostic_stats_jit(self._state))
        stats = {name: np.asarray(value).item() for name, value in values.items()}
        stats.update(actor_updates=self.total_updates, task_actor_updates=self.task_updates,
                     current_task=self.current_task, last_refresh_update=self.last_refresh_update,
                     anchor_states_total=int(self.anchor_obs.shape[0]), finished_tasks=self.finished_tasks,
                     anchor_pool_size=int(self._anchor_pool.shape[0]),
                     anchor_conditions=int(len(np.unique(self.anchor_cond))) if self.anchor_cond.size else 0)
        return stats

    def state_dict(self):
        """Host checkpoint including optimizer states and in-progress counting."""
        return {"format_version": 1, "config": copy.deepcopy(self.config),
                "current_task": self.current_task, "total_updates": self.total_updates,
                "task_updates": self.task_updates, "last_refresh_update": self.last_refresh_update,
                "learner_state": jax.device_get(serialization.to_state_dict(self._state)),
                "anchors": {"obs": self.anchor_obs.copy(), "cond": self.anchor_cond.copy(),
                            "mu": self.anchor_mu.copy(), "sigma": self.anchor_sigma.copy(),
                            "rng": self.anchor_rng.get_state(), "finished_tasks": self.finished_tasks},
                "neuron_masks": np.asarray(jax.device_get(self.neuron_mask_table), np.float32)}

    def load_state_dict(self, state):
        required = {"format_version", "config", "current_task", "total_updates", "task_updates",
                    "last_refresh_update", "learner_state"}
        if not isinstance(state, dict) or set(state) - {"anchors", "neuron_masks"} != required or state["format_version"] != 1:
            raise ValueError("invalid cumulative learner checkpoint format")
        checkpoint_config = copy.deepcopy(state["config"])
        # Pre-variant checkpoints have precisely the legacy actor and parameter
        # names. Normalize only this missing field; all other config checks stay.
        checkpoint_config.setdefault("actor_variant", "legacy")
        # Phase B flags were added after the E0-E3 / Phase A checkpoints were written; their
        # defaults reproduce the earlier behaviour, so older checkpoints normalize to them.
        checkpoint_config.setdefault("anchor_target", "full")
        checkpoint_config.setdefault("log_temperature_min", None)
        checkpoint_config.setdefault("task_code_scale", 1.0)
        checkpoint_config.setdefault("reset_all_optimizers", False)
        checkpoint_config.setdefault("forward_masking", False)
        checkpoint_config.setdefault("scale_position", "post_adam")
        checkpoint_config.setdefault("actor_optimizer", "adam")
        checkpoint_config.setdefault("actor_learning_rate", float(checkpoint_config["learning_rate"]))
        # A probe may load a post-Adam checkpoint into a pre-Adam learner (same state tree).
        if self.config.get("scale_position") != checkpoint_config.get("scale_position"):
            checkpoint_config["scale_position"] = self.config["scale_position"]
        if checkpoint_config != self.config:
            raise ValueError("checkpoint learner configuration does not match")
        current_task = state["current_task"]
        if current_task is not None:
            current_task = self._check_task(current_task)
        for name in ("total_updates", "task_updates", "last_refresh_update"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("checkpoint update counters must be nonnegative integers")
        if not state["last_refresh_update"] <= state["task_updates"] <= state["total_updates"]:
            raise ValueError("checkpoint update counters are inconsistent")
        restored = serialization.from_state_dict(self._state, state["learner_state"])
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(self._state)
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(restored)
        if expected_tree != actual_tree:
            raise ValueError("checkpoint state tree does not match the learner")
        for expected, actual in zip(expected_leaves, actual_leaves):
            if np.shape(expected) != np.shape(actual) or np.dtype(expected.dtype) != np.asarray(actual).dtype:
                raise ValueError("checkpoint array shape/dtype does not match")
            if not np.all(np.isfinite(np.asarray(actual))):
                raise ValueError("checkpoint contains nonfinite arrays")
        counts = flatten_dict(unfreeze(restored.counts))
        seen = flatten_dict(unfreeze(restored.seen))
        masks = flatten_dict(unfreeze(restored.current_mask))
        for path, count in counts.items():
            count, selected, mask = np.asarray(count), np.asarray(seen[path]), np.asarray(masks[path])
            if np.any(count < 0) or np.any(selected & (count == 0)) or np.any(mask & ~selected):
                raise ValueError("checkpoint counts/seen/current-mask are inconsistent")
            if path[-1] != "kernel" and (np.any(count) or np.any(selected) or np.any(mask)):
                raise ValueError("non-kernel parameters cannot have cumulative selections")
            if current_task is None and (np.any(count) or np.any(selected) or np.any(mask)):
                raise ValueError("unstarted checkpoint cannot contain selections")
            if current_task is not None and np.any(count > current_task + 1):
                raise ValueError("a coordinate cannot be counted more than once per task")
        restored = jax.tree_util.tree_map(jnp.asarray, restored)
        if self.config["importance_mode"] == "magnitude":
            expected_scales = _scale_tree(restored.actor_params, restored.counts, self.config["decay_factor"])
            message = "checkpoint scales do not equal lambda**counts"
        elif self.config["importance_mode"] in ("mas", "fisher"):
            expected_scales = self._importance_scales(restored.importance)
            message = "checkpoint scales do not equal max(floor, 1/(1+beta*importance))"
        else:
            # external: multipliers are caller-defined; only their range is checked.
            expected_scales, message = None, ""
            for leaf in jax.tree_util.tree_leaves(restored.scales):
                host = np.asarray(leaf)
                if np.any(host <= 0.0) or np.any(host > 1.0):
                    raise ValueError("checkpoint external scales must be in (0, 1]")
        if expected_scales is not None:
            for actual, expected in zip(jax.tree_util.tree_leaves(restored.scales),
                                        jax.tree_util.tree_leaves(expected_scales)):
                if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                    raise ValueError(message)
        anchors = state.get("anchors")
        if anchors is not None:
            if anchors["obs"].shape[1:] != (self.observation_dim,) or anchors["mu"].shape[1:] != (self.action_dim,):
                raise ValueError("checkpoint anchor arrays do not match the learner dimensions")
            self.anchor_obs = np.asarray(anchors["obs"], np.float32)
            self.anchor_cond = np.asarray(anchors.get("cond", anchors.get("task")), np.int32)
            self.anchor_mu = np.asarray(anchors["mu"], np.float32)
            self.anchor_sigma = np.asarray(anchors["sigma"], np.float32)
            self.anchor_rng.set_state(anchors["rng"])
            self.finished_tasks = int(anchors["finished_tasks"])
        neuron_masks = state.get("neuron_masks")
        if neuron_masks is not None:
            neuron_masks = np.asarray(neuron_masks, np.float32)
            if neuron_masks.shape != tuple(self.neuron_mask_table.shape):
                raise ValueError("checkpoint neuron mask table shape does not match the learner")
            if not np.all((neuron_masks == 0) | (neuron_masks == 1)):
                raise ValueError("checkpoint neuron masks must be binary")
            self.neuron_mask_table = jnp.asarray(neuron_masks)
        self._state = restored
        self.current_task = current_task
        self._rebuild_anchor_pool()
        self.total_updates = state["total_updates"]
        self.task_updates = state["task_updates"]
        self.last_refresh_update = state["last_refresh_update"]
