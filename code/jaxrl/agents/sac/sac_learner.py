"""Implementations of algorithms for continuous control."""

from copy import deepcopy
import functools
from typing import Optional, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from optax import global_norm
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from flax import linen as nn
from flax.core import freeze, unfreeze, FrozenDict
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import sparse_encode

import jaxrl.networks.common as utils_fn
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2


@jax.jit
def _update_sac_jit(
    rng: PRNGKey, actor: TrainState, critic: TrainState, target_critic: TrainState,
    temp: TrainState, batch: Batch, discount: float, tau: float,
    target_entropy: float
    ) -> Tuple[PRNGKey, TrainState, TrainState, TrainState, TrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount)

    new_target_critic = target_update(new_critic, target_critic, tau)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SACLearner(object):
    def __init__(self,
                 population_size,
                 use_critic_pop,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 pi_opt_configs: dict = {},
                 q_opt_configs: dict = {},
                 t_opt_configs: dict = {},
                 actor_configs: dict = {},
                 critic_configs: dict = {},
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        action_dim = actions.shape[-1]
        self.population_size = population_size
        self.use_critic_pop = use_critic_pop
        if target_entropy is None:
            self.target_entropy = -1.0 * action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        critic_key_pop = jax.random.split(critic_key, self.population_size)
        if self.use_critic_pop:
            critic_pop = []
            critic_target_pop = []
            for i in range(self.population_size):
                critic_def = critic_net.DoubleCritic(**critic_configs)
                _, critic_params = critic_def.init(
                    critic_key_pop[i], observations, actions
                ).pop('params')
                critic = TrainState.create(
                    apply_fn=critic_def.apply,
                    params=critic_params,
                    tx=utils_fn.set_optimizer(**q_opt_configs)
                )
                critic_pop.append(critic)

                tc_def = critic_net.DoubleCritic(**critic_configs)
                _, tc_params = tc_def.init(
                    critic_key_pop[i], observations, actions
                ).pop('params')
                target_critic = TrainState.create(
                    apply_fn=tc_def.apply,
                    params=tc_params,
                    tx=utils_fn.set_optimizer(**q_opt_configs)
                )
                critic_target_pop.append(target_critic)

        else:
            critic_def = critic_net.DoubleCritic(**critic_configs)
            _, critic_params = critic_def.init(
                critic_key, observations, actions
            ).pop('params')
            critic = TrainState.create(
                apply_fn=critic_def.apply,
                params=critic_params,
                tx=utils_fn.set_optimizer(**q_opt_configs))
            tc_def = critic_net.DoubleCritic(**critic_configs)
            _, tc_params = tc_def.init(
                critic_key, observations, actions
            ).pop('params')
            target_critic = TrainState.create(
                apply_fn=tc_def.apply,
                params=tc_params,
                tx=utils_fn.set_optimizer(**q_opt_configs)
            )


        self.temp_pop = []
        for i in range(self.population_size):
            temp_def = temperature.Temperature(init_temperature)
            _, temp_params = temp_def.init(temp_key).pop('params')
            temp = TrainState.create(
                apply_fn=temp_def.apply,
                params=temp_params,
                tx=utils_fn.set_optimizer(**t_opt_configs)
            )
            self.temp_pop.append(temp)

        if self.use_critic_pop:
            self.critic_pop = critic_pop
            self.target_critic_pop = critic_target_pop
        else:
            self.critic_pop = [critic]
            self.target_critic_pop = [target_critic]


        self.rng = rng

        # for reset models
        self.dummy_a = actions
        self.dummy_o = observations
        self.actor_cfgs = actor_configs
        self.critic_cfgs = critic_configs
        self.init_temp = init_temperature

        self.step = 0

        self.invalid_filter = jax.jit(
            lambda x:
            tree_map(
                lambda y: jnp.where(jnp.logical_or(jnp.isnan(y), jnp.isinf(y)), 0, y),
                x
            )
        )



@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: MPNTrainState,
    observations: np.ndarray,
    task_i: jnp.ndarray,
    temperature: float = 1.0
    ) -> Tuple[PRNGKey, jnp.ndarray, dict]:
    
    rng, key = jax.random.split(rng)
    dist, dicts = actor(
        observations,
        task_i,
        temperature
    )
    return rng, dist.sample(seed=key), dicts


def _update_alpha(
    actor: MPNTrainState, critic: TrainState, temp: TrainState,
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any],
    actor_params, critic_params, temp_params,
    batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(param) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': param}, batch.observations, jnp.array([task_id]))
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic.apply_fn({'params': critic_params['param']},batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp.apply_fn({'params': temp_params['param']}) - q).mean()

        _info = {
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        for k  in dicts['fau']:
            _info[k + '_fau'] = jnp.mean(dicts['fau'][k])

        return actor_loss, _info

    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor_params['param'])

    # recording info
    g_norm = global_norm(grads_actor)
    actor_info[ 'g_norm_actor'] = g_norm
    for p, v in param_mask.items():
        if p[-1] == 'kernel':
            actor_info[ 'used_capacity_'+p[0]] = 1.0 - jnp.mean(v)

    # only update coefficients (alpha)
    #new_actor = actor.apply_grads_alpha(grads=grads_actor)

    new_actor = actor.apply_grads_alpha(grads=grads_actor, params=actor_params['param'], step=actor_params['step'], opt_state_alpha =actor_params['opt_state_alpha'], opt_state_theta=actor_params['opt_state_theta'] )

    return rng, {'param': new_actor.params, 'step': new_actor.step,  'opt_state_theta': new_actor.opt_state_theta, 'opt_state_alpha': new_actor.opt_state_alpha}, actor_info

def _update_alpha_vmap(
        actor,
        critic,
        temp,
        actor_pop_num: int,
        rng: jax.random.PRNGKey,
        task_ids: int,  # shape = [B]
        param_masks: FrozenDict[str, Any],  # Stack multiple masks along [B]; otherwise use in_axes=None
        stacked_actors_params,
        stacked_critics_params,  # B critics, likewise
        stacked_temps_params,  # B temperature states
        batch_stacked  # B batches
) -> Tuple[jax.random.PRNGKey, MPNTrainState, Any]:
    """
    Update multiple actors in parallel with a single vmap call.
    To validate a single actor first, provide inputs with B=1 to exercise the vmap logic.
    """
    print("actor rng", rng, rng.shape)
    all_keys= jax.random.split(rng, 1 + 1)

    new_rng = all_keys[0]
    keys = all_keys[1:]
    # Generate B keys, one for each actor
    #new_rng, key = jax.random.split(rng)

    # Which dimensions should be vectorized?
    # The vmap in_axes and out_axes roughly correspond to (rng, task_id, param_mask, actor, critic, temp, batch)
    # Set in_axes to None for any shared (unbatched) argument
    update_fn = jax.vmap(
        _update_alpha,
        in_axes=(None, None, None, 0, None, None, 0, 0, 0, 0),  # Vectorize mapped arguments along axis 0
        out_axes=(0, 0, 0)             # Stack outputs along axis 0 as well
    )

    # Run the parallel update
    rngs_out, new_actor_params, info_batches = update_fn(actor,critic, temp,keys, task_ids, param_masks, stacked_actors_params, stacked_critics_params, stacked_temps_params, batch_stacked)

    # Each update produces a new RNG; typically take the last one or combine them using another rule
    # Here, simply use the last RNG as the new RNG
    # new_rng = rngs_out[-1]



    return new_rng, new_actor_params, info_batches


def _update_theta_single(
     actor: MPNTrainState, critic: TrainState, temp: TrainState,
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any],
    actor_params, critic_params, temp_params,
    batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_loss_fn(param) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params':param}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic.apply_fn({'params': critic_params['param']}, batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp.apply_fn({'params': temp_params['param']}) - q).mean()

        _info = {
            'hac_sac_loss': actor_loss,
             'entropy': -log_probs.mean(),
            'means': dicts['means'].mean(),
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        for k  in dicts['fau']:
            _info[k + '_fau'] = jnp.mean(dicts['fau'][k])

        return actor_loss, _info

    # grads of actor
    grads_actor, actor_info = jax.grad(actor_loss_fn, has_aux=True)(actor_params['param'])
    # recording info
    g_norm = global_norm(grads_actor)
    actor_info['g_norm_actor'] = g_norm
    for p, v in param_mask.items():
        if p[-1] == 'kernel':
            actor_info['used_capacity_'+p[0]] = 1.0 - jnp.mean(v)
    # Masking gradients according to cumulative binary masks

    #print("grads_actor shape", grads_actor.shape)

    unfrozen_grads = unfreeze(grads_actor)
    for path, value in param_mask.items():
        cursor = unfrozen_grads
        for key in path[:-1]:
            if key in cursor:
                cursor = cursor[key]
        cursor[path[-1]] *= value

    # only update policy parameters (theta)
    # new_actor = actor.apply_grads_theta(grads=freeze(unfrozen_grads))

    new_actor = actor.apply_grads_theta(grads=freeze(unfrozen_grads), params=actor_params['param'], step=actor_params['step'], opt_state_theta =actor_params['opt_state_theta'],opt_state_alpha =actor_params['opt_state_alpha'])

    return rng, {'param': new_actor.params, 'step': new_actor.step,  'opt_state_theta': new_actor.opt_state_theta, 'opt_state_alpha': new_actor.opt_state_alpha}, actor_info



def _update_theta_vmap(
        actor,
        critic,
        temp,
        actor_pop_num: int,
        rng: jax.random.PRNGKey,
        task_ids: int,  # shape = [B]
        param_masks: FrozenDict[str, Any],  # Stack multiple masks along [B]; otherwise use in_axes=None
        stacked_actors_params,
        stacked_critics_params,  # B critics, likewise
        stacked_temps_params,  # B temperature states
        batch_stacked  # B batches
) -> Tuple[jax.random.PRNGKey, MPNTrainState, Any]:
    """
    Update multiple actors in parallel with a single vmap call.
    To validate a single actor first, provide inputs with B=1 to exercise the vmap logic.
    """
    print("actor rng", rng, rng.shape)
    # Generate B keys, one for each actor
    #rngs = jax.random.split(rng, num_actors)
    all_keys= jax.random.split(rng, 1 + 1)

    new_rng = all_keys[0]
    keys = all_keys[1:]
    # Which dimensions should be vectorized?
    # The vmap in_axes and out_axes roughly correspond to (rng, task_id, param_mask, actor, critic, temp, batch)
    # Set in_axes to None for any shared (unbatched) argument
    update_fn = jax.vmap(
        _update_theta_single,
        in_axes=(None, None ,None, 0, None, None, 0, 0, 0, 0),  # Vectorize mapped arguments along axis 0
        out_axes=(0, 0, 0)             # Stack outputs along axis 0 as well
    )

    # Run the parallel update
    rngs_out, new_actors_params, info_batches = update_fn(actor, critic, temp, keys, task_ids, param_masks, stacked_actors_params, stacked_critics_params, stacked_temps_params, batch_stacked)

    # Each update produces a new RNG; typically take the last one or combine them using another rule
    # Here, simply use the last RNG as the new RNG


    return new_rng, new_actors_params, info_batches









def _update_temp(
    temp: TrainState, temp_param, actor_entropy: float, target_entropy: float
    ) -> Tuple[TrainState, InfoDict]:

    def temperature_loss_fn(param):
        temperature = temp.apply_fn({'params': param})
        temp_loss = temperature * (actor_entropy['entropy'] - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    
    grads_temp, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(temp_param['param'])
    # recording info
    temp_info['g_norm_temp'] = global_norm(grads_temp)
    new_temp = temp.apply_gradients(grads=grads_temp, params=temp_param['param'], step=temp_param['step'], opt_state = temp_param['opt_state'])
    return  {"param":new_temp.params, "opt_state":new_temp.opt_state, "step":new_temp.step}, temp_info




def _update_critic(
        actor: MPNTrainState, critic: TrainState, target_critic: TrainState, temp: TrainState,
    rng: PRNGKey, task_id: int, actor_params, critic_params,
    target_critic_params, update_target: bool, temp_params, batch,
    discount: float, tau: float) -> Tuple[PRNGKey, TrainState, TrainState, InfoDict]:



    rng, key = jax.random.split(rng)
    dist, _ = actor.apply_fn({'params': actor_params['param']}, batch.next_observations, jnp.array([task_id]))
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic.apply_fn({'params': target_critic_params['param']},batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q -= temp.apply_fn({'params': temp_params['param']}) * next_log_probs
    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()}

    grads_critic, critic_info = jax.grad(critic_loss_fn, has_aux=True)(critic_params['param'])
    critic_info['g_norm_critic'] = global_norm(grads_critic)
    new_critic = critic.apply_gradients(grads=grads_critic, params=critic_params['param'], step=critic_params['step'], opt_state = critic_params['opt_state'])
    if update_target:
        new_target_critic = target_update(new_critic, target_critic.update_params(target_critic_params['param']), tau)
    else:
        new_target_critic = target_update(new_critic, target_critic.update_params(target_critic_params['param']), target_critic, 0)

    return rng, {"param":new_critic.params, "opt_state":new_critic.opt_state, "step":new_critic.step}, {"param":new_target_critic.params, "opt_state":new_target_critic.opt_state, "step":new_target_critic.step}, critic_info


@jax.jit
def _rand_net_distill(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, batch: Batch) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:
    
    rng, _ = jax.random.split(rng)
    def distill_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        _, dict = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        means, std = dict['means'], dict['stddev']
        tar_means, tar_std = jnp.zeros_like(means), jnp.ones_like(std)
        # KL-divergence between two Gaussians 
        actor_loss = -0.5 + (jnp.log(tar_std / std) + (std**2 + (means - tar_means)**2) / (2 * tar_std**2)).mean()

        _info = {
            'rnd_loss': actor_loss
        }
        return actor_loss, _info
    
    # grads of actor
    grads_actor, actor_info = jax.grad(distill_loss_fn, has_aux=True)(actor.params)

    # Masking gradients according to cumulative binary masks
    unfrozen_grads = unfreeze(grads_actor)
    for path, value in param_mask.items():
        cursor = unfrozen_grads
        for key in path[:-1]:
            if key in cursor:
                cursor = cursor[key]
        cursor[path[-1]] *= value
    
    # only update policy parameters (theta)
    new_actor = actor.only_apply_grads_theta(grads=freeze(unfrozen_grads))
    return rng, new_actor, actor_info

#
#
# def _update_alpha(actor, all_actor_params,  observations_stacked,
#         rng, task_id: int, param_mask: FrozenDict[str, Any], critic: TrainState, temp: TrainState):
#     rng, key = jax.random.split(rng)
#
#     def single_actor_loss_fn(single_actor_params: Params, observations,   single_key):
#         dist, dicts = actor.apply_fn(
#             {'params': single_actor_params}, observations, jnp.array([task_id]))
#         actions = dist.sample(seed=single_key)
#         log_probs = dist.log_prob(actions)
#         q1, q2 = critic(observations, actions)
#         q = jnp.minimum(q1, q2)
#         actor_loss = (log_probs * temp()- q).mean()
#         _info = {
#             'hac_sac_loss': actor_loss,
#             'entropy': -log_probs.mean(),
#             'means': dicts['means'].mean()
#         }
#         for k in dicts['masks']:
#             _info[k + '_rate_act'] = jnp.mean(dicts['masks'][k])
#         for k in dicts['fau']:
#             _info[k + '_fau'] = jnp.mean(dicts['fau'][k])
#         return actor_loss, _info
#
#     single_actor_grad_fn = jax.value_and_grad(single_actor_loss_fn, has_aux=True)
#
#     batched_grad_fn = jax.vmap(
#         single_actor_grad_fn,
#         in_axes= (0, 0,  None),
#         out_axes=(0, 0)
#     )
#
#     # print("observations_stacked", observations_stacked.shape)
#     # print("task_id", jnp.array([task_id]).shape)
#     # print("keys", keys.shape)
#
#     (_, actor_info_batched), grads_batched = batched_grad_fn(
#        all_actor_params,
#        observations_stacked,
#        key
#     )
#
#     g_norms= jax.vmap(global_norm)(grads_batched)
#     # grads of actor
#
#     def update_actor_info(single_actor_info, g_norm, param_mask):
#         new_info = dict(single_actor_info)
#         new_info['g_norm_actor'] = g_norm
#         for p, v in param_mask.items():
#             if p[-1] == 'kernel':
#                 new_info['used_capacity_' + p[0]] = 1.0 - jnp.mean(v)
#         return new_info
#
#     #print("Pre??? \n", actor_info_batched)
#
#     actor_info_batched_updated = jax.vmap(update_actor_info, in_axes=(0, 0, None))(
#         actor_info_batched,  # (N,) info dictionary for each actor
#         g_norms,  # (N,) gradient norm for each actor
#         jax.tree_map(lambda v: v, param_mask)  # Pass param_mask as a static argument
#     )
#     def apply_grads_alpha_single(actor_params, grads_actor):
#         new_actor = actor.replace(params=actor_params)
#         new_actor = new_actor.apply_grads_alpha(grads=grads_actor)
#         return new_actor.params
#
#
#     batched_apply_grads_alpha = jax.vmap(
#         apply_grads_alpha_single,
#         in_axes=(0,0),
#         out_axes=0
#     )
#     new_actors = batched_apply_grads_alpha(all_actor_params, grads_batched)
#     return rng, new_actors, actor_info_batched_updated


# def _update_theta(actor, all_actor_params,  observations_stacked,
#         rng, task_id: int, param_mask: FrozenDict[str, Any], critic: TrainState, temp: TrainState):
#     # all_keys = jax.random.split(rng, 6+1)
#     # keys = all_keys[:-1]
#     # rng = all_keys[-1]
#
#     rng, key = jax.random.split(rng)
#     def single_actor_loss_fn(single_actor_params: Params, observations,  single_key):
#         dist, dicts = actor.apply_fn(
#             {'params': single_actor_params}, observations, jnp.array([task_id]))
#         actions = dist.sample(seed=single_key)
#         log_probs = dist.log_prob(actions)
#         q1, q2 = critic(observations, actions)
#         q = jnp.minimum(q1, q2)
#         actor_loss = (log_probs * temp()- q).mean()
#         _info = {
#             'hac_sac_loss': actor_loss,
#             'entropy': -log_probs.mean(),
#             'means': dicts['means'].mean()
#         }
#         for k in dicts['masks']:
#             _info[k + '_rate_act'] = jnp.mean(dicts['masks'][k])
#
#         for k in dicts['fau']:
#             _info[k + '_fau'] = jnp.mean(dicts['fau'][k])
#         return actor_loss, _info
#
#     single_actor_grad_fn = jax.value_and_grad(single_actor_loss_fn, has_aux=True)
#     batched_grad_fn = jax.vmap(
#         single_actor_grad_fn,
#         in_axes= (0, 0,  None),
#         out_axes=(0, 0)
#     )
#     (_, actor_info_batched), grads_batched = batched_grad_fn(
#        all_actor_params,
#        observations_stacked,
#        key)
#     g_norms= jax.vmap(global_norm)(grads_batched)
#     # grads of actor
#     def update_actor_info(single_actor_info, g_norm, param_mask):
#         new_info = dict(single_actor_info)
#         new_info['g_norm_actor'] = g_norm
#         for p, v in param_mask.items():
#             if p[-1] == 'kernel':
#                 new_info['used_capacity_' + p[0]] = 1.0 - jnp.mean(v)
#         return new_info
#
#     actor_info_batched_updated = jax.vmap(update_actor_info, in_axes=(0, 0, None))(
#         actor_info_batched,  # (N,) info dictionary for each actor
#         g_norms,  # (N,) gradient norm for each actor
#         jax.tree_map(lambda v: v, param_mask)  # Pass param_mask as a static argument
#     )
#
#     def apply_grads_theta_single(actor_params, grads_actor):
#         unfrozen_grads =  unfreeze(grads_actor)
#         for path, value in param_mask.items():
#             cursor = unfrozen_grads
#             for key in path[:-1]:
#                 if key in cursor:
#                     cursor = cursor[key]
#             cursor[path[-1]] *= value
#         new_actor = actor.replace(params=actor_params)
#         new_actor = new_actor.apply_grads_theta(grads=freeze(unfrozen_grads))
#         return new_actor.params
#
#     batched_apply_grads_theta = jax.vmap(
#         apply_grads_theta_single,
#         in_axes=(0,0),
#         out_axes=0
#     )
#     new_actors = batched_apply_grads_theta(all_actor_params, grads_batched)
#     return rng, new_actors, actor_info_batched_updated
#



@functools.partial(jax.jit, static_argnames=('update_target', 'critic_pop_num', 'actor_pop_num'))
def _update_cotasp(stacked_actor_params, stacked_temp_params, stacked_critic_params, stacked_target_critics_params, actor_pop_num, critic_pop_num, rng: PRNGKey, task_id: int, tau: float, discount: float,
    target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any],
    actor_pop, critic_pop, target_critic_pop,
    update_target, temp_pop, batch_list
    ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:


    keys = jax.random.split(rng, actor_pop_num + 1)

    new_rng = keys[0]
    keys = keys[1:]

    batch_stacked = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *batch_list)  # jnp.stack(batch_list, axis=0)


    vmap_fn  =jax.vmap(
        _update_cotasp_single,
        in_axes=( 0, 0, 0, 0, 0,
                 None, None, None, None, None, None, None, None, None, None ,None,
                 0),
        out_axes=(0, 0, 0, 0, 0, 0)  # Stack outputs along axis 0
    )

    _, new_actor_params, new_temp_params, new_critic_params, new_target_critic_params, info_batch = vmap_fn(
         stacked_actor_params, stacked_critic_params, stacked_target_critics_params, stacked_temp_params,keys,
        task_id, tau, discount, target_entropy, optimize_alpha, param_mask,
        actor_pop[0], critic_pop[0], target_critic_pop[0], update_target, temp_pop[0], batch_stacked)

#    for i in range(actor_pop_num):
#        param_info = jax.tree_util.tree_map(lambda x: x[i], new_actor_params)
#        actor_pop[i] = actor_pop[i].update(param_info['param'], param_info['opt_state_theta'], param_info['opt_state_alpha'], param_info['step'])
#        temp_param_info = jax.tree_util.tree_map(lambda x: x[i], new_temp_params)
#        temp_pop[i] = temp_pop[i].update(temp_param_info['param'], temp_param_info['opt_state'],  temp_param_info['step'])
#    for i in range(actor_pop_num):
#        critic_param_info = jax.tree_util.tree_map(lambda x: x[i],new_critic_params)
#        critic_pop[i] = critic_pop[i].update(critic_param_info['param'], critic_param_info['opt_state'],
#                                                       critic_param_info['step'])
#        target_critic_param_info = jax.tree_util.tree_map(lambda x: x[i], new_target_critic_params)
#        target_critic_pop[i] = target_critic_pop[i].update(target_critic_param_info['param'], target_critic_param_info['opt_state'], target_critic_param_info['step'])
    
    actor_pop = None
    temp_pop = None
    critic_pop =None
    target_critic_pop = None
    return  actor_pop, temp_pop, critic_pop, target_critic_pop, new_rng, new_actor_params, new_temp_params, new_critic_params, new_target_critic_params, info_batch

def _update_cotasp_single( actor_param, critic_param, target_critic_param, temp_param, rng: PRNGKey, task_id: int, tau: float, discount: float,
                       target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any],
                       actor: MPNTrainState, critic: TrainState, target_critic: TrainState,
                       update_target, temp: TrainState, batch: Batch
                       ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:


    actor = actor.update(actor_param['param'], actor_param['opt_state_theta'], actor_param['opt_state_alpha'], actor_param['step'])
    critic = critic.update(critic_param['param'], critic_param['opt_state'], critic_param['step'])
    target_critic = target_critic.update(target_critic_param['param'], target_critic_param['opt_state'], target_critic_param['step'])
    temp = temp.update(temp_param['param'], temp_param['opt_state'], temp_param['step'])

    # optimizing critics
    new_rng, new_critic_param, new_target_critic_param, critic_info = _update_critic(actor, critic, target_critic, temp,
        rng, task_id, actor_param, critic_param, target_critic_param, update_target, temp_param,
        batch, discount, tau
    )
    # optimizing either alpha or theta
    new_rng, new_actor_param, actor_info = jax.lax.cond(
        optimize_alpha,
        _update_alpha,
        _update_theta_single,
         actor, critic, temp, new_rng, task_id, param_mask, actor_param, new_critic_param, temp_param, batch
    )
    # updating temperature coefficient
    new_temp_param, temp_info = _update_temp(
         temp, temp_param,  actor_info, target_entropy
    )
    return new_rng, new_actor_param, new_temp_param, new_critic_param, new_target_critic_param, {
        **actor_info,
        **temp_info,
        **critic_info}


import time

import numpy as np

import copy


class MaskSampler:
    def __init__(self, hidn):
        self.hidn = hidn
        self.selected_indices = set()

        self.history_masks = []
        self.previous_task_embedding = []

    def caculate_similarity(self, e, task_id):

        if task_id == 0:
            return -1, 0
        elif task_id == 1:

            emb = self.previous_task_embedding[0]
            dot_product = np.dot(np.array(emb).flatten(), np.array(e).flatten())
            norm_A = np.linalg.norm(np.array(emb).flatten())
            norm_B = np.linalg.norm(np.array(e).flatten())
            similarity = dot_product / (norm_A * norm_B)
            return 0, similarity
        else:
            similarities = []



            for emb in self.previous_task_embedding:
                dot_product = np.dot(np.array(emb).flatten(), np.array(e).flatten())
                norm_A = np.linalg.norm(np.array(emb).flatten())
                norm_B = np.linalg.norm(np.array(e).flatten())
                similarity = dot_product / (norm_A * norm_B)
                similarities.append(similarity)
            sorted_indices = np.argsort(similarities)[::-1]
            top_k_indices = sorted_indices[:5]  # Take the first top_k indices
            top_k_similarities = np.array(similarities)[top_k_indices]
            # Normalize to a probability distribution (softmax)
            exp_similarities = np.exp(top_k_similarities / 0.5)
            probabilities = exp_similarities / np.sum(exp_similarities)
            # Randomly select an index according to the softmax probabilities
            selected_local_index = np.random.choice(len(self.previous_task_embedding[:5]), p=probabilities)
            # Map back to the original index
            selected_index = top_k_indices[selected_local_index]
            selected_similarity = similarities[selected_index]
            return selected_index, selected_similarity

    def sample(self, n, current_index):
        mask = np.full(self.hidn, 0.0)
        if len(self.previous_task_embedding) > 0:
            selected_ind = current_index
        if len(self.previous_task_embedding) == 0:
            available_indices = list(set(range(self.hidn)) - self.selected_indices)
            selected = np.random.choice(available_indices, n, replace=False)
            mask[selected] = 1.0
            count = np.count_nonzero(mask > 0)
            assert count == n
        else:
            prev_mask = self.history_masks[selected_ind]
            available_indices = list(set(range(self.hidn)) - self.selected_indices)
            if n >= len(available_indices):
                selected = np.random.choice(available_indices, len(available_indices), replace=False)
            else:
                selected = np.random.choice(available_indices, int(n), replace=False)
            mask[selected] = 1.0
            mask = prev_mask + mask
            assert np.sum(mask > 1) == 0
        return mask

    def add_history_mask(self, mask, e):
        self.previous_task_embedding.append(e)
        indices = np.where(mask.flatten() > 0)
        indices_list = [index[0] for index in zip(*indices)]
        self.selected_indices.update(indices_list)
        self.history_masks.append(copy.deepcopy(mask))


class CoTASPLearner(SACLearner):
    def __init__(
        self,
        population_size,
        use_critic_pop,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        task_num: int,
        load_policy_dir: Optional[str] = None,
        load_dict_dir: Optional[str] = None,
        update_dict = True,
        update_coef = True,
        dict_configs: dict = {},
        pi_opt_configs: dict = {},
        q_opt_configs: dict = {},
        t_opt_configs: dict = {},
        actor_configs: dict = {},
        critic_configs: dict = {},
        tau: float = 0.005,
        discount: float = 0.99,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0):
        super(CoTASPLearner, self).__init__(population_size, use_critic_pop, seed, observations, actions, pi_opt_configs,
              q_opt_configs, t_opt_configs, actor_configs, critic_configs, discount, 
              tau, target_update_period, target_entropy, init_temperature)

        print("1")
        action_dim = actions.shape[-1]

        self.rng, actor_key = jax.random.split(self.rng, 2)

        actor_keys = jax.random.split(actor_key, self.population_size)

        actor_configs['task_num'] = task_num
        actor_configs['action_dim'] = action_dim
        print("2")
        self.actor_pop = []
        for i in range(self.population_size):

            actor_def = policies.MetaPolicy(**actor_configs)
            _, actor_params = actor_def.init(actor_keys[i], observations, jnp.array([0])).pop('params')
            actor = MPNTrainState.create(
                apply_fn=actor_def.apply,
                params=actor_params,
                tx=utils_fn.set_optimizer(**pi_opt_configs)
            )
            self.actor_pop.append(actor)
        # if load_policy_dir is not None:
        #     actor = actor.load(load_policy_dir)
        print("3")
        # get grads masking func
        def get_grad_masks(model, masks):
            g_masks = model.get_grad_masks(masks)
            return g_masks
        get_grad_masks_jit = jax.jit(nn.apply(get_grad_masks, actor_def))

        # preset dict learner for each layer:


        self.layer_sampler = []
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            self.layer_sampler.append(MaskSampler(hidn))

        # self.dict4layers = {}
        # for _ in range(self.population_size):
        #     dict4layers = {}
        #     for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
        #         dict_learner = OnlineDictLearnerV2(
        #             384,
        #             hidn,
        #             seed*(_+1)+id_layer+1,
        #             None, # whether using svd dictionary initialization
        #             **dict_configs)
        #         dict4layers[f'embeds_bb_{id_layer}'] = dict_learner
        #     self.dict4layers[_] = dict4layers
        print("4")
        # if load_dict_dir is not None:
        #     for k in self.dict4layers.keys():
        #         self.dict4layers[k].load(f'{load_dict_dir}/{k}.pkl')

        # initialize param_masks
        self.rng, _, dicts = _sample_actions(
            self.rng, self.actor_pop[0], self.dummy_o, jnp.array([0])
        )
        self.cumul_masks = tree_map(lambda x: jnp.zeros_like(x), dicts['masks'])
        self.param_masks = freeze(
            get_grad_masks_jit({'params': actor.params}, self.cumul_masks)
        )

        # initialize other things
        self.update_dict = update_dict
        self.update_coef = update_coef
        self.actor_cfgs = actor_configs
        self.get_grad_masks = get_grad_masks_jit
        self.target_update_period = target_update_period
        self.task_embeddings = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

        self.current_masks = {}

        self.batch_actor_params = None
        self.batch_critic_params = None
        self.batch_target_critic_params = None
        self.batch_temp_params = None



        print("5")

    def reduce_mask(self, bar: float, batch: Batch, task_id: int):

        all_masks = []
        for agent_index in range(self.population_size):
            # sample a batch to calculate each output
            dist, dicts = self.actor_pop[agent_index](batch.observations, jnp.array([task_id]))
            record_layer_output = dicts['record_layer_output']
            mask_dict = {}

            one_mask = []
            for index, k in enumerate(dicts['masks']):

                print("current ratio", jnp.mean(dicts['masks'][k] > 0.0))
                out_value = record_layer_output[k]
                current_mask = self.current_masks[agent_index][index]
                mask_indices = jnp.where(current_mask == 1.0)[0]
                out_value_masked = out_value[mask_indices]

                weighted_values = out_value_masked / jnp.sum(out_value_masked)

                zero_positions = jnp.where(weighted_values <= bar)[0]
                used_positions = jnp.where(weighted_values > bar)[0]

                new_mask = self.current_masks[agent_index][index].at[mask_indices[zero_positions]].set(0)
                # num_to_modify = int(len(mask_indices) * p_ratio)
                # sorted_indices = jnp.argsort(out_value_masked)[:num_to_modify]
                # new_mask = self.current_masks[index].at[mask_indices[sorted_indices]].set(0)
                mask_dict[index] = new_mask
                # test_mask = (new_mask == 1) & (layer_cucml_mask == 0)
                # each_layer[k] = jnp.mean(test_mask)
                # each_layer_total[k] = jnp.mean(new_mask)
                # print(index, "????", jnp.mean(current_mask > 0.0), jnp.mean(new_mask > 0.0), jnp.mean(out_value_masked), jnp.max(out_value_masked), jnp.min(out_value_masked))
                # print(bar,"   ",jnp.mean(weighted_values), jnp.max(weighted_values), jnp.min(weighted_values))

                # print("========================")
                # print(weighted_values)
                # print("===========================")
                #print(zero_positions)
                # print(bar)
            actor_params = unfreeze(self.actor_pop[agent_index].params)
            my_index = 0
            for index, k in enumerate(self.actor_pop[agent_index].params.keys()):
                if k.startswith('embeds'):
                    # alpha_l = self.dict4layers[k].get_alpha(task_e)
                    alpha_l = jnp.asarray(mask_dict[my_index].flatten())
                    one_mask.append(np.array(alpha_l))
                    print(my_index, "final alpha_l", jnp.mean(alpha_l > 0.0))
                    my_index += 1
                    # Replace the i-th row
                    actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
            all_masks.append(one_mask)
            self.actor_pop[agent_index] = self.actor_pop[agent_index].update_params(freeze(actor_params))



        actor_param_list = [{"param": actor.params, "opt_state_theta": actor.opt_state_theta,
                             "opt_state_alpha": actor.opt_state_alpha, "step": actor.step} for actor in self.actor_pop]
        stacked_actor_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *actor_param_list)

        self.batch_actor_params = stacked_actor_params


        return all_masks


    def get_current_mask_activate_ratio(self, bar: float, batch: Batch, task_id: int):

        all_mask_values = {}
        for agent_index in range(self.population_size):
            # sample a batch to calculate each output
            dist, dicts = self.actor_pop[agent_index](batch.observations, jnp.array([task_id]))
            record_layer_output = dicts['record_layer_output']
            mask_values = []
            for index, k in enumerate(dicts['masks']):

                #print("current ratio", jnp.mean(dicts['masks'][k] > 0.0))
                out_value = record_layer_output[k]
#                current_mask = self.current_masks[agent_index][index]
                mask_values.append(out_value)

            all_mask_values[agent_index] = mask_values


        return all_mask_values




       # return each_layer_value_mean, each_layer_used_value_mean, org_current_dict, each_layer, each_layer_total, new_dict_to_test, new_dict_to_test_0_dot_1


    def start_task(self, seleced_ratio, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_e)
        self.task_e = task_e
        # if task_id == 0 :
        #     sampled_neuro_num = int(seleced_ratio*1024)
        # else:
        #     sampled_neuro_num = int(seleced_ratio*1024/2)

        max_ratio = seleced_ratio
        min_ratio = 0.01
        all_masks = []
        for pppp in range(self.population_size):

            self.current_masks[pppp] = []
            select_index, similarity = self.layer_sampler[0].caculate_similarity(task_e, task_id)

            # Capacity study: the per-layer neuron budget follows the actual actor width instead of
            # the original hard-coded 1024 (identical result at width 1024: int(0.1 * 1024) = 102).
            if task_id == 0 :
                ratio = seleced_ratio
            else:
                ratio = min_ratio + max(0.0, (1.0 - similarity))*(max_ratio - min_ratio)
                print("task_id ", task_id, " add ratio", ratio)
            layer_widths = list(self.actor_cfgs['hidden_dims'])
            # set initial alpha for each layer of MPN
            actor_params = unfreeze(self.actor_pop[pppp].params)
            my_index = 0

            one_mask = []
            for index, k in enumerate(self.actor_pop[pppp].params.keys()):
                if k.startswith('embeds'):
                    sampled_neuro_num = max(1, int(ratio * layer_widths[my_index]))
                    alpha_l = self.layer_sampler[my_index].sample(sampled_neuro_num, select_index)
                    # self.dict4layers[_][k].get_alpha(task_e)
                    my_index +=1
                    alpha_l = jnp.asarray(alpha_l.flatten())
                    self.current_masks[pppp].append(jnp.where(alpha_l > 0, 1.0, 0.0))
                    actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
                    one_mask.append(np.array(alpha_l))
            all_masks.append(one_mask)
            self.actor_pop[pppp] = self.actor_pop[pppp].update_params(freeze(actor_params))

        return all_masks



    def fuse_params(self, params_a, params_b, alpha=0.1):
        return freeze(jax.tree_map(lambda a, b: alpha * a + (1 - alpha) * b, unfreeze(params_a), unfreeze(params_b)))

    def _merge_one_tensor(self,replace_ratio, w_a, w_b, m_a, m_b, m_common):
        """
        Merge weights of the same shape elementwise using masks.
        m_a, m_b, and m_common all have shape (out_dim,).
        """
        m_a = jnp.asarray(m_a, dtype=w_a.dtype)
        m_b = jnp.asarray(m_b, dtype=w_a.dtype)
        m_common = jnp.asarray(m_common, dtype=w_a.dtype)


        np.set_printoptions(threshold=np.inf)
        # Mutual-exclusion check (optional)
        # print("m_a", m_a)
        # print("m_b", m_b)
        # print("m_common", m_common)
        #
        # print("final", m_a + m_b + m_common)

        assert jnp.all((m_a + m_b + m_common) <= 1 + 1e-5)


        # print(m_a + m_b + m_common )
        # print(jnp.mean(m_a + m_b + m_common ))

        assert  jnp.all((m_a + m_b + m_common) > 0.1)
        def _bcast(mask, w):
            # while mask.ndim < w.ndim:
            #     mask = mask[..., None]  # Repeatedly add a trailing dimension
            print(mask.shape, w.shape)
            return jnp.broadcast_to(mask, w.shape)
        mask_a = _bcast(m_a, w_a)
        mask_b = _bcast(m_b, w_a)
        mask_c = _bcast(m_common, w_a)

        return w_a*mask_c + replace_ratio * w_a *mask_a  + (1-replace_ratio) * w_b * mask_a  + replace_ratio*w_b * mask_b + (1-replace_ratio) * w_a * mask_b


    def set_new_mask(self,  replace_ratio, task_id,  change_ind_index, new_masks,parents_index, uncommon_parents_masks, common_masks):


        for changed_index, new_mask, parent_index_list, parents_un_common_masks_pairs, common_mask  in zip(change_ind_index, new_masks, parents_index, uncommon_parents_masks, common_masks):
            # parents_mask_index_list contains the two parent masks, which indicate active parameters; use them to extract each parent's parameters separately

            child  = self.actor_pop[changed_index]
            parent_A = self.actor_pop[parent_index_list[0]]
            parent_B = self.actor_pop[parent_index_list[1]]

            child_p = unfreeze(child.params)

            mask_index = 0
            for name, p_A_layer in parent_A.params.items():
                if not name.startswith('backbones'):
                    continue

                #print(np.array(parents_un_common_masks_pairs).shape, np.array(parents_un_common_masks_pairs[0]).shape, np.array(parents_un_common_masks_pairs[0][mask_index]).shape)
                if np.mean(parents_un_common_masks_pairs[0][mask_index]) > 0.0 :
                    p_B_layer = parent_B.params[name]
                    merged_kernel = self._merge_one_tensor(
                        replace_ratio,
                        p_A_layer['kernel'],
                        p_B_layer['kernel'],
                        parents_un_common_masks_pairs[0][mask_index],  # A‑mask
                        parents_un_common_masks_pairs[1][mask_index],  # B‑mask
                        common_mask[mask_index]  # Shared
                    )
                    merged_bias = self._merge_one_tensor(
                        replace_ratio,
                        p_A_layer['bias'],
                        p_B_layer['bias'],
                        parents_un_common_masks_pairs[0][mask_index],
                        parents_un_common_masks_pairs[1][mask_index],
                        common_mask[mask_index]
                    )
                    child_p[name]['kernel'] =merged_kernel
                    child_p[name]['bias'] = merged_bias
                # Could the corresponding parameters be replaced here, layer by layer?
                mask_index += 1
            self.actor_pop[changed_index] = child.replace(params=freeze(child_p))

            self.critic_pop[changed_index] =  self.critic_pop[parent_index_list[0]].replace()
            self.temp_pop[changed_index] = self.temp_pop[parent_index_list[0]].replace()
            self.target_critic_pop[changed_index] = self.target_critic_pop[parent_index_list[0]].replace()

            self.current_masks[changed_index] = []
            # set initial alpha for each layer of MPN
            actor_params = unfreeze(self.actor_pop[changed_index].params)
            my_index = 0
            for index, k in enumerate(self.actor_pop[changed_index].params.keys()):
                if k.startswith('embeds'):
                    alpha_l = jnp.asarray(new_mask[my_index])
                    my_index += 1
                    self.current_masks[changed_index].append(jnp.where(alpha_l > 0, 1.0, 0.0))
                    actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
            self.actor_pop[changed_index] = self.actor_pop[changed_index].update_params(freeze(actor_params))

        actor_param_list = [{"param": actor.params, "opt_state_theta": actor.opt_state_theta,
                             "opt_state_alpha": actor.opt_state_alpha, "step": actor.step} for actor in
                            self.actor_pop]
        temp_param_list = [{"param": temp.params, "opt_state": temp.opt_state, "step": temp.step} for temp in
                           self.temp_pop]
        critic_param_list = [{"param": critic.params, "opt_state": critic.opt_state, "step": critic.step} for critic
                             in self.critic_pop]
        critic_target_param_list = [
            {"param": target_critic.params, "opt_state": target_critic.opt_state, "step": target_critic.step} for
            target_critic in self.target_critic_pop]

        stacked_actor_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *actor_param_list)
        stacked_temp_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *temp_param_list)
        stacked_critic_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *critic_param_list)
        stacked_target_critics_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0),
                                                     *critic_target_param_list)

        self.batch_actor_params = stacked_actor_params
        self.batch_critic_params = stacked_critic_params
        self.batch_target_critic_params = stacked_target_critics_params
        self.batch_temp_params = stacked_temp_params





    def set_mask(self,  replace_ratio, task_id,  change_ind_index, new_masks,parents_index):

        for changed_index, new_mask, parent_index_list in zip(change_ind_index, new_masks, parents_index):

            params_a = self.actor_pop[parent_index_list[0]].params
            params_b = self.actor_pop[parent_index_list[1]].params
            fused_params = self.fuse_params(params_a, params_b, alpha=replace_ratio)
            # use elite to replace current individual
            self.actor_pop[changed_index] = self.actor_pop[parent_index_list[0]].replace()
            # update params with the fused params
            self.actor_pop[changed_index] = self.actor_pop[changed_index].replace(params=fused_params)
            self.critic_pop[changed_index] =  self.critic_pop[parent_index_list[0]].replace()
            self.temp_pop[changed_index] = self.temp_pop[parent_index_list[0]].replace()
            self.target_critic_pop[changed_index] = self.target_critic_pop[parent_index_list[0]].replace()

            self.current_masks[changed_index] = []
            # set initial alpha for each layer of MPN
            actor_params = unfreeze(self.actor_pop[changed_index].params)
            my_index = 0
            for index, k in enumerate(self.actor_pop[changed_index].params.keys()):
                if k.startswith('embeds'):
                    alpha_l = jnp.asarray(new_mask[my_index])
                    my_index += 1
                    self.current_masks[changed_index].append(jnp.where(alpha_l > 0, 1.0, 0.0))
                    actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
            self.actor_pop[changed_index] = self.actor_pop[changed_index].update_params(freeze(actor_params))

        actor_param_list = [{"param": actor.params, "opt_state_theta": actor.opt_state_theta,
                             "opt_state_alpha": actor.opt_state_alpha, "step": actor.step} for actor in
                            self.actor_pop]
        temp_param_list = [{"param": temp.params, "opt_state": temp.opt_state, "step": temp.step} for temp in
                           self.temp_pop]
        critic_param_list = [{"param": critic.params, "opt_state": critic.opt_state, "step": critic.step} for critic
                             in self.critic_pop]
        critic_target_param_list = [
            {"param": target_critic.params, "opt_state": target_critic.opt_state, "step": target_critic.step} for
            target_critic in self.target_critic_pop]

        stacked_actor_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *actor_param_list)
        stacked_temp_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *temp_param_list)
        stacked_critic_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *critic_param_list)
        stacked_target_critics_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0),
                                                     *critic_target_param_list)

        self.batch_actor_params = stacked_actor_params
        self.batch_critic_params = stacked_critic_params
        self.batch_target_critic_params = stacked_target_critics_params
        self.batch_temp_params = stacked_temp_params



    def rand_net_distill(self, task_id: int, batch: Batch) -> InfoDict:

        for _ in range(self.population_size):
            new_rng, new_actor, info = _rand_net_distill(
                self.rng, task_id, self.param_masks, self.actor_pop[_], batch
            )
            self.rng = new_rng
            self.actor_pop[_] = new_actor
        return info

    def sample_actions(self, select_actor_index,
                       observations: np.ndarray,
                       task_id: int,
                       temperature: float = 1.0) -> jnp.ndarray:

        rng, actions, _ = _sample_actions(self.rng, self.actor_pop[select_actor_index], observations,
                                          jnp.array([task_id]), temperature)
        self.rng = rng
        actions = self.invalid_filter(actions)

        return actions

    def trans_actor(self):
        assert len(self.actor_pop) == self.population_size
        for i in range(len(self.actor_pop)):
            param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_actor_params)
            self.actor_pop[i] = self.actor_pop[i].update(param_info['param'], param_info['opt_state_theta'], param_info['opt_state_alpha'], param_info['step'])
           # temp_param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_temp_params)
           # self.temp_pop[i] = self.temp_pop[i].update(temp_param_info['param'], temp_param_info['opt_state'],temp_param_info['step'])

    def trans_all(self):
        for i in range(len(self.actor_pop)):
            param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_actor_params)
            self.actor_pop[i] = self.actor_pop[i].update(param_info['param'], param_info['opt_state_theta'],  param_info['opt_state_alpha'], param_info['step'])
            temp_param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_temp_params)
            self.temp_pop[i] = self.temp_pop[i].update( temp_param_info['param'], temp_param_info['opt_state'], temp_param_info['step'])
        for i in range(len(self.critic_pop)):
            critic_param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_critic_params)
            self.critic_pop[i] = self.critic_pop[i].update( critic_param_info['param'], critic_param_info['opt_state'], critic_param_info['step'])
            target_critic_param_info = jax.tree_util.tree_map(lambda x: x[i], self.batch_target_critic_params)
            self.target_critic_pop[i] = self.target_critic_pop[i].update( target_critic_param_info['param'], target_critic_param_info['opt_state'], target_critic_param_info['step'])



    def update(self, task_id: int, batch_list , optimize_alpha: bool=False, use_info=False) -> InfoDict:

        s = time.time()
        # for i in range(5):
        if not self.update_coef:
            optimize_alpha = False

        update_target = self.step % self.target_update_period == 0

        if self.batch_actor_params is None:
            actor_param_list = [{"param": actor.params, "opt_state_theta": actor.opt_state_theta,
                                 "opt_state_alpha": actor.opt_state_alpha, "step": actor.step} for actor in
                                self.actor_pop]
            temp_param_list = [{"param": temp.params, "opt_state": temp.opt_state, "step": temp.step} for temp in
                               self.temp_pop]
            critic_param_list = [{"param": critic.params, "opt_state": critic.opt_state, "step": critic.step} for critic
                                 in self.critic_pop]

            critic_target_param_list = [
                {"param": target_critic.params, "opt_state": target_critic.opt_state, "step": target_critic.step} for
                target_critic in self.target_critic_pop]

            stacked_actor_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *actor_param_list)
            stacked_temp_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *temp_param_list)
            stacked_critic_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0), *critic_param_list)
            stacked_target_critics_params = jax.tree_map(lambda *arrs: jnp.stack(arrs, axis=0),
                                                         *critic_target_param_list)

            self.batch_actor_params = stacked_actor_params
            self.batch_critic_params = stacked_critic_params
            self.batch_target_critic_params = stacked_target_critics_params
            self.batch_temp_params = stacked_temp_params

        s1 = time.time()
        # for i in range(5):
        actor_pop, temp_pop, critic_pop, target_critic_pop, new_rng, new_actor_params, new_temp_params, new_critic_params, new_target_critic_params, info_batch= _update_cotasp( self.batch_actor_params, self.batch_temp_params, self.batch_critic_params, self.batch_target_critic_params, len(self.actor_pop), len(self.critic_pop),
            self.rng, task_id, self.tau, self.discount, self.target_entropy, optimize_alpha,
            self.param_masks, self.actor_pop, self.critic_pop, self.target_critic_pop, update_target, self.temp_pop, batch_list)

        self.step += 1
        self.rng = new_rng

        self.batch_actor_params = new_actor_params
        self.batch_critic_params = new_critic_params
        self.batch_target_critic_params = new_target_critic_params
        self.batch_temp_params = new_temp_params
        s2 = time.time()
        # for i in range(5):
        info = {}
        if use_info:
            for i in range(self.population_size):
                for k, v in info_batch.items():
                    # Construct key: "agent_<id>/<k>"
                    info[f"agent_{i}/{k}"] = v[i]
        s3 = time.time()
        # for i in range(5):
        return info, np.array([s1-s, s2-s1, s3- s2])

    def reset_agent(self, elite_index):

        self.trans_all()

        self.batch_actor_params = None
        self.batch_critic_params = None
        self.batch_target_critic_params = None
        self.batch_temp_params = None

        for _ in range(self.population_size):
            if _ != elite_index:
                self.actor_pop[_] = self.actor_pop[_].update_params(self.actor_pop[elite_index].params)

        # re-initialize params of critic ,target_critic and temperature
        self.rng, key_critic, key_temp = jax.random.split(self.rng, 3)

        critic_keys = jax.random.split(key_critic, self.population_size)

        for index, critic in enumerate(self.critic_pop):
            critic = utils_fn.reset_model(
                critic,
                critic_net.DoubleCritic,
                self.critic_cfgs,
                [critic_keys[index], self.dummy_o, self.dummy_a])
            self.critic_pop[index] = critic

            self.target_critic_pop[index] = self.target_critic_pop[index].update_params(critic.params)

        # self.critic = utils_fn.reset_model(
        #     self.critic,
        #     critic_net.DoubleCritic,
        #     self.critic_cfgs,
        #     [key_critic, self.dummy_o, self.dummy_a])
        #self.target_critic = self.target_critic.update_params(self.critic.params)

        for index in range(self.population_size):
            self.temp_pop[index] = utils_fn.reset_model(
                self.temp_pop[index],
                temperature.Temperature,
                {'init_log_temp': self.init_temp},
                [key_temp])

        # re-initialize unused params of meta-policy network
        # new_rng, new_params = utils_fn.reset_free_params(
        #     self.actor.params,
        #     self.param_masks,
        #     policies.MetaPolicy,
        #     self.actor_cfgs,
        #     [key_actor, self.dummy_o, jnp.array([0])],
        #     adaptive_init=True
        # )
        # re-initialize log_std_layer's params
        # self.rng, new_params = utils_fn.reset_logstd_layer(
        #     key_actor,
        #     self.actor.params,
        #     self.actor_cfgs['final_fc_init_scale'],
        #     self.actor_cfgs['state_dependent_std']
        # )
        # self.actor = self.actor.update_params(new_params)

        # reset optimizers
        for i in range(len(self.actor_pop)):
            self.temp_pop[i] = self.temp_pop[i].reset_optimizer()
        for i in range(len(self.critic_pop)):
            self.critic_pop[i] = self.critic_pop[i].reset_optimizer()
        self.reset_actor_optimizer()
        
    def reset_actor_optimizer(self):
        for _ in range(self.population_size):
            self.actor_pop[_] = self.actor_pop[_].reset_optimizer()


    def end_task(self, best_index, task_id: int, save_actor_dir: str, save_dict_dir: str):
        self.step = 0

        self.rng, _, dicts = _sample_actions(
            self.rng, self.actor_pop[best_index], self.dummy_o, jnp.array([task_id])
        )
        current_masks = dicts['masks']

        a = []
        for key in dicts['masks']:
            a.append(dicts['masks'][key].tolist())

        str_a = str(a)

        print("task id", task_id, "vectors")
        print(str_a)

        for index, mask_key in enumerate(dicts['masks'].keys()):
            self.layer_sampler[index].add_history_mask(np.array(dicts['masks'][mask_key]), self.task_e)



        # cumulative attention from all the previous tasks
        self.cumul_masks = tree_map(
            lambda a, b: jnp.maximum(a, b), self.cumul_masks, current_masks
        )
        # To condition the training of task t + 1, we compute
        # the gradient masks according to cumulative binary masks
        grad_masks = self.get_grad_masks(
            {'params': self.actor_pop[best_index].params}, self.cumul_masks)

        self.param_masks = freeze(grad_masks)

        # update dictionary learners
        dict_stats = {}
        # if self.update_dict:
        #
        #
        #     for k in self.actor_pop[best_index].params.keys():
        #         if k.startswith('embeds'):
        #             optimal_alpha_l = self.actor_pop[best_index].params[k]['embedding'][task_id]
        #             optimal_alpha_l = np.array([optimal_alpha_l.flatten()])
        #             task_e = self.task_embeddings[task_id]
        #             # online update dictionary via CD
        #
        #             for _ in range(self.population_size):
        #                 self.dict4layers[_][k].update_dict(optimal_alpha_l, task_e)
        #                 dict_stats[k] = {
        #                     #'sim_mat': self.dict4layers[_][k]._compute_overlapping(),
        #                     'change_of_d': np.array(self.dict4layers[_][k].change_of_dict)
        #                 }
        # else:
        # for k in self.actor_pop[best_index].params.keys():
        #     if k.startswith('embeds'):
        #         dict_stats[k] = {
        #             'sim_mat': self.dict4layers[k]._compute_overlapping(),
        #             'change_of_d': 0
        #         }

        self.reset_agent(elite_index=best_index)

        if save_actor_dir is not None and save_dict_dir is not None:
            # save actor params
            self.actor_pop[best_index].save(save_actor_dir)
            # save dicts
            # for k in self.dict4layers.keys():
            #     self.dict4layers[k].save(f'{save_dict_dir}/{k}.pkl')

        return str_a
        # return dict_stats


class TaDeLL(SACLearner):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        load_dict_dir: Optional[str] = None,
        dict_configs: dict = {},
        pi_opt_configs: dict = {},
        q_opt_configs: dict = {},
        t_opt_configs: dict = {},
        actor_configs: dict = {},
        critic_configs: dict = {},
        tau: float = 0.005,
        discount: float = 0.99,
        target_update_period: int = 1,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0):
        super(TaDeLL, self).__init__(seed, observations, actions, pi_opt_configs,
              q_opt_configs, t_opt_configs, actor_configs, critic_configs, discount, 
              tau, target_update_period, target_entropy, init_temperature)

        flat_actor_params, pack_fn = ravel_pytree(unfreeze(self.actor.params))
        feature_dim = len(flat_actor_params) + 384
        self.dict_learner = OnlineDictLearnerV2(
            feature_dim,
            20,
            seed,
            None, # whether using svd dictionary initialization
            **dict_configs
        )

        if load_dict_dir is not None:
            self.dict_learner.load(load_dict_dir)

        self.pack_jit = jax.jit(pack_fn)
        self.dict_cfgs = dict_configs
        self.num_params = len(flat_actor_params)
        self.eval_actor = deepcopy(self.actor)
        self.task_params = []
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')

    def start_task(self, description: str):
        self.task_emb = self.task_encoder.encode(description)[np.newaxis]

        # set actor parameters via sparse coding of task embedding
        task_code = sparse_encode(
            self.task_emb,
            self.dict_learner.D[:, self.num_params:],
            algorithm=self.dict_cfgs['method'], 
            alpha=self.dict_cfgs['alpha'],
            check_input=False,
            positive=self.dict_cfgs['positive_code'],
            max_iter=10000)
        
        recon_params = np.dot(
            task_code, self.dict_learner.D[:, :self.num_params]
        )

        new_actor_params = self.pack_jit(recon_params.flatten())
        self.actor = self.actor.update_params(freeze(new_actor_params))

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       eval_mode: bool = False) -> jnp.ndarray:

        if eval_mode:
            rng, actions = policies.sample_actions(self.rng, self.eval_actor.apply_fn,
                                                   self.eval_actor.params, observations,
                                                   temperature)
        else:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, observations,
                                                   temperature)
        self.rng = rng

        actions = self.invalid_filter(actions)
        return actions

    def select_actor(self, task_id):
        if task_id < len(self.task_params):
            self.eval_actor = self.eval_actor.update_params(self.task_params[task_id])
        else:
            self.eval_actor = self.eval_actor.update_params(self.actor.params)

    def end_task(self, save_dict_dir: str):
        # update dictionary
        self.task_params.append(deepcopy(self.actor.params))
        flat_actor_params, _ = ravel_pytree(unfreeze(self.actor.params))
        sample = np.hstack([flat_actor_params[np.newaxis], self.task_emb])
        s_code = self.dict_learner.get_alpha(sample)
        self.dict_learner.update_dict(s_code, sample)

        # reset params for critic ,target_critic and temperature
        self.rng, key_critic, key_temp = jax.random.split(self.rng, 3)

        self.critic = utils_fn.reset_model(
            self.critic, 
            critic_net.DoubleCritic,
            self.critic_cfgs,
            [key_critic, self.dummy_o, self.dummy_a])

        self.target_critic = self.target_critic.update_params(self.critic.params)

        self.temp = utils_fn.reset_model(
            self.temp,
            temperature.Temperature,
            {'init_log_temp': self.init_temp},
            [key_temp])

        # reset optimizers
        self.actor = self.actor.reset_optimizer()
        self.critic = self.critic.reset_optimizer()
        self.temp = self.temp.reset_optimizer()

        if save_dict_dir is not None:
            # save actor params
            self.dict_learner.save(save_dict_dir)

