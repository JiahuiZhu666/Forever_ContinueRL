from typing import Dict, List

import gym
import numpy as np
import jax.numpy as jnp


def evaluate(agent, env: gym.Env, num_episodes: int, with_task_embed=False, task_i=None) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            if with_task_embed:
                action = agent.sample_a_with_task_embed(observation[np.newaxis], temperature=0.0)
            else:
                if task_i is None:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0.0)
                else:
                    action = agent.sample_actions(observation[np.newaxis], task_i, temperature=0.0)
            observation, _, done, info = env.step(action)
        for k in stats.keys():
            stats[k].append(info['episode'][k])

        if 'success' in info:
            if successes is None:
                successes = 0.0
            successes += info['success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats


import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any
import gym
import multiprocessing
from functools import partial

# def evaluate_single_env(
#     task_i: int,
#     env: gym.Env,
#     agent,
#     select_index,
#     num_episodes: int,
#     naive_sac: bool = False,
#     tadell: bool = False
# ) -> Dict[str, float]:
#     """
#     Evaluate a single environment and return statistics across multiple episodes.
#     """
#     list_log_keys = ['return']
#     stats = {f'{task_i}-{env.name}/{k}': [] for k in list_log_keys}
#     successes = None
#
#     # When using tadell, first tell the agent which actor to select
#     if tadell:
#         agent.select_actor(task_i)
#
#     for _ in range(num_episodes):
#         observation, done = env.reset(), False
#         info = {}
#
#         while not done:
#             # Choose the sample_actions call appropriate to the configuration
#             if naive_sac:
#                 action = agent.sample_actions(observation[np.newaxis], temperature=0)
#             elif tadell:
#                 action = agent.sample_actions(observation[np.newaxis], temperature=0, eval_mode=True)
#             else:
#                 action = agent.sample_actions(select_index, observation[np.newaxis], task_i, temperature=0)
#
#             action = np.asarray(action, dtype=np.float32).flatten()
#             observation, _, done, info = env.step(action)
#
#         # Collect logs
#         for k in list_log_keys:
#             stats[f'{task_i}-{env.name}/{k}'].append(info['episode'][k])
#
#         # Track the success rate if info contains success
#         if 'success' in info:
#             if successes is None:
#                 successes = 0.0
#             successes += info['success']
#
#     # Postprocess the results, including averaging episode arrays
#     for k in list_log_keys:
#         stats[f'{task_i}-{env.name}/{k}'] = float(np.mean(stats[f'{task_i}-{env.name}/{k}']))
#
#     if successes is not None:
#         stats[f'{task_i}-{env.name}/success'] = float(successes / num_episodes)
#
#     return stats
#
#
# def evaluate_cl(
#     select_index,
#     agent,
#     envs: List[gym.Env],
#     num_episodes: int,
#     naive_sac=False,
#     tadell=False
# ) -> Dict[str, float]:
#     """
#     Evaluate multiple environments in parallel and aggregate their statistics.
#     """
#     # Create a process pool; omit processes to use all CPU cores (the default is os.cpu_count())
#     pool = multiprocessing.Pool(processes=len(envs))
#
#     # Build the argument list for each environment
#     # Passing agent directly may cause pickling issues; initialize it inside each process if necessary.
#     tasks = []
#     for task_i, env in enumerate(envs):
#         tasks.append((task_i, env, agent, select_index, num_episodes, naive_sac, tadell))
#
#     # Use starmap to evaluate each environment in parallel
#     results = pool.starmap(evaluate_single_env, tasks)
#     pool.close()
#     pool.join()
#
#     # Aggregate statistics across all environments
#     stats_agg: Dict[str, Any] = {}
#     sum_return = 0.0
#     sum_success = 0.0
#
#     for env_stats in results:
#         for k, v in env_stats.items():
#             stats_agg[k] = v
#
#     # Assume keys use the names "task_i-env_name/return" and "task_i-env_name/success"
#     num_envs = len(envs)
#     for k, v in stats_agg.items():
#         if k.endswith('/return'):
#             sum_return += v
#         elif k.endswith('/success'):
#             sum_success += v
#
#     stats_agg['avg_return'] = sum_return / num_envs
#     stats_agg['test/deterministic/average_success'] = sum_success / num_envs
#
#     return stats_agg



def evaluate_cl(select_index, agent, envs: List[gym.Env], num_episodes: int, naive_sac=False, tadell=False) -> Dict[str, float]:
    stats = {}
    sum_return = 0.0
    sum_success = 0.0
    list_log_keys = ['return']

    # dummy inputs
    # dummy_obs = jnp.ones((128, 12))
    top_ten_success = 0.0
    top_ten_return = 0.0
    for task_i, env in enumerate(envs):
        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = []
        successes = None

        if tadell:
            agent.select_actor(task_i)

        for _ in range(num_episodes):
            observation, done = env.reset(), False
            while not done:

                if naive_sac:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0)
                    action = np.asarray(action, dtype=np.float32).flatten()
                elif tadell:
                    action = agent.sample_actions(observation[np.newaxis], temperature=0, eval_mode=True)
                    action = np.asarray(action, dtype=np.float32).flatten()
                else:
                    action = agent.sample_actions(select_index, observation[np.newaxis], task_i, temperature=0)
                    action = np.asarray(action, dtype=np.float32).flatten()

                observation, _, done, info = env.step(action)

            for k in list_log_keys:
                stats[f'{task_i}-{env.name}/{k}'].append(info['episode'][k])

            if 'success' in info:
                if successes is None:
                    successes = 0.0
                successes += info['success']

        for k in list_log_keys:
            stats[f'{task_i}-{env.name}/{k}'] = np.mean(stats[f'{task_i}-{env.name}/{k}'])

        if successes is not None:
            stats[f'{task_i}-{env.name}/success'] = successes / num_episodes
            sum_success += stats[f'{task_i}-{env.name}/success']
            if task_i < 10:
                top_ten_success += stats[f'{task_i}-{env.name}/success']


        sum_return += stats[f'{task_i}-{env.name}/return']
        if task_i < 10:
            top_ten_return += stats[f'{task_i}-{env.name}/success']
        # stats[f'{task_i}-{env.name}/check_dummy_action'] = agent.sample_actions(dummy_obs, task_i, temperature=0).mean()

    stats['avg_return'] = sum_return / len(envs)
    stats['test/deterministic/average_success'] = sum_success / len(envs)

    stats['top_10_avg_return'] = top_ten_return / 10.0
    stats['test/deterministic/top_10_average_success'] = top_ten_success / 10.0

    return stats
