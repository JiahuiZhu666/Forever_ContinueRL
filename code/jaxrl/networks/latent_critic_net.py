"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP, activation_fn


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)



def AvgL1Norm(x, eps=1e-8):
    return x / jnp.abs(x).mean(-1, keepdims=True).clip(min=eps)





class Critic(nn.Module):
    hidden_dims: Sequence[int]
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 s_z: jnp.ndarray,
                 s_a_z: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions, s_z, s_a_z], -1)
        critic = MLP(
            (*self.hidden_dims, 1), 
            activations=activation_fn(self.name_activation),
            use_layer_norm=self.use_layer_norm,
            activate_final=False)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    name_activation: str = 'leaky_relu'
    use_layer_norm: bool = True
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, s_z, s_a_z):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        name_activation=self.name_activation,
                        use_layer_norm=self.use_layer_norm)(states, actions, s_z, s_a_z)
        return qs



# Define Latent_Encoder as a Flax model
class LatentEncoder(nn.Module):
    latent_dim: int
    name_activation: str = 'elu'
    use_Avg: bool = True
    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        z_encoder = MLP(
            (256, self.latent_dim),
            activations=activation_fn(self.name_activation),
            activate_final=False)(observations)

        if self.use_Avg:
            return AvgL1Norm(z_encoder)
        else:
            return z_encoder


class LatentPredict(nn.Module):
    latent_dim: int
    name_activation: str = 'elu'

    @nn.compact
    def __call__(self, latent_representation: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([latent_representation, actions], -1)
        z_encoder = MLP(
            (256,  self.latent_dim),
            activations=activation_fn(self.name_activation),
            activate_final=False)(inputs)

        return z_encoder


class LatentPredict_Reward(nn.Module):
    latent_dim: int
    name_activation: str = 'elu'

    @nn.compact
    def __call__(self, latent_representation: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([latent_representation, actions], -1)
        reward_pre = MLP(
            (256, 1),
            activations=activation_fn(self.name_activation),
            activate_final=False)(inputs)

        return reward_pre
