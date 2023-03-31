import functools
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jraph
import numpy as np

from pesnet.nn import (Activation, ActivationWithGain, AutoMLP, Dense,
                       Dense_no_bias, Embed, named, residual)


class MessagePassing(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation

    @nn.compact
    def __call__(self, n_embed, e_embed, senders, receivers):
        inp_features = jnp.concatenate([
            n_embed[senders],
            n_embed[receivers]
        ], axis=-1)
        inp_features = AutoMLP(self.out_dim, self.n_layers,
                activation=self.activation)(inp_features)
        inp_features *= Dense_no_bias(inp_features.shape[-1])(e_embed)
        return jraph.segment_mean(
            inp_features,
            senders,
            num_segments=n_embed.shape[0],
            indices_are_sorted=True
        )


class Update(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation

    @nn.compact
    def __call__(self, n_embed, msg):
        inp_features = jnp.concatenate([n_embed, msg], axis=-1)
        return residual(n_embed, AutoMLP(self.out_dim, self.n_layers, activation=self.activation)(inp_features))


class NodeOut(nn.Module):
    max_charge: int
    out_dim: int
    depth: int
    activation: Activation

    @nn.compact
    def __call__(self, x, charges):
        out_bias = Embed(self.max_charge, self.out_dim)(charges)
        return AutoMLP(self.out_dim, self.depth, self.activation, final_bias=False)(x) + out_bias


GlobalOut = functools.partial(named, 'GlobalOut', AutoMLP)


class DenseEdgeEmbedding(nn.Module):
    out_dim: int
    activation: str
    sigma_init: float

    @nn.compact
    def __call__(self, edges):
        sigma = self.param(
            'sigma',
            lambda key, shape, dtype=jnp.float32: jax.random.normal(key, shape, dtype) + self.sigma_init,
            (self.out_dim,)
        )
        env = Dense_no_bias(self.out_dim)(jnp.exp(-(edges[..., -1:]/sigma)**2))
        result = Dense(self.out_dim, bias_init=jnn.initializers.normal(2.0))(edges)
        result = ActivationWithGain(self.activation)(result)
        result = nn.LayerNorm()(result)
        return result * env


class GNN(nn.Module):
    charges: Tuple[int, ...]
    node_out_dims: Tuple[int, ...]
    global_out_dims: Tuple[int, ...]
    layers: Sequence[Tuple[int, int]] = ((32, 64), (32, 64))
    embedding_dim: int = 32
    msg_mlp_depth: int = 2
    update_mlp_depth: int = 2
    out_mlp_depth: int = 2
    rbf_dim: int = 32
    rbf_cutoff: float = 10.0
    aggregate_before_out: bool = True
    activation: Activation = nn.silu

    @nn.compact
    def __call__(self, nuclei):
        axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        nuclei = nuclei.reshape(-1, 3) @ axes.value
        n_nuclei = nuclei.shape[0]
        max_charge = max(self.charges)+1
        charges = jnp.array(self.charges)

        # Construct fully connected graph
        senders, receivers = np.where(np.ones((n_nuclei, n_nuclei)) - np.eye(n_nuclei))
        senders, receivers = jnp.array(senders), jnp.array(receivers)

        # Edge embedding
        edges = nuclei[senders] - nuclei[receivers]
        edges = jnp.concatenate([
            edges,
            jnp.linalg.norm(edges, axis=-1, keepdims=True)
        ], -1)
        e_embed = DenseEdgeEmbedding(self.rbf_dim, self.activation, self.rbf_cutoff)(edges)

        # Node embedding
        n_embed = Embed(max_charge, self.embedding_dim)(charges)

        # Message passing and update
        embeddings = [n_embed]
        for layer in self.layers:
            msg = MessagePassing(layer[0], self.msg_mlp_depth, activation=self.activation)(
                n_embed, e_embed, senders, receivers)
            n_embed = Update(layer[1], self.update_mlp_depth,
                             activation=self.activation)(n_embed, msg)
            embeddings.append(n_embed)

        # Output
        if self.aggregate_before_out:
            n_embed = jnp.concatenate(embeddings, axis=-1)
        # Node
        node_output = [
            NodeOut(max_charge, out, self.out_mlp_depth,
                    self.activation)(n_embed, charges)
            for out in self.node_out_dims
        ]
        # Global
        if len(self.global_out_dims) > 0:
            global_inp = jnp.mean(n_embed, axis=0)
        global_output = [
            GlobalOut(out, self.out_mlp_depth,
                  activation=self.activation)(global_inp)
            for out in self.global_out_dims
        ]
        return node_output, global_output


class GNNPlaceholder(nn.Module):
    @nn.compact
    def __call__(self, nuclei):
        return [], []
