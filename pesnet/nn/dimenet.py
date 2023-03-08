# Implementation of DimeNet++, adapted from PyTorch Geometric
import functools
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from jax.nn.initializers import zeros

from pesnet.nn.gnn import GlobalOut, NodeOut
from pesnet.nn import (MLP, Activation, ActivationWithGain, Embed,
                       glorot_orthogonal, residual)
from pesnet.nn.spherical_basis import (make_bessel_basis,
                                    make_positional_encoding,
                                    make_real_sph_harm)

Dense = functools.partial(
    nn.Dense,
    kernel_init=glorot_orthogonal(),
    bias_init=zeros
)
Dense_no_bias = functools.partial(Dense, use_bias=False)


class EdgeEmbedding(nn.Module):
    embedding: int
    activation: Activation
    directional: bool
    cutoff: float
    n_sph: int
    n_rad: int

    @nn.compact
    def __call__(self, rbf, charges, differences, senders, receivers):
        act = ActivationWithGain(self.activation)
        x = Embed(95, self.embedding)(charges)
        rbf = act(Dense(self.embedding)(rbf))
        if self.directional:
            directional = act(Dense(self.embedding)(make_positional_encoding(cutoff=self.cutoff, n_sph=self.n_sph, n_rad=self.n_rad)(differences)))
            return act(Dense(self.embedding)(jnp.concatenate([x[senders], x[receivers], rbf, directional], axis=-1)))
        else:
            return act(Dense(self.embedding)(jnp.concatenate([x[senders], x[receivers], rbf], axis=-1)))


class Envelope(nn.Module):
    exponent: int

    @nn.compact
    def __call__(self, x):
        x = jnp.minimum(x, 1)
        p = self.exponent + 1
        a = -(p + 1) * (p + 2) /2
        b = p * (p + 2)
        c = -p * (p + 1)/2

        x_pow_p0 = x ** (p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class ExponentialEnvelope(nn.Module):
    decay: int
    
    @nn.compact
    def __call__(self, x):
        return jnp.exp(-x*self.decay)


class BesselBasisLayer(nn.Module):
    num_radial: int
    cutoff: float
    envelope: nn.Module

    @nn.compact
    def __call__(self, x):
        dist = jnp.linalg.norm(x, axis=-1, keepdims=True)
        freq = self.param(
            'freq',
            lambda *_: jnp.arange(1, self.num_radial + 1) * jnp.pi,
            (self.num_radial,)
        )
        norm = (2/self.cutoff) ** 0.5
        return norm * self.envelope(dist) * jnp.sin(freq * dist/self.cutoff) / dist


class SphericalBasisLayer(nn.Module):
    num_spherical: int
    num_radial: int
    cutoff: float
    envelope: nn.Module

    @nn.compact
    def __call__(self, distances, senders, receivers):
        n = distances.shape[0]
        dist = jnp.linalg.norm(distances, axis=-1, keepdims=True)
        normed_d = distances/dist
        dist = dist / self.cutoff
        rbf = make_bessel_basis(self.num_spherical, self.num_radial)(dist)
        env = self.envelope(dist)
        rbf = rbf.reshape(n, np.prod(rbf.shape[1:])) * env.reshape(n, np.prod(env.shape[1:]))

        cbf = make_real_sph_harm(self.num_spherical, True)(jax.vmap(jnp.dot)(normed_d[receivers], normed_d[senders]))
        n, k = self.num_spherical, self.num_radial
        out = (rbf[senders].reshape(-1, n, k) * cbf.reshape(-1, n, 1)).reshape(-1, n*k)
        return out


class ResidualLayer(nn.Module):
    hidden_channels: int
    activation: Activation

    @nn.compact
    def __call__(self, x):
        return residual(x, MLP([self.hidden_channels]*2, activation=self.activation)(x))


class InteractionBlock(nn.Module):
    hidden_channels: int
    int_emb_size: int
    basis_emb_size: int
    num_before_skip: int
    num_after_skip: int
    activation: Activation

    @nn.compact
    def __call__(self, rbf, sbf, edges, senders, receivers):
        activation = ActivationWithGain(self.activation)

        x_ji = activation(Dense(self.hidden_channels)(edges))
        x_kj = activation(Dense(self.hidden_channels)(edges))

        rbf = Dense_no_bias(self.basis_emb_size)(rbf)
        rbf = Dense_no_bias(self.hidden_channels)(rbf)
        x_kj = x_kj * rbf

        x_kj = activation(Dense(self.int_emb_size)(x_kj))

        sbf = Dense_no_bias(self.basis_emb_size)(sbf)
        sbf = Dense_no_bias(self.int_emb_size)(sbf)
        x_kj = x_kj[senders] * sbf

        x_kj = jraph.segment_mean(x_kj, receivers, edges.shape[0])
        x_kj = activation(Dense(self.hidden_channels)(x_kj))

        h = x_ji + x_kj
        for _ in range(self.num_before_skip):
            h = ResidualLayer(self.hidden_channels, self.activation)(h)
        h = residual(edges, activation(Dense(self.hidden_channels)(h)))
        for _ in range(self.num_after_skip):
            h = ResidualLayer(self.hidden_channels, self.activation)(h)
        return h


class NodeAggregate(nn.Module):
    n_nodes: int
    out_emb_size: int

    @nn.compact
    def __call__(self, rbf, x, receivers):
        rbf = Dense_no_bias(x.shape[-1])(rbf)

        x = rbf * x
        node_x = jraph.segment_mean(x, receivers, self.n_nodes, indices_are_sorted=True)
        return Dense_no_bias(self.out_emb_size)(node_x)


class DimeNet(nn.Module):
    node_out_dims: Tuple[int, ...]
    global_out_dims: Tuple[int, ...]

    charges: Tuple[int, ...]

    directional: bool = True
    concat_before_out: bool = False

    emb_size: int = 128
    out_emb_size: int = 256
    int_emb_size: int =  64
    basis_emb_size: int = 8
    num_blocks: int = 4

    num_spherical: int = 7
    num_radial: int = 6
    num_rbf: int = 32
    cutoff: float = 10
    envelope_exponent: int = 1
    envelope_type: str = 'none'

    num_before_skip: int = 1
    num_after_skip: int = 2
    num_dense_output: int = 3

    activation: Activation = nn.silu

    @nn.compact
    def __call__(self, nuclei):
        assert self.envelope_type in ['polynomial', 'exponential', 'none']
        if self.directional:
            axes = self.variable(
                'constants',
                'axes',
                jnp.eye,
                3
            )
            nuclei = nuclei.reshape(-1, 3) @ axes.value
        n_nuclei = nuclei.shape[0]
        max_charge = max(self.charges)
        charges = jnp.array(self.charges)

        # Construct fully connected graph
        idx_i, idx_j = np.where(np.ones((n_nuclei, n_nuclei)) - np.eye(n_nuclei))
        idx_i, idx_j = jnp.array(idx_i), jnp.array(idx_j)

        differences = nuclei[idx_i] - nuclei[idx_j]
        n_edges = idx_i.size

        # Edge-edge connections
        idx_ij, idx_jk = np.where(np.ones((n_edges, n_edges)) - np.eye(n_edges))
        idx_ij, idx_jk = jnp.array(idx_ij), jnp.array(idx_jk)
        
        if self.envelope_type == 'polynomial':
            envelope = Envelope(self.envelope_exponent)
        elif self.envelope_type == 'exponential':
            envelope = ExponentialEnvelope(self.envelope_exponent)
        elif self.envelope_type == 'none':
            envelope = lambda x: jnp.ones_like(x)
        else:
            raise NotImplementedError(f"Envelope type {self.envelope_type} is not supported!")
        rbf = BesselBasisLayer(self.num_rbf, self.cutoff, envelope)(differences)
        sbf = SphericalBasisLayer(self.num_spherical, self.num_radial, self.cutoff, envelope)(differences, idx_ij, idx_jk)

        # Edge embedding
        x = EdgeEmbedding(
            self.emb_size,
            self.activation,
            self.directional,
            self.cutoff,
            self.num_spherical,
            self.num_radial)(rbf, charges, differences, idx_i, idx_j)
        xs = [x]
        for _ in range(self.num_blocks):
            x = InteractionBlock(
                self.emb_size,
                self.int_emb_size,
                self.basis_emb_size,
                self.num_before_skip,
                self.num_after_skip,
                self.activation)(rbf, sbf, x, idx_ij, idx_jk)
            xs.append(x)
        
        if self.concat_before_out:
            xs = [jnp.concatenate(xs, axis=-1)]
        else:
            xs = xs
        
        node_output = [0 for _ in self.node_out_dims]
        global_output = [0 for _ in self.global_out_dims]
        for x in xs:
            node_in = NodeAggregate(len(self.charges), self.out_emb_size)(rbf, x, idx_i)

            node_output = [
                out + NodeOut(max_charge, out_dim, self.num_dense_output, self.activation)(node_in, charges)
                for out, out_dim in zip(node_output, self.node_out_dims)
            ]

            if len(self.global_out_dims) > 0:
                global_in = jnp.mean(node_in, axis=0) 
            global_output = [
                out + GlobalOut(out_dim, self.num_dense_output, self.activation)(global_in)
                for out, out_dim in zip(global_output, self.global_out_dims)
            ]
        return node_output, global_output
