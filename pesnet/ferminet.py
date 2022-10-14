import functools
from dataclasses import field
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from pesnet.nn import (Activation, ActivationWithGain, AutoMLP, residual)


# We initialize with larger variance to saturate more neurons at initiailzation
# this stabilized the optimization a bit.
Dense =  functools.partial(
    nn.Dense,
    kernel_init=jnn.initializers.variance_scaling(
        2,
        mode="fan_in",
        distribution="truncated_normal"
    ),
    bias_init=jnn.initializers.normal(np.sqrt(2))
)
Dense_no_bias = functools.partial(nn.Dense, use_bias=False)


def construct_single_features(
    h_one: jnp.ndarray,
    h_two: jnp.ndarray,
    spins: Tuple[int, int]
) -> jnp.ndarray:
    """Construct the electron specific input to the next layer.

    Args:
        h_one (jnp.ndarray): (N, single_dim)
        h_two (jnp.ndarray): (N, N, pair_dim)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        jnp.ndarray: (N, single_dim + 2*pair_dim)
    """
    h_twos = h_two.split(spins[0:1], axis=0)
    g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]
    return jnp.concatenate([h_one] + g_two, axis=1)


def construct_global_features(
    h_one: jnp.ndarray,
    spins: Tuple[int, int]
) -> jnp.ndarray:
    """Construct the global input to the next layer.

    Args:
        h_one (jnp.ndarray): (N, single_dim)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        jnp.ndarray: (single_dim)
    """
    h_ones = h_one.split(spins[0:1], axis=0)
    g_one = [jnp.mean(h, axis=0, keepdims=True)
             for h in h_ones if h.size > 0]
    return jnp.concatenate(g_one, axis=-1)


class IsotropicEnvelope(nn.Module):
    """
    Isotropic envelope as proposed by
    Spencer et al., 2020
    """
    determinants: int

    @nn.compact
    def __call__(self, x):
        # x is of shape n_elec, n_nuclei, 4
        n_elec, n_nuclei, _ = x.shape
        nparams = n_elec * self.determinants
        sigma = self.param(
            'sigma',
            jnn.initializers.ones,
            (n_nuclei, nparams)
        )
        pi = self.param(
            'pi',
            jnn.initializers.ones,
            (n_nuclei, nparams)
        )
        sigma = nn.softplus(sigma)
        pi = nn.sigmoid(pi)
        return jnp.sum(jnp.exp(-jnp.linalg.norm(x, axis=-1, keepdims=True) * sigma) * pi, axis=1)


class InvariantEncoding(nn.Module):
    """
    Creates electron and electron-electron embeddings invariant to nuclei permutations.
    """
    nuclei_embedding: int
    out_dim: int
    mlp_depth: int
    activation: Activation
    mlp_activation: Optional[Activation] = None

    @nn.compact
    def __call__(self, electrons, atoms):
        n_elec = electrons.shape[0]
        n_atoms = atoms.shape[0]
        activation = ActivationWithGain(self.activation)
        mlp_activation = self.activation if self.mlp_activation is None else self.mlp_activation

        # Electron-atom distances
        r_im = electrons[:, None] - atoms[None]
        r_im_norm = jnp.linalg.norm(r_im, keepdims=True, axis=-1)
        h_one = jnp.concatenate([r_im, r_im_norm], axis=-1)

        # Electron-electron distances
        r_ij = electrons[:, None] - electrons[None]
        r_ij_norm = jnp.linalg.norm(
            r_ij + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        ) * (1.0 - jnp.eye(n_elec)[..., None])
        h_two = jnp.concatenate([r_ij, r_ij_norm], axis=-1)

        # Invariant electron-nuclei embedding
        nuc_embedding = self.param(
            'nuc_embedding',
            jnn.initializers.normal(0.1),
            (n_atoms, self.nuclei_embedding)
        )
        h_one = Dense(self.nuclei_embedding)(h_one)
        h_one = (h_one + nuc_embedding[None])
        h_one = activation(h_one)

        h_one = residual(h_one, AutoMLP(
            self.out_dim,
            self.mlp_depth,
            scale='linear',
            activation=mlp_activation
        )(h_one)).mean(1)

        return h_one, h_two, r_im


class FermiLayer(nn.Module):
    """
    Interaction layer as proposed by
    Pfau et al. 2020
    with efficiency improvements from
    Wilson et al. 2021
    """
    spins: Tuple[int, int]
    single_out: int
    pair_out: int
    activation: Activation

    @nn.compact
    def __call__(self, h_one, h_two):
        activation = ActivationWithGain(self.activation)

        # Single update
        one_in = construct_single_features(h_one, h_two, self.spins)
        global_in = construct_global_features(h_one, self.spins)
        h_one_new = Dense(self.single_out)(one_in)
        h_one_new += Dense_no_bias(self.single_out)(global_in)
        h_one_new = activation(h_one_new/jnp.sqrt(2))
        h_one = residual(h_one, h_one_new)

        # Pairwise update
        if self.pair_out > 0:
            h_two_new = activation(Dense(self.pair_out)(h_two))
            h_two = residual(h_two, h_two_new)
        return h_one, h_two


class Orbitals(nn.Module):
    """
    Construct orbital matrices.
    Given: (N,D), (NxMx4)
    Result: (K, N_up, N_up), (K, N_down, N_down)
    """
    spins: Tuple[int, int]
    determinants: int

    @nn.compact
    def __call__(self, h_one, r_im):
        # r_im is n_elec, n_atom, 4
        h_by_spin = h_one.split(self.spins[0:1], axis=0)
        r_im_by_spin = r_im.split(self.spins[0:1], axis=0)

        orbitals = [
            Dense(self.determinants*s)(h) *
            IsotropicEnvelope(self.determinants)(r)
            for h, r, s in zip(h_by_spin, r_im_by_spin, self.spins)
            if s > 0
        ]
        return [o.reshape(s, self.determinants, s).transpose(1, 0, 2) for o, s in zip(orbitals, self.spins)]


class LogSumDet(nn.Module):
    """
    Computes \log\sum_{k=1}^K w_k \det\Phi_k^{up} \det\Phi_k^{down}
    """
    @nn.compact
    def __call__(self, xs):
        # Special case for 1x1 matrices
        # Here we avoid going into the log domain
        det1 = functools.reduce(
            lambda a, b: a*b,
            [x.reshape(-1) for x in xs if x.shape[-1] == 1],
            1
        )

        sign_in, logdet = functools.reduce(
            lambda a, b: (a[0]*b[0], a[1]+b[1]),
            [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
            (1, 0)
        )

        maxlogdet = jax.lax.stop_gradient(jnp.max(logdet))
        det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

        w = self.param(
            'w',
            jnn.initializers.ones,
            (det.size,)
        )
        result = jnp.vdot(w, det)

        sign_out = jnp.sign(result)
        log_out = jnp.log(jnp.abs(result)) + maxlogdet
        return sign_out, log_out


class FermiNet(nn.Module):
    """
    FermiNet (Pfau et al., 2020), with improvements from
    Wilson et al., 2021 and Gao and Günnemann, 2022.
    """
    n_nuclei: int
    spins: Tuple[int, int]
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    determinants: int = 16
    input_config: dict = field(default_factory=lambda *_: {
        'activation': 'tanh',
        'nuclei_embedding': 32,
        'out_dim': 32,
        'mlp_depth': 2
    })
    activation: Activation = nn.tanh

    def setup(self):
        self.axes = self.variable(
            'constants',
            'axes',
            jnp.eye,
            3
        )
        self.input_construction = InvariantEncoding(
            **self.input_config
        )
        # Do not compute an update for the last pairwise layer
        hidden_dims = [list(h) for h in self.hidden_dims]
        hidden_dims[-1][1] = 0
        self.fermi_layers = [
            FermiLayer(
                spins=self.spins,
                single_out=single,
                pair_out=pair,
                activation=self.activation
            )
            for single, pair in hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.determinants)

        self.logsumdet = LogSumDet()

    def encode(self, electrons, atoms):
        # Prepare input
        atoms = atoms.reshape(-1, 3) @ self.axes.value
        electrons = electrons.reshape(-1, 3) @ self.axes.value
        h_one, h_two, r_im = self.input_construction(electrons, atoms)

        # Fermi interaction
        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two)

        return h_one, r_im

    def orbitals(self, electrons, atoms):
        h_one, r_im = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        # Compute orbitals
        h_one, r_im = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        # Compute log det
        sign, log_psi = self.logsumdet(orbitals)

        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]
