import functools
from dataclasses import field
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from pesnet.nn import (Activation, ActivationWithGain, AutoMLP, Dense,
                       Dense_no_bias, residual)


class InvariantEncoding(nn.Module):
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

        # Electron-atom distances
        r_im = electrons[:, None] - atoms[None]
        r_im_norm = jnp.linalg.norm(r_im, keepdims=True, axis=-1)
        r_im = jnp.concatenate([r_im, r_im_norm], axis=-1)
        scaling = jnp.log(1+r_im_norm)/r_im_norm
        h_one = r_im * scaling

        # Electron-electron distances
        r_ij = electrons[:, None] - electrons[None]
        r_ij_norm = jnp.linalg.norm(
            r_ij + jnp.eye(n_elec)[..., None],
            keepdims=True,
            axis=-1
        )
        scaling = jnp.log(1+r_ij_norm)/r_ij_norm
        r_ij = jnp.concatenate([r_ij, r_ij_norm * (1.0 - jnp.eye(n_elec)[..., None])], axis=-1)
        h_two = r_ij * scaling

        # Invariant electron-nuclei embedding
        nuc_embedding = self.param(
            'nuc_embedding',
            jnn.initializers.normal(1/np.sqrt(4)),
            (n_atoms, 4, self.nuclei_embedding)
        )
        nuc_bias = self.param(
            'nuc_bias',
            jnn.initializers.normal(1.0),
            (n_atoms, self.nuclei_embedding)
        )
        h_one = jnp.einsum('nmi,mio->nmo', h_one, nuc_embedding) + nuc_bias
        h_one = activation(h_one)
        h_one = h_one.mean(1)
        return h_one, h_two, r_im, r_ij


def aggregate_features(h_one, h_two, spins, absolute_spins):
    spins = np.array(spins)
    # Single input
    pair_features = jnp.stack(
        [
            jnp.mean(h, axis=1)
            for h in jnp.split(h_two, spins[:1], axis=1) 
            if h.size > 0
        ], axis=-2)
    if not absolute_spins:
        pair_features = pair_features.at[spins[0]:].set(pair_features[spins[0]:, (1, 0)])
    one_in = jnp.concatenate([h_one, pair_features.reshape(h_one.shape[0], -1)], axis=-1)

    # Global input
    up, down = jnp.split(h_one, spins[:1], axis=0)
    g_up, g_down = up.mean(0), down.mean(0)
    g_in = jnp.stack([g_up, g_down], axis=0)
    if absolute_spins:
        g_in = g_in.reshape(1, -1)
    else:
        g_in = g_in[np.array([[0, 1], [1, 0]])].reshape(2, -1)
    return one_in, g_in


class FermiLayer(nn.Module):
    spins: Tuple[int, int]
    single_out: int
    pair_out: int
    activation: Activation
    absolute_spins: bool
    update_pair_independent: bool

    @nn.compact
    def __call__(self, h_one, h_two):
        activation = ActivationWithGain(self.activation)

        # Single update
        one_in, g_in = aggregate_features(h_one, h_two, self.spins, self.absolute_spins)
        h_one_new = Dense(self.single_out)(one_in)

        # Global Contribution
        g_new = Dense_no_bias(self.single_out)(g_in)
        if not self.absolute_spins:
            g_new = g_new.repeat(np.array(self.spins), axis=0)
        h_one_new = (h_one_new + g_new) / jnp.sqrt(2.0)

        h_one_new = activation(h_one_new)
        h_one = residual(h_one, h_one_new)
        
        # Pairwise update
        if self.pair_out > 0:
            if self.update_pair_independent:
                h_two_new = Dense(self.pair_out)(h_two)
            else:
                u, d = jnp.split(h_two, self.spins[:1], axis=0)
                uu, ud = jnp.split(u, self.spins[:1], axis=1)
                du, dd = jnp.split(d, self.spins[:1], axis=1)
                same = Dense(self.pair_out)
                diff = Dense(self.pair_out)
                h_two_new = jnp.concatenate([
                    jnp.concatenate([same(uu), diff(ud)], axis=1),
                    jnp.concatenate([diff(du), same(dd)], axis=1),
                ], axis=0)
            if h_two.shape != h_two_new.shape:
                h_two = jnp.tanh(h_two_new)
            else:
                h_two = residual(h_two, activation(h_two_new))
        return h_one, h_two


class IsotropicEnvelope(nn.Module):
    charges: Tuple[int, ...]
    out_dim: int
    determinants: int
    sigma_init: float = 1
    pi_init: float = 1

    @nn.compact
    def __call__(self, x):  
        # x is of shape n_elec, n_nuclei, 4
        n_nuclei = x.shape[1]
        def sigma_init(_, shape):
            n_k = np.arange(1, 9)
            n_k = n_k.repeat(n_k**2)
            charges = np.array(self.charges)[:, None]
            return jnp.array(charges / n_k[:shape[1]])[..., None].repeat(shape[2], 2)
        sigma = self.param(
            'sigma',
            sigma_init if isinstance(self.sigma_init, str) else jnn.initializers.constant(self.sigma_init),
            (n_nuclei, self.out_dim, self.determinants)
        ).reshape(n_nuclei, -1)
        pi = self.param(
            'pi',
            jnn.initializers.constant(self.pi_init),
            (n_nuclei, self.out_dim * self.determinants)
        )
        sigma = nn.softplus(sigma)
        pi = nn.softplus(pi)
        return jnp.sum(jnp.exp(-x[..., -1:] * sigma) * pi, axis=1)


class Orbitals(nn.Module):
    spins: Tuple[int, int]
    charges: Tuple[int, ...]
    determinants: int
    full_det: bool
    share_weights: bool = False

    @nn.compact
    def __call__(self, h_one, r_im):
        # h_one is n_elec, D
        # r_im is n_elec, n_atom, 4
        kernel_init = functools.partial(jnn.initializers.variance_scaling, mode='fan_in', distribution='truncated_normal')

        # Orbital functions
        def orbital_fn(h, r, out_dim, kernel_scale=1, sigma_init=1, pi_init=1):
            n_param = out_dim * self.determinants
            # Different initialization to ensure zero off diagonals
            if isinstance(kernel_scale, slice):
                dense = Dense(
                    n_param,
                    kernel_init=lambda key, shape, dtype=jnp.float_: jnp.concatenate([
                        kernel_init(1.0)(key, shape=(*shape[:-1], self.determinants*self.spins[0]), dtype=dtype),
                        jnp.zeros((*shape[:-1],self.determinants*self.spins[1]), dtype=dtype)
                    ][kernel_scale], axis=-1)
                )
            else:
                dense = Dense(n_param, kernel_init=kernel_init(kernel_scale))
            # Actual orbital function
            return (dense(h) * IsotropicEnvelope(self.charges, out_dim, self.determinants, sigma_init=sigma_init, pi_init=pi_init)(r))\
                .reshape(-1, out_dim, self.determinants)

        # Case destinction for weight sharing 
        if self.share_weights:
            uu, dd = jnp.split(orbital_fn(h_one, r_im, max(self.spins)), self.spins[:1], axis=0)
            if self.full_det:
                ud, du = jnp.split(orbital_fn(
                    h_one,
                    r_im,
                    max(self.spins),
                    kernel_scale=0
                ), self.spins[:1], axis=0)
                orbitals = (jnp.concatenate([
                    jnp.concatenate([uu[:, :self.spins[0]], ud[:, :self.spins[1]]], axis=1),
                    jnp.concatenate([du[:, :self.spins[0]], dd[:, :self.spins[1]]], axis=1),
                ], axis=0),)
            else:
                orbitals = (uu[:, :self.spins[0]], dd[:, :self.spins[1]])
        else:
            h_by_spin = jnp.split(h_one, self.spins[:1], axis=0)
            r_im_by_spin = jnp.split(r_im, self.spins[:1], axis=0)
            orbitals = tuple(
                orbital_fn(h, r, d, kernel_scale=i)
                for h, r, d, i in zip(
                    h_by_spin,
                    r_im_by_spin,
                    (sum(self.spins),)*2 if self.full_det else self.spins,
                    (slice(None), slice(None, None, -1)) if self.full_det else (1.0,)*2
                )
            )
            if self.full_det:
                orbitals = (jnp.concatenate(orbitals, axis=0),)
        return tuple(o.transpose(2, 0, 1) for o in orbitals)


class LogSumDet(nn.Module):
    @nn.compact
    def __call__(self, xs):
        # Special case for 1x1 matrices
        # Here we avoid going into the log domain
        det1 = functools.reduce(
            lambda a, b: a*b,
            [x.reshape(-1) for x in xs if x.shape[-1] == 1],
            jnp.ones(())
        )

        sign_in, logdet = functools.reduce(
            lambda a, b: (a[0]*b[0], a[1]+b[1]),
            [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
            (jnp.ones(()), jnp.zeros(()))
        )

        maxlogdet = jax.lax.stop_gradient(jnp.max(logdet))
        det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

        w = self.param(
            'w',
            jnn.initializers.ones,
            det.shape
        )
        result = jnp.vdot(w, det)

        sign_out = jnp.sign(result)
        log_out = jnp.log(jnp.abs(result)) + maxlogdet
        return sign_out, log_out


class Jastrow(nn.Module):
    spins: tuple[int, int]
    
    @nn.compact
    def __call__(self, r_ij):
        a_par_w, a_anti_w = self.param(
            'weight',
            jnn.initializers.constant(1e-2),
            (2,)
        )
        a_par, a_anti = self.param(
            'alpha',
            jnn.initializers.ones,
            (2,)
        )
        r_ij = r_ij[..., -1]
        uu, ud, du, dd = [
            s
            for split in jnp.split(r_ij, self.spins[:1], axis=0)
            for s in jnp.split(split, self.spins[:1], axis=1)
        ]
        same = jnp.concatenate([uu.reshape(-1), dd.reshape(-1)])
        diff = jnp.concatenate([ud.reshape(-1), du.reshape(-1)])
        result = -(1/4) * a_par_w * (a_par**2 / (a_par + same)).sum()
        result += -(1/2) * a_anti_w * (a_anti**2 / (a_anti + diff)).sum()
        return result



class FermiNet(nn.Module):
    charges: Tuple[int, ...]
    spins: Tuple[int, int]
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    determinants: int = 16
    full_det: bool = False
    input_config: dict = field(default_factory=lambda *_: {
        'activation': 'tanh',
        'nuclei_embedding': 64,
        'out_dim': 64,
        'mlp_depth': 2
    })
    jastrow_config: Optional[dict] = field(default_factory=lambda *_: {
        'activation': 'silu',
        'n_layers': 3
    })
    activation: Activation = nn.silu
    absolute_spins: bool = False
    update_pair_independent: bool = False

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
                activation=self.activation,
                absolute_spins=self.absolute_spins,
                update_pair_independent=self.update_pair_independent
            )
            for single, pair in hidden_dims
        ]

        self.to_orbitals = Orbitals(self.spins, self.charges, self.determinants, self.full_det, not self.absolute_spins)

        self.logsumdet = LogSumDet()
        if self.jastrow_config is not None:
            self.jastrow = AutoMLP(1, **self.jastrow_config)
            self.jastrow_weight = self.param(
                'jastrow_weight',
                jnn.initializers.zeros,
                ()
            )
        self.cusp_jastorw = Jastrow(self.spins)

    def encode(self, electrons, atoms):
        # Prepare input
        atoms = atoms.reshape(-1, 3) @ self.axes.value
        electrons = electrons.reshape(-1, 3) @ self.axes.value
        h_one, h_two, r_im, r_ij = self.input_construction(electrons, atoms)

        # Fermi interaction
        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two)

        return h_one, r_im, r_ij

    def orbitals(self, electrons, atoms):
        h_one, r_im, _ = self.encode(electrons, atoms)
        return self.to_orbitals(h_one, r_im)

    def signed(self, electrons, atoms):
        # Compute orbitals
        h_one, r_im, r_ij = self.encode(electrons, atoms)
        orbitals = self.to_orbitals(h_one, r_im)
        # Compute log det
        sign, log_psi = self.logsumdet(orbitals)

        # Optional jastrow factor
        if self.jastrow_config is not None:
            log_psi += self.jastrow(h_one).mean() * self.jastrow_weight
        log_psi += self.cusp_jastorw(r_ij)

        return sign, log_psi

    def __call__(self, electrons, atoms):
        return self.signed(electrons, atoms)[1]
