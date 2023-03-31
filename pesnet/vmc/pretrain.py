import functools
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pesnet.utils.jax_utils import pmean
from pesnet.systems.scf import Scf
from pesnet.utils.typing import McmcFn, OrbitalFunction, PretrainStep


def eval_orbitals(scf_approx: List[Scf], electrons: jax.Array, spins: Tuple[int, int]) -> Tuple[jax.Array, jax.Array]:
    """Returns the molecular orbitals of Hartree Fock calculations.

    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons (jax.Array): (..., H, N, 3)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        Tuple[jax.Array, jax.Array]: [(..., H, spin_up, spin_up), (..., H, spin_down, spin_down)] molecular orbitals
    """
    if not isinstance(scf_approx, (list, tuple)):
        scf_approx = [scf_approx]
    n_scf = len(scf_approx)
    N = sum(spins)
    
    assert electrons.shape[-3] == n_scf, f'#SCF: {n_scf}, Electrons: {electrons.shape}'
    leading_dims = electrons.shape[:-3]
    electrons = electrons.reshape(-1, n_scf, N, 3)
    electrons = electrons.swapaxes(0, 1).reshape(n_scf, -1, 3) # (n_scf, batch*nelec, 3)

    # (2, n_scf, batch*nelec, nbasis)
    mos = np.array([
        scf.eval_molecular_orbitals(e)
        for scf, e in zip(scf_approx, np.asarray(electrons))
    ]).swapaxes(0, 1)
    # Reshape into (2, n_scf, batch, nelec, nbasis)
    mos = mos.reshape(2, n_scf, -1, N, mos.shape[-1])
    # Transpose to (2, batch, n_scf, nelec, nbasis)
    mos = mos.swapaxes(1, 2)

    alpha_spin = mos[0][..., :spins[0], :spins[0]]
    beta_spin = mos[1][..., spins[0]:, :spins[1]]
    # expand first dimensions again
    return (alpha_spin.reshape(*leading_dims, n_scf, spins[0], spins[0]),
        beta_spin.reshape(*leading_dims, n_scf, spins[1], spins[1]))


def make_pretrain_step(
    mcmc_step: McmcFn,
    batch_orbitals: OrbitalFunction,
    opt_update: optax.TransformUpdateFn,
    distinct_orbitals: bool = False,
    full_det: bool = False
) -> PretrainStep:
    """Returns the pretraining step function.

    Args:
        mcmc_step (Callable): Sampling function, see `mcmc.py`.
        batch_orbitals (Callable): Wave function orbital function
        opt_update (Callable): Optimizer update function
        full_det (bool, optional): Whether the network uses `full_det=True`. Defaults to False.
    """
    def pretrain_step(key, params, electrons, atoms, targets, opt_state, mcmc_width):
        def loss_fn(params, electrons, atoms, targets):
            orbitals = batch_orbitals(
                params,
                electrons,
                atoms
            )
            
            if full_det and len(targets) == 2:
                na = targets[0].shape[-2]
                full_orbitals = orbitals[0]
                orbitals = [
                    full_orbitals[..., :na, :na],
                    full_orbitals[..., na:, na:],
                ]
            k = orbitals[0].shape[-3]
            n_configs = targets[0].shape[1]
            if n_configs > 1 and distinct_orbitals:
                idx = np.linspace(0, n_configs-1, k).round().astype(np.int32)
                idx2 = np.arange(k)
                result = functools.reduce(
                    jnp.add,
                    [jnp.mean((t[idx] - o[idx, :, idx2])**2) for t, o in zip(targets, orbitals)],
                    0
                )
            else:
                result = functools.reduce(
                    jnp.add,
                    [jnp.mean((t[..., None, :, :] - o)**2) for t, o in zip(targets, orbitals)],
                    0
                )
            return pmean(result)
        val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
        loss_val, grad = val_and_grad(
            params,
            electrons,
            atoms,
            targets
        )
        grad = pmean(grad)

        updates, opt_state = opt_update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        params = pmean(params)

        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(
            subkey,
            params,
            electrons,
            atoms,
            mcmc_width
        )
        return params, electrons, opt_state, loss_val, pmove
    return pretrain_step
