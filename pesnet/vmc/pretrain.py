import functools
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from pesnet.utils.jax_utils import pmean
from pesnet.nn import MLP, ParamTree
from pesnet.systems.scf import Scf


def eval_orbitals(scf_approx: List[Scf], electrons: jax.Array, spins: Tuple[int, int]) -> Tuple[jax.Array, jax.Array]:
    """Returns the molecular orbitals of Hartree Fock calculations.

    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons ([type]): (H, B, N, 3)
        spins ([type]): (spin_up, spin_down)

    Returns:
        Tuple[jax.Array, jax.Array]: [(H, B, spin_up, spin_up), (H, B, spin_down, spin_down)] molecular orbitals
    """
    if isinstance(scf_approx, (list, tuple)):
        n_scf = len(scf_approx)
    else:
        n_scf = 1
        scf_approx = [scf_approx]
    leading_dims = electrons.shape[:-1]
    electrons = electrons.reshape([n_scf, -1, 3])  # (n_scf, batch*nelec, 3)
    # (2, n_scf, batch*nelec, nbasis)
    mos = np.array([
        scf.eval_molecular_orbitals(e)
        for scf, e in zip(scf_approx, np.asarray(electrons))
    ]).swapaxes(0, 1)
    # Reshape into (2, n_scf, batch, nelec, nbasis)
    mos = mos.reshape(2, *leading_dims, sum(spins), -1)

    alpha_spin = mos[0][..., :spins[0], :spins[0]]
    beta_spin = mos[1][..., spins[0]:, :spins[1]]
    return alpha_spin, beta_spin


def eval_slater(scf_approx: List[Scf], electrons: jax.Array, spins: Tuple[int, int]) -> Tuple[jax.Array, jax.Array]:
    """Evaluate the Slater determinants of Hartree Fock calculations

    Args:
        scf_approx (List[Scf]): Hartree Fock solutions
        electrons (jax.Array): (H, B, N, 3)
        spins (Tuple[int, int]): (spin_up, spin_down)

    Returns:
        Tuple[jax.Array, jax.Array]: (sign, log_psi)
    """
    matrices = eval_orbitals(scf_approx, electrons, spins)
    slogdets = [np.linalg.slogdet(elem) for elem in matrices]
    signs = np.array([elem[0] for elem in slogdets])
    log_wfs = np.array([elem[1] for elem in slogdets])
    log_slater = np.sum(log_wfs, axis=0)
    sign = np.prod(signs, axis=0)
    return sign, log_slater


def make_pretrain_step(
    mcmc_step: Callable,
    batch_orbitals: Callable,
    batch_get_params: ParamTree,
    opt_update: Callable,
    distinct_orbitals: bool = False,
    full_det: bool = False
):
    """Returns the pretraining step function.

    Args:
        mcmc_step (Callable): Sampling function, see `mcmc.py`.
        batch_orbitals (Callable): Wave function orbital function
        batch_get_params (ParamTree): wave function parameters
        opt_update (Callable): Optimizer update function
        full_det (bool, optional): Whether the network uses `full_det=True`. Defaults to False.
    """
    def pretrain_step(t, params, electrons, atoms, targets, opt_state, key, mcmc_width, gpu_idx):
        def loss_fn(params, electrons, atoms, targets):
            fermi_params = batch_get_params(params, atoms)
            orbitals = batch_orbitals(
                fermi_params,
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
            n_devices = jax.device_count()
            configs_per_device = targets[0].shape[0]
            if (n_devices*configs_per_device) > 1 and distinct_orbitals:
                assert k % n_devices == 0
                k_by_dev = k // n_devices
                offset = gpu_idx * k_by_dev
                idx = np.around(
                    np.linspace(
                        0,
                        configs_per_device-1,
                        k_by_dev
                    )
                ).astype(np.int32)
                idx2 = np.arange(k_by_dev) + offset
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
            params,
            electrons,
            atoms,
            subkey,
            mcmc_width
        )
        return params, electrons, opt_state, loss_val, pmove
    return pretrain_step
