from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pesnet.jax_utils import pmean
from pesnet.nn import MLP, ParamTree
from pesnet.systems.scf import Scf


def eval_orbitals(scf_approx: List[Scf], electrons: jnp.ndarray, spins: Tuple[int, int], signs: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the molecular orbitals of Hartree Fock calculations.

    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons ([type]): (H, B, N, 3)
        spins ([type]): (spin_up, spin_down)

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: [(H, B, spin_up, spin_up), (H, B, spin_down, spin_down)] molecular orbitals
    """
    if isinstance(scf_approx, (list, tuple)):
        n_scf = len(scf_approx)
    else:
        n_scf = 1
        scf_approx = [scf_approx]
    leading_dims = electrons.shape[:-1]
    electrons = electrons.reshape([n_scf, -1, 3])  # (batch*nelec, 3)
    # (batch*nelec, nbasis), (batch*nelec, nbasis)
    mos = [scf.eval_molecular_orbitals(e)
           for scf, e in zip(scf_approx, electrons)]
    mos = (np.stack([mo[0] for mo in mos], axis=0),
           np.stack([mo[1] for mo in mos], axis=0))
    # Reshape into (batch, nelec, nbasis) for each spin channel
    mos = [mo.reshape(leading_dims + (sum(spins), -1)) for mo in mos]

    alpha_spin = mos[0][..., :spins[0], :spins[0]]
    beta_spin = mos[1][..., spins[0]:, :spins[1]]

    # Adjust signs
    if signs is not None:
        alpha_signs, beta_signs = signs
        alpha_spin[..., 0] *= alpha_signs[..., None, None]
        beta_spin[..., 0] *= beta_signs[..., None, None]
    return alpha_spin, beta_spin


def make_pretrain_step(
    mcmc_step: Callable,
    batch_orbitals: Callable,
    batch_get_params: ParamTree,
    opt_update: Callable,
    full_det: bool = False,
    train_gnn: bool = False
):
    """Returns the pretraining step function.

    Args:
        mcmc_step (Callable): Sampling function, see `mcmc.py`.
        batch_orbitals (Callable): Wave function orbital function
        batch_get_params (ParamTree): wave function parameters
        opt_update (Callable): Optimizer update function
        full_det (bool, optional): Whether the network uses `full_det=True`. Defaults to False.
        train_gnn (bool, optional): Whether to train the GNN as well. Defaults to False.
    """
    def pretrain_step(params, electrons, atoms, targets, opt_state, key, mcmc_width):
        def loss_fn(params, electrons, atoms, targets):
            fermi_params = batch_get_params(params, atoms)
            orbitals = batch_orbitals(
                fermi_params,
                electrons,
                atoms
            )
            if full_det and len(targets) == 2:
                leading_dims = targets[0].shape[:-2]
                na = targets[0].shape[-2]
                nb = targets[1].shape[-1]
                targets = [jnp.concatenate(
                    (jnp.concatenate((targets[0], jnp.zeros((*leading_dims, na, nb))), axis=-1),
                     jnp.concatenate((jnp.zeros((*leading_dims, nb, na)), targets[1]), axis=-1)),
                    axis=-2)]
            k = orbitals[0].shape[-3]
            n_devices = jax.device_count()
            configs_per_device = targets[0].shape[0]
            assert k % n_devices == 0
            if configs_per_device > 1:
                idx = np.around(
                    np.linspace(
                        0,
                        configs_per_device-1,
                        k
                    )
                ).astype(np.int32)
                idx2 = np.arange(k)
                result = jnp.array([
                    jnp.mean((t[idx] - o[idx, :, idx2])**2) for t, o in zip(targets, orbitals)
                ]).sum()
            else:
                result = jnp.array([
                    jnp.mean((t[..., None, :, :] - o)**2) for t, o in zip(targets, orbitals)
                ]).sum()
            return pmean(result)

        val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
        loss_val, grad = val_and_grad(
            params,
            electrons,
            atoms,
            targets
        )
        grad = pmean(grad)

        # Rescale GNN params if a MetaGNN is  used
        if 'params' in grad['gnn']:
            gnn_grad = grad['gnn']
            node_out_grad = [
                (k, gnn_grad['params'][k]['Embed_0']['embedding'])
                for k in gnn_grad['params'].keys()
                if 'NodeOut' in k
            ]
            global_out_grad = [
                (k, MLP.extract_final_linear(gnn_grad['params'][k])['bias'])
                for k in gnn_grad['params'].keys()
                if 'GlobalOut' in k
            ]

            scaling = 0.1 if train_gnn else 0

            # Rescale GNN gradients
            gnn_grad['params'] = jax.tree_map(
                lambda x: scaling * x, gnn_grad['params'])

            # Reset final biases
            for k, val in node_out_grad:
                gnn_grad['params'][k]['Embed_0']['embedding'] = val
            for k, val in global_out_grad:
                MLP.extract_final_linear(gnn_grad['params'][k])['bias'] = val

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


def determine_hf_signs(scfs: List[Scf], electrons: jnp.ndarray, spins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n_scfs = len(scfs)
    # Compute orbitals
    # We use some samples from each configuration for this
    samples = np.asarray(electrons).reshape(-1, electrons.shape[-1])
    up_down_orbitals = eval_orbitals(scfs, np.broadcast_to(
        samples, (*electrons.shape[:2], *samples.shape)), spins)
    # Compute sign of psi
    result = []
    for orbitals in up_down_orbitals:
        signs, ld = np.linalg.slogdet(orbitals)
        signs, ld = signs.reshape(n_scfs, -1), ld.reshape(n_scfs, -1)
        dets = np.exp(ld)
        dets /= dets.sum(-1, keepdims=True)
        base = signs[0]
        result.append(np.array([(np.vdot(base == s, d) >= 0.5)*2-1 for s,
                      d in zip(signs, dets)]).reshape(*orbitals.shape[:-3]))
    return result
