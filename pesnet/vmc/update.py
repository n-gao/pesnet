from enum import Enum
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey

from pesnet.utils.jax_utils import pgather, pmean_if_pmap
from pesnet.utils.jnp_utils import tree_dot
from pesnet.utils.typing import (Array, EnergyFn, GradientFn, McmcFn,
                                 NaturalGradientPreconditioner, VMCState,
                                 VMCTempResult, VMCTrainingStep, WaveFunction)


class ClipStatistic(Enum):
    MEDIAN = 'median'
    MEAN = 'mean'


def local_energy_diff(
        e_loc: jax.Array,
        clip_local_energy: float,
        stat: str | ClipStatistic
) -> jax.Array:
    stat = ClipStatistic(stat)
    if stat is ClipStatistic.MEAN:
        stat_fn = jnp.mean
    elif stat is ClipStatistic.MEDIAN:
        stat_fn = jnp.median
    else:
        raise ValueError()
    if clip_local_energy > 0.0:
        full_e = pgather(e_loc)
        clip_center = stat_fn(full_e, axis=(0, 1))
        mad = jnp.abs(full_e - clip_center).mean(axis=(0, 1))
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center - max_dev, clip_center+max_dev)
    center = pmean_if_pmap(jnp.mean(e_loc, axis=0, keepdims=True))
    e_loc -= center
    return e_loc


def make_gradient_fn(
    network: WaveFunction,
    natgrad_precond: NaturalGradientPreconditioner,
    clip_local_energy: float,
    clip_stat: str = ClipStatistic.MEDIAN
) -> GradientFn:
    @jax.jit
    def loss_and_grad(params, electrons, atoms, e_l, natgrad_state):
        aux_data = {}

        n = e_l.size

        diff = local_energy_diff(e_l, clip_local_energy, clip_stat)

        def loss(p):
            log_psi = network(p, electrons, atoms)
            return jnp.vdot(log_psi, diff) / n

        grad = jax.grad(loss)(params)
        grad = pmean_if_pmap(grad)

        natgrad, natgrad_state, damping = natgrad_precond(
            params,
            1/n,
            (electrons, atoms),
            {'e_l': e_l},
            grad,
            natgrad_state
        )
        aux_data['grad_norm'] = {
            'final': jnp.sqrt(tree_dot(natgrad, natgrad)),
            'euclidean': jnp.sqrt(tree_dot(grad, grad))
        }
        aux_data['damping'] = damping
        return natgrad, natgrad_state, aux_data
    return loss_and_grad


def make_training_step(
        mcmc_step: McmcFn,
        grad_fn: GradientFn,
        el_fn: EnergyFn,
        opt_update: optax.TransformUpdateFn) -> VMCTrainingStep:
    """Creates the training step.

    Args:
        mcmc_step (Callable): MCMC sampling method.
        val_and_grad (Callable): Function that returns the loss and gradient.
        el_fn (Callable): Local Energy function
        opt_update (Callable): Otimizer update function

    Returns:
        Callable: training step function
    """
    def step(key: PRNGKey, state: VMCState, electrons: Array, atoms: Array, mcmc_width: Array):
        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(
            subkey,
            state.params,
            electrons,
            atoms,
            mcmc_width
        )

        e_l = el_fn(state.params, electrons, atoms)

        grads, natgrad_state, aux_data = grad_fn(
            state.params,
            electrons,
            atoms,
            e_l,
            state.natgrad_state
        )
        # Compute variance per configuration
        E = pmean_if_pmap(jnp.mean(e_l, axis=0))
        E_var = pmean_if_pmap(((e_l - E) ** 2).mean(0))

        # Optimizer step
        updates, opt_state = opt_update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        return VMCTempResult(electrons, e_l, E, E_var, pmove), VMCState(params, opt_state, natgrad_state), aux_data
    return step


def init_electrons(
        key: PRNGKey,
        atom_positions: Array,
        charges: Array,
        spins: tuple[int, int],
        batch_size: int) -> Array:
    """Initializes electron positions by normal distributions
    around the nuclei. For heavy atoms this function tries to
    match the number of spin up and spin down electrons per 
    nuclei. Otherwise, it could happen that one atom has only
    spin up and one only spin down electrons.

    Args:
        key (jax.Array): jax.random.PRNGKey
        atom_positions (jax.Array): (..., M, 3)
        charges (jax.Array): (M)
        spins (Tuple[int, int]): (spin_up, spin_down)
        batch_size (int): total batch size

    Returns:
        jax.Array: (..., batch_size//len(...), N, 3)
    """
    n_devices = jax.device_count()
    n_electrons = sum(spins)
    if atom_positions.ndim > 2:
        config_shape = atom_positions.shape[:-2]
        n_configs = np.prod(config_shape)
        atom_positions = atom_positions.reshape(
            -1, atom_positions.shape[-2], atom_positions.shape[-1])
    else:
        config_shape = ()
        n_configs = 1
        atom_positions = atom_positions[None]

    electrons = []
    # Ensure that the batch size is divisble by the number of devices
    batch_size_per_config = batch_size // n_configs // n_devices * n_devices
    for config in range(n_configs):
        a_p = atom_positions[config]
        if sum(charges) != n_electrons:
            p = jnp.array(charges)/sum(charges)
            key, subkey = jax.random.split(key)
            atom_idx = jax.random.choice(subkey,
                a_p.shape[0], shape=(batch_size_per_config, n_electrons), replace=True, p=p)
        else:
            charges = np.array(charges)
            atom_idx = np.zeros((batch_size_per_config, n_electrons), dtype=np.int32)
            nalpha = np.ceil(charges/2).astype(jnp.int32)
            nbeta = np.floor(charges/2).astype(jnp.int32)
            assert sum(nalpha) + sum(nbeta) == sum(spins)
            while (sum(nalpha), sum(nbeta)) != spins:
                key, subkey = jax.random.split(key)
                i = jax.random.randint(subkey, (), 0, len(nalpha))
                a, b = nalpha[i], nbeta[i]
                nalpha[i], nbeta[i] = b, a
            alpha_idx = jnp.array([
                i for i in range(len(nalpha))
                for _ in range(nalpha[i])
            ])
            beta_idx = jnp.array([
                i for i in range(len(nbeta))
                for _ in range(nbeta[i])
            ])
            atom_idx[:] = jnp.concatenate([alpha_idx, beta_idx])
            
        electrons.append(a_p[atom_idx].reshape(batch_size_per_config, n_electrons*3))
    electrons = jnp.array(electrons).swapaxes(0, 1).reshape(batch_size_per_config, *config_shape, n_electrons, 3)
    key, subkey = jax.random.split(key)
    return electrons + jax.random.normal(subkey, shape=electrons.shape)


def coordinate_transform(old_atoms: Array, new_atoms: Array, electrons: Array) -> Array:
    # old_atoms: M x 3
    # new_atoms: M x 3
    # electrons: N x 3
    diff = old_atoms[:, None] - electrons
    dist = jnp.linalg.norm(diff, axis=-1)
    closest_nuc = jnp.argmin(dist, axis=0)

    nuc_delta = new_atoms - old_atoms
    return electrons + nuc_delta[closest_nuc]
