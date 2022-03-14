import numbers
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pesnet.jax_utils import pmean_if_pmap
from pesnet.jnp_utils import tree_add, tree_dot, tree_mul, tree_sub
from pesnet.optim import cg


def local_energy_diff(
        e_loc: jnp.ndarray,
        clip_local_energy: float,
        method: str = 'ferminet'
) -> jnp.ndarray:
    """Different local energy clipping methods. Implemented is the
    one from FermiNet, PauliNet and one based on FermiNet but with
    the median instead of the mean.

    Args:
        e_loc (jnp.ndarray): Local energies
        clip_local_energy (float): Clipping range
        method (str, optional): method, one of 'ferminet', 
          'paulinet', 'median' or 'none'. Defaults to 'ferminet'.

    Returns:
        jnp.ndarray: clipped local energy
    """
    if method is not None:
        method = method.lower()
    if method == 'ferminet' or method == 'median':
        if e_loc.ndim > 1:
            def mean_fn(x): return jnp.mean(x, axis=-1, keepdims=True)
        else:
            def mean_fn(x): return pmean_if_pmap(jnp.mean(x))

        if method == 'ferminet':
            e_loc_center = mean_fn(e_loc)
        else:
            e_loc_center = jnp.median(e_loc, axis=-1, keepdims=True)
            if e_loc.ndim == 1:
                e_loc_center = pmean_if_pmap(e_loc_center)

        e_loc -= e_loc_center
        if clip_local_energy > 0.0:
            tv = mean_fn(jnp.abs(e_loc))
            max_deviation = clip_local_energy*tv
            e_loc = jnp.clip(e_loc, -max_deviation, max_deviation)
        return e_loc
    elif method == 'paulinet':
        median = jnp.median(e_loc, axis=-1)
        if e_loc.ndim == 1:
            median = pmean_if_pmap(median)
        e_loc -= median
        if clip_local_energy > 0.0:
            max_deviation = clip_local_energy * jnp.abs(e_loc).mean()
            e_loc = jnp.where(
                jnp.abs(e_loc) <= max_deviation,
                e_loc,
                jnp.sign(e_loc) * max_deviation * (1 +
                                                   jnp.log((1 + (jnp.abs(e_loc) / max_deviation) ** 2) / 2))
            )
        return e_loc
    elif method == 'none' or method is None:
        if e_loc.ndim > 1:
            def mean_fn(x): return jnp.mean(x, axis=-1, keepdims=True)
        else:
            def mean_fn(x): return pmean_if_pmap(jnp.mean(x))
        return e_loc - mean_fn(e_loc)
    else:
        raise ValueError(f'{method} is an invalid clipping mode')


def make_loss(
        batch_network,
        clip_local_energy: float,
        normalize_gradient: bool = True,
        **kwargs):
    """Creates the "loss" function. It assumes that the local energies
    have been computed in advance. This function simply means the local
    energies in the forward pass. To compute the gradient, we clip the
    local energies and center them.

    Args:
        batch_network (Callable): Network that takes a batch of electrons as input.
        clip_local_energy (float): Clipping range for the local energy.
        normalize_gradient (bool, optional): Whether to normalize the gradient
            by the batch size. Defaults to True.

    Returns:
        Callable: Loss function
    """
    @jax.custom_jvp
    def total_energy(params, electrons, atoms, e_l):
        return pmean_if_pmap(jnp.mean(e_l))

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        e_l = primals[-1]
        loss = pmean_if_pmap(jnp.mean(e_l))

        diff = local_energy_diff(e_l, clip_local_energy)

        psi_primal, psi_tangent = jax.jvp(
            batch_network, primals[:-1], tangents[:-1])
        gradients = jnp.vdot(psi_tangent, diff)
        if normalize_gradient:
            gradients /= e_l.size
        return loss, gradients
    return total_energy


def make_loss_and_natural_gradient_fn(
        batched_network,
        clip_local_energy: float,
        clipping_method: str = 'ferminet',
        maxiter=100,
        min_lookback: float = 10,
        lookback_frac: float = 0.1,
        eps: float = 5e-6,
        center: bool = False,
        precondition: bool = False,
        **kwargs):
    """Create loss and gradient function for natural gradient descent.

    Args:
        batched_network (Callable): Batched network that takes a batch of samples
            as input.
        clip_local_energy (float): Clipping range for the local energy.
        clipping_method (str, optional): Clipping method, 
            see `clip_local_energy`. Defaults to 'ferminet'.
        maxiter (int, optional): CG maximum number of iterations. Defaults to 100.
        min_lookback (float, optional): CG stopping criteria, 
            min iterations before lookback. Defaults to 10.
        lookback_frac (float, optional): CG stopping criteria, 
            fraction of iterations to lookback. Defaults to 0.1.
        eps (float, optional): CG epsilon value. Defaults to 5e-6.
        center (bool, optional): Whether to center activations and sensitivities. Defaults to False.
        precondition (bool, optional): Whether to use a preconditioner. Defaults to False.

    Returns:
        Callable: loss and gradient function
    """
    batched_network = jax.jit(batched_network)

    def nat_cg(t, params, electrons, atoms, e_l, last_grad, damping):
        n = e_l.size

        def log_p_closure(p): return batched_network(
            p, electrons, atoms)

        diff = local_energy_diff(e_l, clip_local_energy, clipping_method)

        def loss(p):
            log_psi = batched_network(p, electrons, atoms)
            result = jnp.vdot(log_psi, diff)/n
            return result
        grad = jax.grad(loss)(params)

        if center:
            mean_grads = jax.grad(lambda p: log_p_closure(p).sum())(params)

        _, vjp_fn = jax.vjp(log_p_closure, params)

        if precondition:
            F_diag_sqrt = jax.grad(lambda p: log_p_closure(p).sum())(params)
            F_inv_diag = jax.tree_map(
                lambda x: n/(x**2 + damping), F_diag_sqrt)

            def preconditioner(x):
                return jax.tree_multimap(lambda a, b: b*a, F_inv_diag, x)
        else:
            preconditioner = None

        @jax.jit
        def Fisher_matmul(v):
            w = jax.jvp(log_p_closure, (params,), (v,))[1] / n
            uncentered = vjp_fn(w)[0]

            result = tree_add(uncentered, tree_mul(v, damping))

            if center:
                mg_v = tree_dot(mean_grads, v) / n
                result = tree_sub(result, tree_mul(mean_grads, mg_v))
            return result

        nat_grad = cg(
            A=Fisher_matmul,
            b=grad,
            x0=last_grad,
            min_lookback=min_lookback,
            lookback_frac=lookback_frac,
            eps=eps,
            maxiter=maxiter,
            M=preconditioner
        )[0]
        return pmean_if_pmap(jnp.mean(e_l)), nat_grad
    return nat_cg


def make_training_step(
        mcmc_step,
        val_and_grad,
        el_fn,
        opt_update,
        uses_cg: bool = False):
    """Creates the training step.

    Args:
        mcmc_step (Callable): MCMC sampling method.
        val_and_grad (Callable): Function that returns the loss and gradient.
        el_fn (Callable): Local Energy function
        opt_update (Callable): Otimizer update function
        batch_gnn (Optional[Callable], optional): GNN forward pass. Defaults to None.
        uses_cg (bool, optional): Whether the CG method is used - should be True if natural gradient is used. Defaults to False.

    Returns:
        Callable: training step function
    """
    def step(t, params, electrons, atoms, opt_state, key, mcmc_width, last_grad=None, damping=None):
        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(
            params, electrons, atoms, subkey, mcmc_width)

        e_l = el_fn(params, electrons, atoms)

        if uses_cg:
            # If we use CG we've wrapped the value function by adaptive damping
            grad_and_damping = val_and_grad
            key, subkey = jax.random.split(key)
            grads, damping = grad_and_damping(
                damping=damping,
                t=t,
                params=params,
                electrons=electrons,
                atoms=atoms,
                e_l=e_l,
                last_grad=last_grad)
        else:
            _, grads = val_and_grad(params,  electrons, atoms, e_l)

        # Mean gradients
        # Is meaning gradients or meaning parameters preferred?
        # Probably gradients so the optimizer state is identical
        grads = pmean_if_pmap(grads)
        if uses_cg:
            train_state = {
                'last_grad': grads,
                'damping': damping
            }
        else:
            train_state = {}

        # Compute total energy
        E = e_l.mean(-1)
        if E.ndim == 0:
            E = pmean_if_pmap(E)

        # Compute standard deviation per atom configuration
        E_var = e_l.var(-1)
        if E_var.ndim == 0:
            E_var = pmean_if_pmap(E_var)

        # Optimizer step
        updates, opt_state = opt_update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)
        params = pmean_if_pmap(params)
        return (electrons, params, opt_state, E, E_var, pmove), train_state
    return step


def init_electrons(
        atom_positions: jnp.ndarray,
        charges: jnp.ndarray,
        spins: Tuple[int, int],
        batch_size: int,
        key: jnp.ndarray) -> jnp.ndarray:
    """Initializes electron positions by normal distributions
    around the nuclei. For heavy atoms this function tries to
    match the number of spin up and spin down electrons per 
    nuclei. Otherwise, it could happen that one atom has only
    spin up and one only spin down electrons.

    Args:
        atom_positions (jnp.ndarray): (..., M, 3)
        charges (jnp.ndarray): (M)
        spins (Tuple[int, int]): (spin_up, spin_down)
        batch_size (int): total batch size
        key (jnp.ndarray): jax.random.PRNGKey

    Returns:
        jnp.ndarray: (..., batch_size//len(...), N, 3)
    """
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
    batch_size_per_config = batch_size // n_configs
    for config in range(n_configs):
        a_p = atom_positions[config]
        if sum(charges) != n_electrons:
            p = jnp.array(charges)/sum(charges)
            key, subkey = jax.random.split(key)
            atom_idx = jax.random.choice(subkey, jnp.arange(
                a_p.shape[0]), shape=(batch_size_per_config, n_electrons), replace=True, p=p)
        else:
            charges = np.array(charges)
            atom_idx = np.zeros(
                (batch_size_per_config, n_electrons), dtype=jnp.int32)
            for k in range(batch_size_per_config):
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
                atom_idx[k] = jnp.concatenate([alpha_idx, beta_idx])
            atom_idx = jnp.array(atom_idx)

        electrons.append(a_p[atom_idx].reshape(
            batch_size_per_config, n_electrons*3))
    electrons = jnp.array(electrons).reshape(
        *config_shape, batch_size_per_config, n_electrons*3)
    key, subkey = jax.random.split(key)
    return electrons + jax.random.normal(subkey, shape=electrons.shape)*0.8


def make_schedule(params: dict) -> Callable[[int], float]:
    """Simple function to create different kind of schedules.

    Args:
        params (dict): Parameters for the schedules.

    Returns:
        Callable[[int], float]: schedule function
    """
    if isinstance(params, numbers.Number):
        def result(t): return params
    elif callable(params):
        result = params
    elif isinstance(params, dict):
        if 'schedule' not in params or params['schedule'] == 'hyperbola':
            assert 'init' in params
            assert 'delay' in params
            init = params['init']
            delay = params['delay']
            decay = params['decay'] if 'decay' in params else 1
            def result(t): return init * jnp.power(1/(1 + t/delay), decay)
        elif params['schedule'] == 'exponential':
            assert 'init' in params
            assert 'delay' in params
            init = params['init']
            delay = params['delay']
            def result(t): return init * jnp.exp(-t/delay)
        else:
            raise ValueError()
        if 'min' in params:
            val_fn = result
            def result(t): return jnp.maximum(val_fn(t), params['min'])
    else:
        raise ValueError()
    return result
