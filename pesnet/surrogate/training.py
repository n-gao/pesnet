from typing import Any, Callable, Tuple
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from pesnet.utils.jax_utils import pmean_if_pmap
from pesnet.utils import ema_value, ema_update


def make_surrogate_training_step(
    energy_net: Callable[..., jax.Array],
    opt_update: Callable[..., Tuple[Any, Any]],
    n_steps: int,
    loss_decay: float,
    ema_decay_min: float,
    ema_decay_max: float,
    threshold: float,
    loss: str,
    **_: Any
) -> Callable:
    """
    Create a training step function for the surrogate model.

    Args:
        energy_net: A function that computes the energy given a set of atomic positions.
        opt_update: A function that updates the model parameters based on the computed gradients.
        n_steps: The number of optimization steps to take for each training step.
        loss_decay: The decay rate for the exponential moving average of the loss.
        ema_decay_min: The minimum decay rate for the exponential moving average of the parameters.
        ema_decay_max: The maximum decay rate for the exponential moving average of the parameters.
        threshold: The threshold for the loss below which the decay rate of the exponential moving average of the parameters increases.
        loss: The type of loss function to use. Can be 'mae', 'rmse', or 'mse'.

    Returns:
        A function that performs a single training step on the surrogate model.

    Raises:
        ValueError: If an invalid loss function is specified.

    """
    loss = loss.lower()
    if loss == 'mae':
        loss_fn = lambda pred, target, sigma: (jnp.abs(pred - target)/sigma).mean()
    elif loss == 'rmse':
        loss_fn = lambda pred, target, sigma: (((pred - target) ** 2)/(sigma**2)).mean() ** 0.5
    elif loss == 'mse':
        loss_fn = lambda pred, target, sigma: (((pred - target) ** 2)/sigma).mean()
    else:
        raise ValueError(f"'{loss}' is an invalid loss function.")
    
    def step(params, atoms, E_l, opt_state, emas, offset):
        ema_loss, ema_err, ema_params = emas['loss'], emas['err'], emas['params']
        E, E_var = E_l.mean(-1), E_l.var(-1)
        target = E - offset
        sigma = jnp.sqrt(E_var / E_l.shape[-1])
        E_err = sigma.mean()
        def update(x, _):
            params, opt_state = x
            def loss(params):
                pred = energy_net(params, atoms)
                loss = loss_fn(pred, target, sigma) / (1/sigma).mean() # renormalize the loss
                return loss
            loss, grads = jax.value_and_grad(loss)(params)
            updates, opt_state = opt_update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = pmean_if_pmap(params)
            return (params, opt_state), loss
        (params, opt_state), losses = jax.lax.scan(update, (params, opt_state), jnp.arange(n_steps), n_steps)

        loss = jnp.abs(energy_net(ema_value(ema_params, params), atoms) - target).mean()
        ema_loss = ema_update(ema_loss, loss, loss_decay)
        ema_err = ema_update(ema_err, E_err, loss_decay)
        ema_loss, ema_err = pmean_if_pmap((ema_loss, ema_err))

        ema_loss_val = ema_value(ema_loss)
        ema_err_val = ema_value(ema_err)

        thres = threshold * np.sqrt(2/np.pi) * ema_err_val

        alpha = ema_decay_min + (ema_decay_max - ema_decay_min) * (ema_loss_val < thres)
        ema_params = ema_update(ema_params, params, alpha)

        aux_data = {
            'loss': losses[0],
            'ema_loss': ema_loss_val,
            'ema_err': ema_err_val,
            'threshold': thres,
            'alpha': alpha
        }
        return (params, opt_state), {'loss': ema_loss, 'err': ema_err, 'params': ema_params}, aux_data
    return step


def make_many_surrogate_steps(train_step):
    """
    Returns a function that takes in the training function `train_step` and returns another function that performs
    multiple training steps using `train_step` sequentially for a batch of data.
    """
    def many_steps(params, atom_batches, energy_batches, opt_state, emas, offset):
        def step(carry, i):
            atoms, E_l = atom_batches[i], energy_batches[i]
            (params, opt_state), emas = carry
            (params, opt_state), emas, aux_data = train_step(params, atoms, E_l, opt_state, emas, offset)
            return ((params, opt_state), emas), aux_data
        ((params, opt_state), emas), aux_data = jax.lax.scan(step, ((params, opt_state), emas), jnp.arange(len(atom_batches)))
        return (params, opt_state), emas, aux_data
    return many_steps


def make_joint_training_step(
        vmc_step,
        surrogate_step
    ):
    """
    Creates a joint training step function that combines variational Monte Carlo (VMC) and surrogate training.

    Args:
        vmc_step: A function that performs a single step of variational Monte Carlo (VMC) calculation.
        surrogate_step: A function that performs a single step of surrogate training.

    Returns:
        A joint training step function that performs both VMC and surrogate training.
    """
    def step(
        t, 
        atoms, 
        vmc_params,
        surrogate_params,
        ):
        aux_data = {}
        (electrons, vmc_params, vmc_state, e_l, E, E_var, pmove, cg_state), aux_data['cg'] = vmc_step(
            t=t,
            atoms=atoms,
            **vmc_params)
        (surrogate_params, surrogate_state), surrogate_emas, aux_data['surrogate'] = surrogate_step(
            E_l=e_l,
            atoms=atoms,
            **surrogate_params
        )
        pmove = jnp.mean(pmove)
        # Remove n_gpu axis
        aux_data = jtu.tree_map(lambda x: jnp.mean(x), aux_data)
        return (electrons, vmc_params, vmc_state, e_l, E, E_var, pmove, cg_state), (surrogate_params, surrogate_state, surrogate_emas), aux_data
    return step
