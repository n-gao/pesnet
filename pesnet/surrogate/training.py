from typing import Any, Tuple
from chex import ArrayTree
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pesnet.utils.jax_utils import pmean_if_pmap
from pesnet.utils import ema_make, ema_value, ema_update
from pesnet.utils.typing import Surrogate, SurrogateEMAs, SurrogateState, SurrogateTraining


def make_surrogate_training_step(
    energy_net: Surrogate,
    optimizer: optax.GradientTransformation,
    n_steps: int,
    loss_decay: float,
    ema_decay_min: float,
    ema_decay_max: float,
    threshold: float,
    loss: str,
    **_: Any
) -> SurrogateTraining:
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

    def init(params: ArrayTree) -> SurrogateState:
        return SurrogateState(
            params,
            optimizer.init(params),
            SurrogateEMAs(
                ema_make(-1 * jnp.ones(())),
                ema_make(-1 * jnp.ones(())),
                ema_make(params)
            ),
            jnp.zeros(())
        )

    loss = loss.lower()
    if loss == 'mae':
        loss_fn = lambda pred, target, sigma: (jnp.abs(pred - target)/sigma).mean()
    elif loss == 'rmse':
        loss_fn = lambda pred, target, sigma: (((pred - target) ** 2)/(sigma**2)).mean() ** 0.5
    elif loss == 'mse':
        loss_fn = lambda pred, target, sigma: (((pred - target) ** 2)/sigma).mean()
    else:
        raise ValueError(f"'{loss}' is an invalid loss function.")
    
    def step(state: SurrogateState, atoms: jax.Array, E_l: jax.Array) -> Tuple[SurrogateState, ArrayTree]:
        # E_l is of shape (n_devices, batch_size, configurations)
        # the n_devices is the same as batch_size and hidden via pmap.
        params, opt_state, emas = state.params, state.opt_state, state.emas
        E = pmean_if_pmap(E_l.mean(0))
        E_var = pmean_if_pmap(((E_l - E)**2).mean(0))
        target = E - state.offset
        sigma = jnp.sqrt(E_var / (E_l.shape[0] * jax.device_count()))
        E_err = sigma.mean()
        def update(x, _):
            params, opt_state = x
            def loss(params):
                pred = energy_net(params, atoms)
                loss = loss_fn(pred, target, sigma) / (1/sigma).mean() # renormalize the loss
                return loss
            loss, grads = jax.value_and_grad(loss)(params)
            grads = pmean_if_pmap(grads)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss
        (params, opt_state), losses = jax.lax.scan(update, (params, opt_state), jnp.arange(n_steps), n_steps)

        loss = jnp.abs(energy_net(ema_value(emas.params, params), atoms) - target).mean()
        ema_loss = ema_update(emas.loss, loss, loss_decay)
        ema_err = ema_update(emas.err, E_err, loss_decay)
        ema_loss, ema_err = pmean_if_pmap((emas.loss, ema_err))

        ema_loss_val = ema_value(ema_loss)
        ema_err_val = ema_value(ema_err)

        thres = threshold * np.sqrt(2/np.pi) * ema_err_val

        alpha = ema_decay_min + (ema_decay_max - ema_decay_min) * (ema_loss_val < thres)
        ema_params = ema_update(emas.params, params, alpha)

        aux_data = {
            'loss': losses[0],
            'ema_loss': ema_loss_val,
            'ema_err': ema_err_val,
            'threshold': thres,
            'alpha': alpha
        }
        return SurrogateState(params, opt_state, SurrogateEMAs(ema_loss, ema_err, ema_params), state.offset), aux_data
    return SurrogateTraining(init, step)
