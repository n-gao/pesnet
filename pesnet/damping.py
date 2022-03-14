"""
This file contains different damping schedules. The most important
function here is `make_damping_fn` which creates any of the defined
schedules given a configuration.
"""
import jax.numpy as jnp

from pesnet.jax_utils import pmean_if_pmap
from pesnet.training import make_schedule


def make_damping_schedule(val_and_grad, **kwargs):
    # Simple fixed schedules based on the step number
    schedule = make_schedule(kwargs)

    def eval_and_schedule(damping, t, params, electrons, atoms, e_l, **kwargs):
        damping = schedule(t)
        _, grads = val_and_grad(
            t,
            params,
            electrons,
            atoms,
            e_l,
            damping=damping,
            **kwargs
        )
        return grads, damping
    return eval_and_schedule


def make_std_based_damping_fn(val_and_grad, base, target_pow=0.5, **kwargs):
    # A simple damping scheme based on the standard deviation of the local energy.
    def data_based(damping, t, params, electrons, atoms, e_l, **kwargs):
        target = pmean_if_pmap(
            base * jnp.power(jnp.sqrt(e_l.var(-1).mean()), target_pow))
        damping = jnp.where(damping < target, damping/0.999, damping)
        damping = jnp.where(damping > target, 0.999*damping, damping)
        damping = jnp.clip(damping, 1e-8, 1e-1)
        _, grads = val_and_grad(
            t, params, electrons, atoms, e_l, damping=damping, **kwargs)
        return grads, damping
    return data_based


def make_damping_fn(
        method: str,
        val_and_grad,
        kwargs):
    method = method.lower()
    if method == 'schedule':
        return make_damping_schedule(
            val_and_grad,
            init=kwargs['init'],
            **kwargs['schedule'])
    elif method == 'std_based':
        return make_std_based_damping_fn(
            val_and_grad,
            **kwargs['std_based']
        )
    else:
        raise ValueError()
