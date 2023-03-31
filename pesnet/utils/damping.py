"""
This file contains different damping schedules. The most important
function here is `make_damping_fn` which creates any of the defined
schedules given a configuration.
"""
from typing import NamedTuple
import chex
import jax
import jax.numpy as jnp

from pesnet.utils.jax_utils import pmean_if_pmap
from pesnet.utils import make_schedule


class DampingScheduleState(NamedTuple):
    step: jax.Array


class StdDampingState(NamedTuple):
    damping: jax.Array


def make_damping_schedule(**kwargs):
    # Simple fixed schedules based on the step number
    schedule = make_schedule(kwargs)

    def eval_and_schedule(damping, state: DampingScheduleState, **_):
        step = state.step + 1
        damping = schedule(step)
        return damping, DampingScheduleState(step)
    return eval_and_schedule, DampingScheduleState(jnp.zeros(()))


def make_std_based_damping_fn(init, base, target_pow: float = 0.5, decay: float = 0.999, max_damp: float = 1e-1, **_):
    # A simple damping scheme based on the standard deviation of the local energy.
    def data_based(e_l, state: StdDampingState, **_):
        damping = state.damping
        target = pmean_if_pmap(
            base * jnp.power(jnp.sqrt(e_l.var(-1).mean()), target_pow))
        damping = jnp.where(damping < target, damping/decay, damping)
        damping = jnp.where(damping > target, decay*damping, damping)
        damping = jnp.clip(damping, 1e-8, max_damp)
        return damping, StdDampingState(damping)
    return data_based, StdDampingState(jnp.ones(()) * init)


def make_damping_fn(
        method: str,
        init: float,
        **kwargs):
    method = method.lower()
    if method == 'schedule':
        return make_damping_schedule(
            init=init,
            **kwargs['schedule'])
    elif method == 'std_based':
        return make_std_based_damping_fn(
            init=init,
            **kwargs['std_based']
        )
    else:
        raise ValueError()
