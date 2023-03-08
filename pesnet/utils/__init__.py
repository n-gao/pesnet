from copy import deepcopy
import numbers
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from uncertainties import ufloat

from .jax_utils import pmap


class MCMCStepSizeScheduler:
    """Utility class to schedule the proposal width of the MCMC.
    """

    def __init__(
            self,
            init_width: jax.Array,
            target_pmove: float = 0.525,
            error: float = 0.025,
            update_interval: int = 20,
            **kwargs) -> None:
        if isinstance(init_width, jax.Array):
            self.width = init_width
        else:
            self.width = jnp.array(init_width)
        self.update_interval = update_interval
        self.target_pmove = target_pmove
        self.error = error
        self.pmoves = np.zeros((update_interval, *self.width.shape))
        self.i = 0

    def __call__(self, pmove: float) -> float:
        if self.i % self.update_interval == 0 and self.i > 0:
            pm_mean = self.pmoves.mean(0)
            self.width = np.where(pm_mean < self.target_pmove - self.error,
                                  self.width / 1.1, self.width)
            self.width = np.where(pm_mean > self.target_pmove + self.error,
                                  self.width * 1.1, self.width)
        self.pmoves[self.i % self.update_interval] = pmove
        self.i += 1
        return self.width


class ExponentiallyMovingAverage:
    """A simple implementation of an exponentially moving average.
    """

    def __init__(self) -> None:
        self._value = 0.
        self._total_weight = 0.

    def update(self, value, decay):
        self._value *= decay
        self._total_weight *= decay
        self._value += value
        self._total_weight += 1.

    @property
    def value(self):
        return self._value / self._total_weight


# EMA for usage in JAX
def ema_make(tree):
    return (jtu.tree_map(lambda x: jnp.zeros_like(x), tree), jnp.zeros(()))


@jax.jit
def ema_update(data, value, decay):
    tree, weight = data
    return jtu.tree_map(lambda a, b: a*decay + b, tree, value), weight*decay + 1


@jax.jit
def ema_value(data, backup = None):
    tree, weight = data
    if backup is None:
        backup = tree
    is_nan = weight == 0
    return jtu.tree_map(lambda x, y: jnp.where(is_nan, y, x/weight), tree, backup)


p_ema_make = pmap(ema_make)
p_ema_value = pmap(ema_value)


class OnlineMean():
    """Compute mean and standard deviations in an online fashion.
    """

    def __init__(self):
        self._mean = 0
        self.n = 0
        self.comoment = 0

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self.comoment/self.n

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def err(self):
        return self.std/np.sqrt(self.n)

    def update(self, x):
        self.n += len(x)
        last_mean = self._mean
        self._mean += (x-self._mean).sum()/self.n
        self.comoment += ((x - self._mean) * (x - last_mean)).sum()

    def __str__(self) -> str:
        return f'{ufloat(self.mean, self.err):S}'


class Stopwatch:
    """Utility class to measure time deltas.
    """

    def __init__(self) -> None:
        self.last_step = time.time()

    def __call__(self):
        new_time = time.time()
        result = new_time - self.last_step
        self.last_step = new_time
        return result


def to_numpy(pytree):
    return jtu.tree_map(lambda x: np.asarray(x), pytree)


def to_jnp(pytree):
    return jtu.tree_map(lambda x: jnp.array(x), pytree)


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


def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """Merge to filter dictionaries. If two values coolide,
    filter 2 overrides filter 1.

    Args:
        filter1 (dict): Filter 1
        filter2 (dict): Filter 2

    Returns:
        dict: Merged filter of 1 and 2
    """
    result = deepcopy(dict1)
    for key, val in dict2.items():
        if key not in result:
            result[key] = val
        elif isinstance(val, dict) and isinstance(result[key], dict):
            result[key] = merge_dictionaries(result[key], val)
        elif isinstance(val, list) and isinstance(result[key], list):
            result[key] = merge_dictionaries(result[key], val)
        else:
            result[key] = val
    return result
