"""
Utilities for working with JAX.
Some of these functions are taken from 
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""
import functools
from typing import Callable, TypeVar
from chex import ArrayTree, PRNGKey

import jax
import jax.tree_util as jtu
from jax import core


T = TypeVar('T')

broadcast = jax.pmap(lambda x: x)
instance = functools.partial(jtu.tree_map, lambda x: x[0])

_p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def p_split(key: PRNGKey) -> tuple[PRNGKey, ...]:
    return _p_split(key)


def replicate(pytree: ArrayTree) -> ArrayTree:
    n = jax.local_device_count()
    stacked_pytree = jtu.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
def pmap(fn: T, *args, **kwargs) -> T:
    return jax.pmap(fn, *args, **kwargs, axis_name=PMAP_AXIS_NAME)
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)
pgather = functools.partial(jax.lax.all_gather, axis_name=PMAP_AXIS_NAME)


def wrap_if_pmap(p_func):
    def p_func_if_pmap(obj):
        try:
            core.axis_frame(PMAP_AXIS_NAME)
            return p_func(obj)
        except NameError:
            return obj
    return p_func_if_pmap


pmean_if_pmap = wrap_if_pmap(pmean)
psum_if_pmap = wrap_if_pmap(psum)
pmax_if_pmap = wrap_if_pmap(pmax)
