"""
Utilities for working with JAX.
Some of these functions are taken from 
https://github.com/deepmind/ferminet/tree/jax/ferminet
"""
import functools

import jax
from jax import core

broadcast = jax.pmap(lambda x: x)

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def replicate(pytree):
    n = jax.local_device_count()
    stacked_pytree = jax.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
    return broadcast(stacked_pytree)


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(jax.lax.pmean, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(jax.lax.psum, axis_name=PMAP_AXIS_NAME)
pmax = functools.partial(jax.lax.pmax, axis_name=PMAP_AXIS_NAME)


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
