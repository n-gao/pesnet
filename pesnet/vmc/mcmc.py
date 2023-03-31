"""
This implementation is inspired by Spencer et al., though,
adapted to support multiple geometries.
These function should not be vmapped!
"""
from typing import Tuple

from chex import ArrayTree, PRNGKey
import jax
import jax.lax as lax
import jax.numpy as jnp

from pesnet.utils.jax_utils import pmean_if_pmap
from pesnet.utils.typing import McmcFn, WaveFunction


def mh_update(
    logprob_fn,
    key: PRNGKey,
    electrons: jax.Array,
    lp_1: jax.Array,
    num_accepts: int,
    stddev: float
) -> Tuple[PRNGKey, jax.Array, jax.Array, jax.Array]:
    """Metropolis Hastings step

    Args:
        logprob_fn (Callable): log probability function
        electrons (jax.Array): (bN, 3)
        key (jax.Array): jax.random.PRNGKey
        lp_1 (jax.Array): (B)
        num_accepts (int): number of past accepts
        stddev (float, optional): proposal width. Defaults to 0.02.
        i (int, optional): step count. Defaults to 0.

    Returns:
        Tuple[jax.Array, jax.Array, jax.Array, int]: 
            new electrons, new key, new log prob, new num_accepts
    """
    key, subkey = jax.random.split(key)
    new_electrons = electrons + stddev * \
        jax.random.normal(subkey, electrons.shape)
    lp_2 = logprob_fn(new_electrons)
    ratio = lp_2 - lp_1

    key, subkey = jax.random.split(key)
    alpha = jnp.log(jax.random.uniform(subkey, lp_1.shape))
    cond = ratio > alpha
    x_new = jnp.where(cond[..., None, None], new_electrons, electrons)
    lp_new = jnp.where(cond, lp_2, lp_1)
    num_accepts += cond

    return key, x_new, lp_new, num_accepts


def make_mcmc(
    batch_network: WaveFunction,
    steps=10,
    **kwargs
) -> McmcFn:
    """Create Monte Carlo Markov Chain function.

    Args:
        batch_network (Callable): batched network.
        steps (int, optional): number of decorrelation steps. Defaults to 10.

    Returns:
        Callable: sampler
    """
    @jax.jit
    def mcmc_step(
            key: PRNGKey,
            params: ArrayTree,
            electrons: jax.Array,
            atoms: jax.Array,
            width: jax.Array):
        """Perform `steps` Metropolis hastings steps.

        Args:
            params (ArrayTree): network parameters
            electrons (jax.Array): (B, N, 3)
            atoms (jax.Array): (M, 3)
            key (jax.Array): jax.random.PRNGKey
            width (jax.Array): proposal width

        Returns:
            Tuple[jax.Array, jax.Array]: new electrons, move probability
        """
        def logprob_fn(x):
            return 2. * batch_network(params, x, atoms)
        # In case that we have a different width for every atom
        # we need to add a dimension to enable broadcasting
        if isinstance(width, jax.Array) and width.ndim == 1:
            width = width[:, None, None]

        def step_fn(x, _):
            return mh_update(logprob_fn, *x, stddev=width), None

        logprob = logprob_fn(electrons)
        num_accepts = jnp.zeros(logprob.shape)

        # Execute step function `steps` times.
        key, electrons, _, num_accepts = lax.scan(
            step_fn,
            (key, electrons, logprob, num_accepts),
            jnp.arange(steps)
        )[0]

        pmove = jnp.mean(num_accepts, axis=0) / steps
        pmove = pmean_if_pmap(pmove)
        return electrons, pmove
    return mcmc_step
