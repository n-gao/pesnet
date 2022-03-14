import functools

import chex
import jax
import jax._src.scipy.sparse.linalg as jssl
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from optax import OptState, Updates


class AccumulateState(OptState):
    count: chex.Array  # shape=(), dtype=jnp.int32.
    updates: Updates


def accumulate(steps_to_aggregate: int):
    """optax transformation to accumulate gradients over multiple iterations.
    First `steps_to_aggregate` updates are skipped and then the average of the
    last `steps_to_aggregate` gradients is applied.

    Args:
        steps_to_aggregate (int): Number of steps to accumulate.
    """
    def init_fn(params):
        return AccumulateState(
            count=jnp.zeros([], jnp.int32),
            updates=jax.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates, state, params=None):
        new_updates = jax.tree_map(
            lambda x, y: x/steps_to_aggregate + y,
            updates,
            state.updates
        )

        def update(_):
            updates = new_updates
            new_state = AccumulateState(
                count=state.count * 0,
                updates=jax.tree_map(jnp.zeros_like, state.updates)
            )
            return updates, new_state

        def skip(_):
            updates = jax.tree_map(jnp.zeros_like, state.updates)
            new_state = AccumulateState(
                count=state.count + 1,
                updates=new_updates
            )
            return updates, new_state

        return jax.lax.cond((state.count+1) >= steps_to_aggregate, update, skip, ())

    return optax.GradientTransformation(init_fn, update_fn)


def cg_solve(A, b, x0=None, *, maxiter, M=jssl._identity, min_lookback=10, lookback_frac=0.1, eps=5e-6):
    steps = jnp.arange(maxiter+1)-1
    gaps = (steps*lookback_frac).astype(jnp.int32)
    gaps = jnp.where(gaps < min_lookback, min_lookback, gaps)
    gaps = jnp.where(gaps > steps, steps, gaps)

    def cond_fun(value):
        x, r, gamma, p, k, cache = value
        gap = gaps[k]
        k = k - 1
        relative_still = jnp.logical_and(
            jnp.abs((cache[k] - cache[k - gap])/cache[k]) < eps*gap, gap >= 1)
        over_max = k >= maxiter
        # We check that we are after the third iteration because the first ones may have close to 0 error.
        converged = jnp.logical_and(k > 2, jnp.abs(cache[k]) < 1e-7)
        return ~(relative_still | over_max | converged)

    def body_fun(value):
        x, r, gamma, p, k, cache = value
        Ap = A(p)
        alpha = gamma / jssl._vdot_real_tree(p, Ap)
        x_ = jssl._add(x, jssl._mul(alpha, p))
        r_ = jssl._sub(r, jssl._mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = jssl._vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = jssl._add(z_, jssl._mul(beta_, p))

        Ax = jssl._add(r_, b)

        val = jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(
            lambda a, b, c: jnp.vdot(a-b, c), Ax, b, x_))
        cache_ = jax.ops.index_update(cache, jax.ops.index[k], val)
        return x_, r_, gamma_, p_, k + 1, cache_

    r0 = jssl._sub(b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = jssl._vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0, 0, jnp.zeros((maxiter,)))

    x_final, _, _, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


def cg(A, b, x0=None, *, maxiter=None, min_lookback=10, lookback_frac=0.1, eps=5e-6, M=None):
    """CG-method with the stopping criterium from Martens 2010.

    Args:
        A (Callable): Matrix A in Ax=b
        b (jnp.ndarray): b
        x0 (jnp.ndarray, optional): Initial value for x. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to None.
        min_lookback (int, optional): Minimum lookback distance. Defaults to 10.
        lookback_frac (float, optional): Fraction of iterations to look back. Defaults to 0.1.
        eps (float, optional): An epsilon value. Defaults to 5e-6.
        M (Callable, optional): Preconditioner. Defaults to None.

    Returns:
        jnp.ndarray: b
    """
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, b)

    b, x0 = jax.device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in jax.tree_leaves(b))
        maxiter = 10 * size

    if M is None:
        M = jssl._identity
    A = jssl._normalize_matvec(A)
    M = jssl._normalize_matvec(M)

    if jax.tree_structure(x0) != jax.tree_structure(b):
        raise ValueError(
            'x0 and b must have matching tree structure: '
            f'{jax.tree_structure(x0)} vs {jax.tree_structure(b)}')

    if jssl._shapes(x0) != jssl._shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: '
            f'{jssl._shapes(x0)} vs {jssl._shapes(b)}')

    solve = functools.partial(
        cg_solve,
        x0=x0,
        maxiter=maxiter,
        min_lookback=min_lookback,
        lookback_frac=lookback_frac,
        eps=eps,
        M=M
    )

    # real-valued positive-definite linear operators are symmetric
    def real_valued(x):
        return not issubclass(x.dtype.type, np.complexfloating)
    symmetric = all(map(real_valued, jax.tree_leaves(b)))
    x = lax.custom_linear_solve(
        A,
        b,
        solve=solve,
        transpose_solve=solve,
        symmetric=symmetric
    )
    info = None
    return x, info
