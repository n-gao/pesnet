import functools

import jax
import jax._src.scipy.sparse.linalg as jssl
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


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

        val = jtu.tree_reduce(jnp.add, jtu.tree_map(
            lambda a, b, c: jnp.vdot(a-b, c), Ax, b, x_))
        cache_ = cache.at[k].set(val)
        return x_, r_, gamma_, p_, k + 1, cache_

    r0 = jssl._sub(b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = jssl._vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0, 0, jnp.zeros((maxiter,)))

    x_final, _, _, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


def cg_solve_fixediter(A, b, x0=None, *, maxiter, M=jssl._identity):
    # Implementation of CG-method with a fixed number of iterations
    def body_fun(value, i):
        del i
        x, r, gamma, p = value
        Ap = A(p)
        alpha = gamma / jssl._vdot_real_tree(p, Ap)
        x_ = jssl._add(x, jssl._mul(alpha, p))
        r_ = jssl._sub(r, jssl._mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = jssl._vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = jssl._add(z_, jssl._mul(beta_, p))

        return (x_, r_, gamma_, p_), None

    r0 = jssl._sub(b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = jssl._vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0)

    x_final, _, _, _ = lax.scan(body_fun, initial_value, jnp.arange(maxiter), maxiter)[0]
    return x_final


def cg(A, b, x0=None, *, maxiter=None, min_lookback=10, lookback_frac=0.1, eps=5e-6, M=None, fixed_iter=False):
    """CG-method with the stopping criterium from Martens 2010.

    Args:
        A (Callable): Matrix A in Ax=b
        b (jax.Array): b
        x0 (jax.Array, optional): Initial value for x. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to None.
        min_lookback (int, optional): Minimum lookback distance. Defaults to 10.
        lookback_frac (float, optional): Fraction of iterations to look back. Defaults to 0.1.
        eps (float, optional): An epsilon value. Defaults to 5e-6.
        M (Callable, optional): Preconditioner. Defaults to None.

    Returns:
        jax.Array: b
    """
    if x0 is None:
        x0 = jtu.tree_map(jnp.zeros_like, b)

    b, x0 = jax.device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in jtu.tree_leaves(b))
        maxiter = 10 * size

    if M is None:
        M = jssl._identity
    A = jssl._normalize_matvec(A)
    M = jssl._normalize_matvec(M)

    if jtu.tree_structure(x0) != jtu.tree_structure(b):
        raise ValueError(
            'x0 and b must have matching tree structure: '
            f'{jtu.tree_structure(x0)} vs {jtu.tree_structure(b)}')

    if jssl._shapes(x0) != jssl._shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: '
            f'{jssl._shapes(x0)} vs {jssl._shapes(b)}')

    if fixed_iter:
        solve = functools.partial(
            cg_solve_fixediter,
            x0=x0,
            maxiter=maxiter,
            M=M
        )
    else:
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
    symmetric = all(map(real_valued, jtu.tree_leaves(b)))
    x = lax.custom_linear_solve(
        A,
        b,
        solve=solve,
        transpose_solve=solve,
        symmetric=symmetric
    )
    info = None
    return x, info
