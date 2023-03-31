import functools
import logging
from typing import Callable

import jax
import jax._src.scipy.sparse.linalg as jssl
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax

from pesnet.utils import make_schedule
from pesnet.utils.damping import make_damping_fn
from pesnet.utils.jax_utils import pmean_if_pmap
from pesnet.utils.jnp_utils import tree_add, tree_mul
from pesnet.utils.typing import NaturalGradient, NaturalGradientState


def make_optimizer(lr: dict, transformations: list[tuple[str, tuple, dict]]) -> optax.GradientTransformation:
    return optax.chain(*[
        getattr(optax, name)(*args, **kwargs)
        for name, args, kwargs in transformations
    ],
        optax.scale_by_schedule(make_schedule(lr)),
        optax.scale(-1.)
    )


def scale_by_trust_ratio_embeddings(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.,
    eps: float = 0.,
) -> optax.GradientTransformation:
    """Scale by trust ratio but for embeddings were we don't want the norm
    over all parameters but just the last dimension.
    """

    def init_fn(params):
        del params
        return optax.ScaleByTrustRatioState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax.NO_PARAMS_MSG)

        def _scale_update(update, param):
            # Clip norms to minimum value, by default no clipping.
            param_norm = optax.safe_norm(param, min_norm, axis=-1, keepdims=True)
            update_norm = optax.safe_norm(update, min_norm, axis=-1, keepdims=True)
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

            # If no minimum norm clipping is used
            # Set trust_ratio to 1 in case where parameters would never be updated.
            zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
            safe_trust_ratio = jnp.where(
                zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

            return update * safe_trust_ratio

        updates = jax.tree_util.tree_map(_scale_update, updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def make_natural_gradient_preconditioner(
    network: Callable[..., jax.Array],
    damping: dict,
    linearize: bool = True,
    precision: str = 'float32',
    **kwargs) -> NaturalGradient:
    logging.info(f'CG precision: {precision}')
    damping_update, damping_state = make_damping_fn(**damping)

    def init(params):
        return NaturalGradientState(
            jtu.tree_map(jnp.zeros_like, params),
            damping_state
        )

    @jax.jit
    def nat_cg(params, norm, inp, damp_inp, grad, natgrad_state: NaturalGradientState):
        with jax.default_matmul_precision(precision):
            def log_p_closure(p):
                return network(p, *inp)
            
            _, vjp_fn = jax.vjp(log_p_closure, params)
            if linearize:
                _, jvp_fn = jax.linearize(log_p_closure, params)
            else:
                jvp_fn = lambda x: jax.jvp(log_p_closure, params, x)[1]

            damping, damping_state = damping_update(state=natgrad_state.damping_state, **damp_inp)

            def Fisher_matmul(v):
                w = jvp_fn(v) * norm
                uncentered = vjp_fn(w)[0]
                result = tree_add(uncentered, tree_mul(v, damping))
                result = pmean_if_pmap(result)
                return result
            
            # Compute natural gradient
            natgrad = cg(
                A=Fisher_matmul,
                b=grad,
                x0=natgrad_state.last_grad,
                fixed_iter=jax.device_count() > 1, # if we have multiple GPUs we must do a fixed number of iterations
                **kwargs,
            )[0]
        return natgrad, NaturalGradientState(natgrad, damping_state), damping
    return NaturalGradient(nat_cg, init)


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
