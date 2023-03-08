# This file is largely taken from Klicpera et al. 2019.
# https://github.com/klicperajo/dimenet/blob/master/dimenet/model/layers/basis_utils.py
import functools
from typing import Callable

import jax.numpy as jnp
import numpy as np
import sympy as sym
from scipy import special as sp
from scipy.optimize import brentq


def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')
    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f


@functools.lru_cache
def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return ((2*l+1) * np.math.factorial(l-abs(m)) / (4*np.pi*np.math.factorial(l+abs(m))))**0.5


def associated_legendre_polynomials(l, zero_m_only=True):
    """
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols('z')
    P_l_m = [[0]*(j+1) for j in range(l)]

    P_l_m[0][0] = 1
    if l > 1:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0])/j)
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1-2*i)*P_l_m[i-1][i-1])
                if i + 1 < l:
                    P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i])
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        ((2*j-1) * z * P_l_m[j-1][i] - (i+j-1) * P_l_m[j-2][i]) / (j - i))

    return P_l_m


@functools.lru_cache
def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


def sympy2jax(symbols, fn) -> Callable:
    """Converts sympy functions to JAX numpy.

    Args:
        symbols ([type]): Sympy symbols to replace by variables.
        fn (function): Sympy function to convert to JAX.

    Returns:
        Callable: JAX function
    """
    return sym.lambdify(symbols, fn, modules={
        'sin': jnp.sin,
        'cos': jnp.cos
    })


def make_bessel_basis(l: int, k: int) -> Callable:
    """Returns a function that computes the l*k bessel basis functions.

    Args:
        l (int): N_sph
        k (int): N_rad

    Returns:
        Callable: Jax bessel basis function
    """
    symbol = sym.symbols('x')
    bessel_fns = bessel_basis(l, k)
    # We need to replace all factors like 1/x and 1/x**2 by
    # 1/(x + 1e-7), 1/(x**2 + 1), ... so they approach 0 close to x=0
    replace_dict = {
        1/(symbol)**i: 1/(symbol**i + 1)
        for i in range(1, l+1)
    }
    bessel_jax = [
        sympy2jax(symbol, fn.xreplace(replace_dict))
        for fn_s in bessel_fns for fn in fn_s
    ]

    def compute_bessel_basis(x):
        rbf = [f(x) for f in bessel_jax]
        rbf = jnp.stack(rbf, axis=-1).reshape(*x.shape, l, k)
        return rbf
    return compute_bessel_basis


def make_real_sph_harm(l: int, spherical_coordinates: bool = False) -> Callable:
    """Returns the real spherical harmonic function in JAX.

    Args:
        l (int): N_sph
        spherical_coordinates (bool, optional): 
            Whether the input is going to be in spherical coordinates. Defaults to False.

    Returns:
        Callable: real spherical harmonic function in JAX
    """
    sph_fns = [s[0] for s in real_sph_harm(
        l, 
        spherical_coordinates=spherical_coordinates
    )]
    if spherical_coordinates:
        symbol = sym.symbols('theta')
    else:
        symbol = sym.symbols('z')
    jax_fns = []
    for i in range(len(sph_fns)):
        if i == 0:
            fn = sympy2jax(symbol, sph_fns[i])
            jax_fns.append(lambda x: fn(x) + jnp.zeros_like(x))
        else:
            jax_fns.append(sympy2jax(symbol, sph_fns[i]))

    def compute_real_sph_harm(x):
        to_stack = [
            fn(x)
            for fn in jax_fns
        ]
        return jnp.stack(to_stack, axis=-1)

    return compute_real_sph_harm


def make_envelope_fn(exponent: int = 5) -> Callable:
    """Envelope function that decays at 1.

    Args:
        exponent (int, optional): Rate of decay. Defaults to 5.

    Returns:
        Callable: Jax envelope function.
    """
    p = exponent + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2

    def envelope(x):
        env_val = 1/(x + 1e-7) + a*x**(p - 1) + b*x**p + c*x**(p + 1)
        return jnp.where(x < 1, env_val, jnp.zeros_like(x))

    return envelope


def make_positional_encoding(n_sph: int, n_rad: int, cutoff: float) -> Callable:
    """Make positional encoding function.

    Args:
        n_sph (int): Number of spherical functions.
        n_rad (int): Number of radial functions.
        cutoff (float): Cutoff radius

    Returns:
        Callable: JAX positional encoding function: (..., 3) -> (...,n_sph*n_rad)
    """
    bessel_fn = make_bessel_basis(n_sph, n_rad)
    sph_harm_fn = make_real_sph_harm(n_sph)
    envelope_fn = make_envelope_fn()

    def positional_encoding(x, axis=None):
        x_norm = jnp.linalg.norm(x, axis=-1)

        if axis is not None:
            x_hat = x@axis
        else:
            x_hat = x
        angles = x_hat/(x_norm[..., None] + 1e-7)
        x_scaled = jnp.abs(x_hat/cutoff)

        rbf = bessel_fn(x_scaled)
        # envelope = envelope_fn(x_scaled)[..., None, None]
        sph = sph_harm_fn(angles)[..., None]

        # Let's sum over the axes
        return (rbf*sph).sum(-3).reshape(*x.shape[:-1], -1)

    return positional_encoding
