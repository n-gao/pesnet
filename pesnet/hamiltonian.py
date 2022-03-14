from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from pesnet.nn import ParamTree


def make_kinetic_energy_function(f):
    """Returns a function that computes the kinetic energy for wave function f.

    Args:
        f (Callable): wave function
    """
    def laplacian_of_f(params: ParamTree, electrons: jnp.ndarray, atoms: jnp.ndarray) -> jnp.ndarray:
        """Computes the kinetic energy for the given parameters.

        Args:
            params (ParamTree): wave function parameters
            electrons (jnp.ndarray): (N, 3) electrons
            atoms (jnp.ndarray): (M, 3) nuclei

        Returns:
            jnp.ndarray: kinetic energy
        """
        def f_closure(x): return f(params, x, atoms)
        electrons = electrons.reshape(-1)
        n = electrons.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f_closure)

        def ith_sum_element(carry: jnp.ndarray, i: int) -> jnp.ndarray:
            grad, gradgrad = jax.jvp(grad_f, (electrons,), (eye[i],))
            return carry + grad[i]**2 + gradgrad[i], None

        return -0.5 * lax.scan(ith_sum_element, 0.0, jnp.arange(n))[0]

    return laplacian_of_f


def potential_energy(electrons: jnp.ndarray, atoms: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
    """Compute the potential energy between electrons and nuclei.
    This function is fully differentiable.

    Args:
        electrons (jnp.ndarray): (N, 3)
        atoms (jnp.ndarray): (M, 3)
        charges (jnp.ndarray): (M)

    Returns:
        jnp.ndarray: potential energy
    """
    electrons = electrons.reshape(-1, 3)
    r_ee = jnp.linalg.norm(
        electrons[None] - electrons[:, None],
        axis=-1
    )
    v_ee = jnp.sum(jnp.triu(1. / r_ee, k=1))

    r_ae = jnp.linalg.norm(atoms[None] - electrons[:, None], axis=-1)
    v_ae = -jnp.sum(charges / r_ae)

    r_aa = jnp.linalg.norm(
        atoms[None] - atoms[:, None],
        axis=-1
    )
    v_aa = jnp.sum(jnp.triu((charges[None] * charges[:, None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa


def make_local_energy_function(f, atoms: jnp.ndarray, charges: Tuple[int, ...]):
    """Returns a function that computes the local energy for wave function f
    with the given atoms and charges.

    Args:
        f (Callable): wave function
        atoms (jnp.ndarray): (M, 3)
        charges (Tuple[int, ...]): (M)

    Returns:
        Callable: local energy function
    """
    charges = jnp.array(charges)
    kinetic_energy_fn = make_kinetic_energy_function(f)

    def local_energy(
            params,
            electrons: jnp.ndarray,
            atoms: jnp.ndarray = atoms,
            charges: jnp.ndarray = charges) -> jnp.ndarray:
        """Computes the local energy (kinetic+potential)

        Args:
            params ([type]): wave function parameters
            electrons (jnp.ndarray): (N, 3)
            atoms (jnp.ndarray, optional): (M, 3). Defaults to atoms.
            charges (jnp.ndarray, optional): (M). Defaults to charges.

        Returns:
            jnp.ndarray: local energy
        """
        potential = potential_energy(electrons, atoms, charges)
        kinetic = kinetic_energy_fn(params, electrons, atoms)
        return potential + kinetic

    return local_energy
