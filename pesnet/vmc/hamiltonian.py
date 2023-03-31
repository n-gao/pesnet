from chex import ArrayTree
import jax
import jax.numpy as jnp
from jax import lax

from pesnet.utils.typing import EnergyFn, WaveFunction


def make_kinetic_energy_function(f: WaveFunction) -> EnergyFn:
    """Returns a function that computes the kinetic energy for wave function f.

    Args:
        f (Callable): wave function
        use_fast_path (bool, optional): Whether to use the fast pass for small systems. 
            Defaults to True.
    """
    def laplacian_of_f(params: ArrayTree, electrons: jax.Array, atoms: jax.Array) -> jax.Array:
        """Computes the kinetic energy for the given parameters.

        Args:
            params (ArrayTree): wave function parameters
            electrons (jax.Array): (N, 3) electrons
            atoms (jax.Array): (M, 3) nuclei

        Returns:
            jax.Array: kinetic energy
        """
        def f_closure(x): return f(params, x, atoms)
        electrons = electrons.reshape(-1)
        n = electrons.shape[0]
        eye = jnp.eye(n)
        grad_f = jax.grad(f_closure)
        grad, dgrad_f = jax.linearize(grad_f, electrons)

        _, diagonal = lax.scan(lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n)
        result = -0.5 * (jnp.sum(diagonal) + jnp.sum(grad ** 2))
        return result
    return laplacian_of_f


def potential_energy(electrons: jax.Array, atoms: jax.Array, charges: jax.Array) -> jax.Array:
    """Compute the potential energy between electrons and nuclei.
    This function is fully differentiable.

    Args:
        electrons (jax.Array): (N, 3)
        atoms (jax.Array): (M, 3)
        charges (jax.Array): (M)

    Returns:
        jax.Array: potential energy
    """
    electrons = electrons.reshape(-1, 3)
    # We do not have to clean the diagonal because jnp.triu discards it!
    r_ee = jnp.linalg.norm(
        electrons[None] - electrons[:, None],
        axis=-1
    )
    v_ee = jnp.sum(jnp.triu(1. / r_ee, k=1))

    r_ae = jnp.linalg.norm(atoms[None] - electrons[:, None], axis=-1)
    v_ae = -jnp.sum(charges / r_ae)

    # We do not have to clean the diagonal because jnp.triu discards it!
    r_aa = jnp.linalg.norm(
        atoms[None] - atoms[:, None],
        axis=-1
    )
    v_aa = jnp.sum(jnp.triu((charges[None] * charges[:, None]) / r_aa, k=1))
    return v_ee + v_ae + v_aa


def make_local_energy_function(f: WaveFunction, atoms: jax.Array, charges: jax.Array) -> EnergyFn:
    """Returns a function that computes the local energy for wave function f
    with the given atoms and charges.

    Args:
        f (Callable): wave function
        atoms (jax.Array): (M, 3)
        charges (jax.Array): (M)

    Returns:
        Callable: local energy function
    """
    charges = jnp.array(charges)
    kinetic_energy_fn = make_kinetic_energy_function(f)

    def local_energy(
            params: ArrayTree,
            electrons: jax.Array,
            atoms: jax.Array = atoms,
            charges: jax.Array = charges) -> jax.Array:
        """Computes the local energy (kinetic+potential)

        Args:
            params ([type]): wave function parameters
            electrons (jax.Array): (N, 3)
            atoms (jax.Array, optional): (M, 3). Defaults to atoms.
            charges (jax.Array, optional): (M). Defaults to charges.

        Returns:
            jax.Array: local energy
        """
        potential = potential_energy(electrons, atoms, charges)
        kinetic = kinetic_energy_fn(params, electrons, atoms)
        return potential + kinetic

    return local_energy
