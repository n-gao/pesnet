from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def find_axes(atoms: jax.Array, charges: jax.Array) -> jax.Array:
    """Generates equivariant axes based on PCA.

    Args:
        atoms (jax.Array): (..., M, 3)
        charges (jax.Array): (M)

    Returns:
        jax.Array: (3, 3)
    """
    atoms = jax.lax.stop_gradient(atoms)
    # First compute the axes by PCA
    atoms = atoms - atoms.mean(-2, keepdims=True)
    s, axes = get_pca_axis(atoms, charges)
    # Let's check whether we have identical eigenvalues
    # if that's the case we need to work with soem pseudo positions
    # to get unique atoms.
    s = jnp.round(s, 5)
    is_ambiguous = jnp.count_nonzero(
        jnp.unique(s, size=3, fill_value=0)) < jnp.count_nonzero(s)
    # We always compute the pseudo coordinates because it causes some compile errors
    # for some unknown reason on A100 cards with jax.lax.cond.
    # We stretch it twice in case that all three singular values are identical
    pseudo_atoms = get_pseudopositions(atoms, charges, 1e-2)
    pseudo_atoms = get_pseudopositions(pseudo_atoms, charges, 1e-3)
    pseudo_s, pseudo_axes = get_pca_axis(pseudo_atoms, charges)
    pseudo_s = jnp.around(pseudo_s, 5)

    # Select pseudo axes if it is ambiguous
    atoms = jnp.where(is_ambiguous, pseudo_atoms, atoms)
    s = jnp.where(is_ambiguous, pseudo_s, s)
    axes = jnp.where(is_ambiguous, pseudo_axes, axes)
    
    order = jnp.argsort(-s)
    axes = axes[:, order]
    
    # Compute an equivariant vector
    distances = jnp.linalg.norm(atoms[None] - atoms[..., None, :], axis=-1)
    weights = distances.sum(-1)
    equi_vec = ((weights * charges)[..., None] * atoms).mean(0)
    equi_vec = jnp.where(jnp.linalg.norm(equi_vec) < 1e-4, jnp.ones((3,)), equi_vec)
    
    ve = equi_vec@axes
    flips = ve < 0
    axes = jnp.where(flips[None], -axes, axes)
    
    right_hand = jnp.stack([axes[:, 0], axes[:, 1], jnp.cross(axes[:, 0], axes[:, 1])], axis=1)
    return right_hand


def get_pca_axis(coords: jax.Array, weights: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
    """Computes the (weighted) PCA of the given coordinates.

    Args:
        coords (jax.Array): (N ,D)
        weights (jax.Array, optional): (N). Defaults to None.

    Returns:
        Tuple[jax.Array, jax.Array]: (D), (D, D)
    """
    if weights is None:
        weights = jnp.ones((*coords.shape[:-1],))
    weights /= weights.sum()

    mean = (weights[..., None]*coords).mean(0)
    centered = coords - mean
    cov = centered.T @ jnp.diag(weights) @ centered

    s, vecs = jnp.linalg.eigh(cov)
    return s, vecs


def get_projection_vector(atoms: jax.Array, charges: jax.Array) -> jax.Array:
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    if atoms.shape[0] == 1:
        return jnp.ones((3,))
    distances = atoms[None] - atoms[..., None, :]
    dist_norm = jnp.linalg.norm(distances, axis=-1)
    coulomb = charges[None] * charges[:, None] / dist_norm
    off_diag_mask = ~np.eye(atoms.shape[0], dtype=bool)
    coulomb, distances = coulomb[off_diag_mask], distances[off_diag_mask]
    idx = jnp.argmax(coulomb)
    scale_vec = distances[idx]
    scale_vec /= jnp.linalg.norm(scale_vec)
    return scale_vec


def get_pseudopositions(atoms: jax.Array, charges: jax.Array, eps: float = 1e-4) -> jax.Array:
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    scale_vec = get_projection_vector(atoms, charges)
    # Projected atom positions
    proj = atoms@scale_vec[..., None] * scale_vec
    pseudo_atoms = proj * eps + atoms
    return pseudo_atoms - pseudo_atoms.mean(0)
