import jax
import jax.numpy as jnp


def tree_mul(tree, x):
    return jax.tree_map(lambda a: a*x, tree)


def tree_dot(a, b):
    return jax.tree_util.tree_reduce(
        jnp.add, jax.tree_map(
            jnp.sum, jax.tree_multimap(jax.lax.mul, a, b))
    )


def tree_add(a, b):
    return jax.tree_util.tree_multimap(
        jnp.add, a, b
    )


def tree_sub(a, b):
    return jax.tree_util.tree_multimap(
        jnp.subtract, a, b
    )
