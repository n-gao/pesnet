import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def tree_mul(tree, x):
    return jtu.tree_map(lambda a: a*x, tree)


def tree_add(tree1, tree2):
    return jtu.tree_map(lambda a, b: a+b, tree1, tree2)


def tree_sub(tree1, tree2):
    return jtu.tree_map(lambda a, b: a-b, tree1, tree2)


def tree_dot(a, b):
    return jtu.tree_reduce(
        jnp.add, jtu.tree_map(
            jnp.sum, jax.tree_map(jax.lax.mul, a, b))
    )
