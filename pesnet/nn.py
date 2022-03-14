import functools
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from jax.nn.initializers import normal, variance_scaling, zeros

Activation = Union[str, Callable[[jnp.ndarray], jnp.ndarray]]
ParamTree = Union[jnp.ndarray,
                  Iterable['ParamTree'],
                  Mapping[Any, 'ParamTree']]

Dense = functools.partial(
    nn.Dense,
    kernel_init=variance_scaling(
        1/2,
        mode="fan_in",
        distribution="truncated_normal"
    ),
    bias_init=normal(1/np.sqrt(2))
)
Dense_no_bias = functools.partial(
    nn.Dense,
    kernel_init=variance_scaling(
        1,
        mode="fan_in",
        distribution="truncated_normal"
    ),
    bias_init=zeros
)

def none(x):
    return x


ACTIVATION_GAINS = {
    nn.tanh: 1.5927812698663606,
    nn.sigmoid: 4.801203511726151,
    none: 1,
}


def activation_function(fn: Union[str, Activation]):
    if callable(fn):
        return fn
    activations = {f.__name__: f for f in ACTIVATION_GAINS.keys()}
    if fn in activations:
        return activations[fn]
    else:
        try:
            return getattr(nn, fn)
        except:
            return getattr(jnp, fn)


def residual(
    x: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    """Adds a residual connection between input x and output y if possible.

    Args:
        x (jnp.ndarray): input
        y (jnp.ndarray): output

    Returns:
        jnp.ndarray: new output
    """
    if x.shape == y.shape:
        return (x + y) / jnp.sqrt(2.0)
    else:
        return y


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Activation
    intermediate_bias: bool = True
    final_bias: bool = True

    @nn.compact
    def __call__(self, x):
        if len(self.hidden_dims) == 0:
            return x

        Dense_inter = Dense if self.intermediate_bias else Dense_no_bias
        Dense_out = Dense if self.final_bias else Dense_no_bias

        activation = ActivationWithGain(self.activation)

        y = x
        for hidden_dim in self.hidden_dims[:-1]:
            y = activation(Dense_inter(hidden_dim)(y))
        y = Dense_out(self.hidden_dims[-1])(y)
        return y

    @staticmethod
    def extract_final_linear(params):
        key = list(params)[-1]
        return params[key]


class AutoMLP(nn.Module):
    out_dim: int
    n_layers: int
    activation: Activation
    scale: str = 'log'
    intermediate_bias: bool = True
    final_bias: bool = True

    @nn.compact
    def __call__(self, x):
        inp_dim = x.shape[-1]
        # We use np instead of jnp to ensure that it is static.
        if self.out_dim > 0 and inp_dim > 0:
            if self.scale == 'log':
                hidden_dims = np.round(
                    np.logspace(
                        np.log(inp_dim),
                        np.log(self.out_dim),
                        self.n_layers + 1,
                        base=np.e
                    )
                ).astype(np.int32)[1:]
            elif self.scale == 'linear':
                hidden_dims = np.round(
                    np.linspace(
                        inp_dim,
                        self.out_dim,
                        self.n_layers + 1
                    )
                ).astype(np.int32)[1:]
            else:
                raise ValueError()
        else:
            hidden_dims = [0]
        if inp_dim == 0:
            hidden_dims = [self.out_dim]

        Dense_inter = Dense if self.intermediate_bias else Dense_no_bias
        Dense_out = Dense if self.final_bias else Dense_no_bias

        activation = ActivationWithGain(self.activation)

        y = x
        for hidden_dim in hidden_dims[:-1]:
            y = activation(Dense_inter(hidden_dim)(y))
        y = Dense_out(hidden_dims[-1])(y)
        return y


class ActivationWithGain(nn.Module):
    activation: Activation

    @nn.compact
    def __call__(self, x):
        activation = activation_function(self.activation)
        return activation(x) * ACTIVATION_GAINS[activation]


def named(name, module, *args, **kwargs):
    return type(name, (module,), {})(*args, **kwargs)


class BesselRBF(nn.Module):
    out_dim: int
    cutoff: float

    @nn.compact
    def __call__(self, x):
        f = self.param(
            'f',
            lambda *_: (jnp.arange(self.out_dim,
                        dtype=jnp.float32) + 1) * jnp.pi,
            (self.out_dim,)
        )
        c = self.param(
            'c',
            lambda *_: jnp.ones(()) * self.cutoff,
            ()
        )
        x_ext = x[..., None] + 1e-8
        result = jnp.sqrt(2./c) * jnp.sin(f*x_ext/c)/x_ext
        return result.reshape(*x.shape, -1)
