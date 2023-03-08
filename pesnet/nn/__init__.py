import functools
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn.initializers import normal, orthogonal, variance_scaling

Activation = Union[str, Callable[[jax.Array], jax.Array]]
ParamTree = Union[jax.Array,
                  Iterable['ParamTree'],
                  Mapping[Any, 'ParamTree']]


ACTIVATION_GAINS = {
    nn.silu: 1.7868129431578026,
    nn.tanh: 1.5927812698663606,
    nn.sigmoid: 4.801203511726151
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


_LAYERS = {
    'Dense': nn.Dense,
    'Dense_no_bias': functools.partial(nn.Dense, use_bias=False)
}

Dense = lambda *args, **kwargs: _LAYERS['Dense'](*args, **kwargs)
Dense_no_bias = lambda *args, **kwargs: _LAYERS['Dense_no_bias'](*args, **kwargs)
Embed = nn.Embed


def glorot_orthogonal(scale=2.0):
    base = orthogonal()
    def _glorot_orthogonal(key, shape, dtype=jnp.float32):
        assert len(shape) == 2
        W = base(key, shape, dtype)
        W *= jnp.sqrt(scale / ((shape[0] + shape[1]) * jnp.var(W)))
        return W
    return _glorot_orthogonal



def set_init_method(method: str = 'default'):
    """
    Globally set the initialization method for dense layers.
    """
    if method == 'default':
        _LAYERS['Dense'] = nn.Dense
        _LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'ferminet':
        _LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=variance_scaling(
                1,
                mode="fan_in",
                distribution="truncated_normal"
            ),
            bias_init=normal(1)
        )
        _LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'pesnet':
        _LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=variance_scaling(
                1/2,
                mode="fan_in",
                distribution="truncated_normal"
            ),
            bias_init=normal(1/np.sqrt(2))
        )
        _LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'orthogonal':
        _LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=orthogonal()
        )
        _LAYERS['Dense_no_bias'] = functools.partial(Dense, use_bias=False)
    elif method == 'orthogonal_glorot':
        _LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=glorot_orthogonal()
        )
        _LAYERS['Dense_no_bias'] = functools.partial(Dense, use_bias=False)
    else:
        raise ValueError()


def residual(
    x: jax.Array,
    y: jax.Array
) -> jax.Array:
    """Adds a residual connection between input x and output y if possible.

    Args:
        x (jax.Array): input
        y (jax.Array): output

    Returns:
        jax.Array: new output
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
    # MLP class which automatically infers hidden dimensions based on the input output dimension
    # and the number of intermediate layers.
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
    # Rescaled activation function such that the output standard deviation is approx. 1.
    activation: Activation

    @nn.compact
    def __call__(self, x):
        activation = activation_function(self.activation)
        if isinstance(activation, nn.Module) or activation not in ACTIVATION_GAINS:
            return activation(x)
        else:
            return activation(x) * ACTIVATION_GAINS[activation]


def named(name, module, *args, **kwargs):
    return type(name, (module,), {})(*args, **kwargs)


class BesselRBF(nn.Module):
    # Bessel RBF from Gasteiger et al. 2020
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


def constant_init(val):
    # We use our own constant init instead the one from jnn.initializers.constant
    # because of backwards compatability
    def init_fn(key, shape, dtype=jnp.float32):
        return jnp.full(shape, val, dtype=dtype)
    return init_fn
