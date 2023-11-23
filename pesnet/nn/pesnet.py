import functools
from copy import deepcopy
from typing import (Any, Callable, Dict, Iterable, List, NamedTuple, Tuple,
                    Union)

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree, PRNGKey
from flax.core import unfreeze
from jax.flatten_util import ravel_pytree

from pesnet.nn import MLP
from pesnet.nn.coords import find_axes
from pesnet.nn.dimenet import DimeNet
from pesnet.nn.ferminet import FermiNet
from pesnet.nn.gnn import GNN, GNNPlaceholder
from pesnet.utils import merge_dictionaries
from pesnet.utils.typing import (OrbitalFunction, ParameterFunction,
                                 Parameters, WaveFunction)

TreeFilter = Dict[Any, Union[None, Any, 'TreeFilter']]


def ravel_chunked_tree(tree: ArrayTree, chunk_size: int = -1) -> Tuple[jax.Array, Callable[[jax.Array], ArrayTree]]:
    """This function takes a ParamTree and flattens it into a two day array
    with chunk_size as the first dimension. If chunk_size is < 0 the first dimension of
    the first array is taken.

    Args:
        tree (ArrayTree): Parameter tree to ravel.
        chunk_size (int, optional): Chunk size. Defaults to -1.

    Returns:
        Tuple[jax.Array, Callable[[jax.Array], ArrayTree]]: chunked array, unravel function
    """
    # This function takes a ParamTree and flattens it into a two day array
    # with chunk_size as the first dimension. If chunk_size is < 0 the first dimension of
    # the first array is taken.
    xs, tree_struc = jax.tree_util.tree_flatten(tree)
    shapes = [x.shape for x in xs]
    chunk_size = xs[0].shape[0] if chunk_size <= 0 else chunk_size
    xs = [x.reshape(chunk_size, -1) for x in xs]
    sizes = [x.shape[1] for x in xs]
    indices = np.cumsum(sizes)

    def unravel(data: jax.Array) -> ArrayTree:
        if data is None or data.size == 0:
            return {}
        if data.ndim == 0:
            return None
        data = data.reshape(chunk_size, -1)
        chunks = jnp.split(data, indices, axis=1)
        xs = [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]
        return jax.tree_util.tree_unflatten(tree_struc, xs)

    if len(xs) == 0:
        return jnp.zeros((chunk_size, 0)), unravel
    result = jnp.concatenate(xs, axis=1).reshape(chunk_size, -1)
    return result, unravel


def extract_from_tree(tree: ArrayTree, filters: TreeFilter, chunk_size: int = None) -> Tuple[ArrayTree, ArrayTree, Callable[[ArrayTree, ArrayTree], ArrayTree]]:
    """Extarcts parameters from a parameter tree.
    Returns a copy of the tree without the filtered parameters.
    The filter is a dict:
    {
        'single': {
            0: 'W'
        },
        'envelope': None
    }
    Values of None indicate that the tree from the key should be extracted.
    If the value is not a dict it is interpreted as the laster filter step.

    Args:
        tree (ArrayTree): Parameters
        filters (TreeFilter): Filter dictionary
        chunk_size (int, optional): Chunk size - useful to extract node features. Defaults to None.

    Returns:
        Tuple[ArrayTree, ArrayTree, Callable[[ArrayTree, ArrayTree], ArrayTree]]: parameters without the extracted, extracted, reconstruct function
    """
    if chunk_size is not None:
        ravel_fn = functools.partial(ravel_chunked_tree, chunk_size=chunk_size)
    else:
        ravel_fn = ravel_pytree

    def _extract_from_tree(tree: ArrayTree, filters: TreeFilter) -> Tuple[ArrayTree, ArrayTree]:
        unravel_fns = []
        out = []
        for k, f in filters.items():
            if len(tree) <= k if isinstance(k, int) else (k not in tree and not k.startswith('$')):
                continue
            # Recursively extract items
            if isinstance(f, dict):
                out_, unravel_ = _extract_from_tree(tree[k], f)[1:]
                unravel_fns = unravel_fns + unravel_
                out = out + out_
            elif isinstance(f, list):
                outs = []
                unravels = []
                for item in f:
                    out_, unravel_ = _extract_from_tree(
                        tree if k.startswith('$') else tree[k], item)[1:]
                    outs += out_
                    unravels += unravel_
                out_, unravel_ = ravel_fn(outs)
                if out_.size > 0:
                    unravel_fns.append(unravel_)
                    unravel_fns += unravels
                    out.append(out_)
                else:
                    unravel_fns.append(None)
            # We are at a leaf
            elif f is None:
                assert tree[k] is not None
                result, unravel = ravel_fn(tree[k])
                if result.size > 0:
                    out.append(result)
                    unravel_fns.append(unravel)
                    tree[k] = None
                else:
                    unravel_fns.append(None)
            # Simpler leaf formulation
            else:
                assert tree[k][f] is not None
                result, unravel = ravel_fn(tree[k][f])
                if result.size > 0:
                    out.append(result)
                    unravel_fns.append(unravel)
                    tree[k][f] = None
                else:
                    unravel_fns.append(None)
        return tree, out, unravel_fns

    new_tree, extracted, unravel_fns = _extract_from_tree(
        deepcopy(tree), filters)

    def _reconstruct(
            tree: ArrayTree,
            filters: TreeFilter,
            replacements: Iterable[jax.Array],
            unravel_fn: Iterable[Callable[[jax.Array], jax.Array]]) -> ArrayTree:
        # This function uses that the tree traversal of the filters
        # does not change.
        # So, we can use iterators for the replacements and unravel functions and do not have
        # to keep track of any indices.
        for k, f in filters.items():
            if len(tree) <= k if isinstance(k, int) else (k not in tree and not k.startswith('$')):
                continue
            # Recursively set items
            if isinstance(f, dict):
                _reconstruct(tree[k], filters[k], replacements, unravel_fn)
            elif isinstance(f, list):
                unravel = next(unravel_fn)
                if unravel is not None:
                    tmp_replacements = iter(unravel(next(replacements)))
                    for i in range(len(f)):
                        _reconstruct(
                            tree if k.startswith('$') else tree[k],
                            f[i],
                            tmp_replacements,
                            unravel_fn
                        )
            # If we are at a leaf reconstruct the old format
            elif f is None:
                unravel = next(unravel_fn)
                if unravel is not None:
                    tree[k] = unravel(next(replacements))
            else:
                unravel = next(unravel_fn)
                if unravel is not None:
                    tree[k][f] = unravel(next(replacements))
        return tree

    def reconstruct(tree: ArrayTree, replacements: jax.Array, copy: bool = False) -> ArrayTree:
        if copy:
            tree = deepcopy(tree)
        if replacements is None:
            return tree
        return _reconstruct(tree, filters, iter(replacements), unravel_fn=iter(unravel_fns))

    return new_tree, extracted, reconstruct


def update_parameter_initialization(
        weight: jax.Array,
        bias: jax.Array,
        flat_original: jax.Array,
        original: List[jax.Array],
        charges: Tuple[int, ...],
        weight_bias_ratio: Tuple[float, float] = (1, 1)) -> Tuple[jax.Array, jax.Array]:
    """Rescales the weights and biases such that the distribution of the output matches 
    the distribution of the original parameters. If the original parameters are initialized
    to fixed numbers (e.g., 0 or 1) we try to match them up to some small noise.

    Args:
        weight (jax.Array): Weight matrix
        bias (jax.Array): Bias vector
        flat_original (jax.Array): Original parameters as single array
        original (List[jax.Array]): Original parameters as list
        weight_bias_ratio (Tuple[float, float], optional): Ratio between weights and biases. Defaults to (1, 1).

    Returns:
        Tuple[jax.Array, jax.Array]: new weights, new bias
    """
    n_nodes = flat_original.shape[0] if flat_original.ndim > 1 else 1
    chunks = [o.size//n_nodes for o in original]
    chunks = [0] + [c for c in chunks if c > 0]
    chunks = np.cumsum(chunks)
    for i in range(len(chunks)-1):
        idx = slice(chunks[i], chunks[i+1])
        chunk = flat_original[..., idx]
        if len(np.unique(chunk, return_counts=True)[1]) < (chunk.size/2):
            # We repalce the biases with the correct terms and set the weights to a
            # very small initialization such that their influence to the result is minor
            if chunk.ndim == 2:
                # We assume that this applies to the envelope parameters
                for j, b in zip(charges, chunk):
                    bias = bias.at[..., j, idx].set(b)
            else:
                bias = bias.at[..., idx].set(chunk)
            weight = weight.at[..., idx].multiply(1e-1)
        else:
            total_weight = sum(weight_bias_ratio)
            kernel_weight = jnp.sqrt(total_weight/weight_bias_ratio[0])
            bias_weight = jnp.sqrt(total_weight/weight_bias_ratio[1])
            base_std = bias.std()
            c_std = chunk.std()
            weight = weight.at[..., idx].multiply(c_std/kernel_weight)
            bias = bias.at[..., idx].multiply(c_std/bias_weight)
    return weight, bias


DEFAULT_FILTER = {
    'params': {
        'to_orbitals': {
            f'IsotropicEnvelope_{i}': {
                'sigma': None,
                'pi': None
            }
            for i in range(2)
        },
        'input_construction': {
            'nuc_embedding': None,
            'nuc_bias': None,
        }
    }
}


class PESNet(NamedTuple):
    ferminet: FermiNet
    gnn: nn.Module
    get_fermi_params: ParameterFunction
    pesnet_fwd: WaveFunction
    pesnet_orbitals: OrbitalFunction
    update_axes: ParameterFunction


def make_pesnet(
    key: PRNGKey,
    charges: Tuple[int, ...],
    spins: Tuple[int, int],
    gnn_params: dict,
    dimenet_params: dict,
    ferminet_params: dict,
    node_filter: dict = {},
    global_filter: dict = {},
    include_default_filter: bool = True,
    meta_model: str = 'gnn',
    **kwargs
) -> Tuple[Parameters, PESNet]:
    """Generate PESNet with parameters and all relevant functions.

    Args:
        key (jax.Array): jax.random.PRNGKey
        charges (Tuple[int, ...]): (M)
        spins (Tuple[int, int]): (spin_up, spin_down)
        gnn_params (dict): GNN initialization parameters 
        ferminet_params (dict): FermiNet initialization parameters
        node_filter (dict, optional): Node filter. Defaults to {}.
        global_filter (dict, optional): Global filter. Defaults to {}.
        include_default_filter (bool, optional): Whether to include the default filters. Defaults to True.

    Returns:
        Tuple[ArrayTree, PESNetFucntions]: parameters, PESNet functions
    """
    meta_model = meta_model.lower()
    assert meta_model in ['gnn', 'dimenet']
    # In case one sets a string we evaluate that string
    if isinstance(node_filter, str):
        node_filter = eval(node_filter)
    if isinstance(global_filter, str):
        global_filter = eval(global_filter)

    # Be safe that one does not use None as value
    node_filter = {} if node_filter is None else node_filter
    global_filter = {} if global_filter is None else global_filter

    n_atoms = len(charges)
    ferminet = FermiNet(
        charges,
        spins,
        **ferminet_params
    )

    # Initialization - we have to construct some toy data
    atoms = jnp.ones((len(charges), 3))
    electrons = jnp.zeros((sum(spins)*3))
    key, subkey = jax.random.split(key)
    fermi_params = unfreeze(ferminet.init(subkey, electrons, atoms))

    # Construct default filters
    if include_default_filter:
        node_filter = merge_dictionaries(DEFAULT_FILTER, node_filter)

    # Extract node features
    fermi_params, node_param_list, node_recover_fn = extract_from_tree(
        fermi_params, node_filter, chunk_size=n_atoms)
    n_params = [p.size//n_atoms for p in node_param_list]

    # Extract global features
    fermi_params, global_param_list, global_reocver_fn = extract_from_tree(
        fermi_params, global_filter, chunk_size=None)
    g_params = [p.size for p in global_param_list]

    use_meta = sum(n_params) + sum(g_params) > 0
    if use_meta:
        if meta_model == 'gnn':
            gnn = GNN(
                charges=charges,
                node_out_dims=n_params,
                global_out_dims=g_params,
                **gnn_params
            )
        elif meta_model == 'dimenet':
            gnn = DimeNet(
                node_out_dims=n_params,
                global_out_dims=g_params,
                charges=charges,
                **dimenet_params,
            )
    else:
        gnn = GNNPlaceholder()
    key, subkey = jax.random.split(key)
    gnn_params = unfreeze(gnn.init(subkey, atoms))

    # We need to reinitialize these arrays such that specific initializations are preserved
    # e.g., the standard deviation of parameters or specifically initialized varibles (e.g. to one or identity)
    if use_meta:
        for i in range(len(node_param_list)):
            node_out = gnn_params['params'][f'NodeOut_{i}']
            embed = node_out['Embed_0']
            mlp = node_out['AutoMLP_0']
            last_layer = MLP.extract_final_linear(mlp)
            last_layer['kernel'], embed['embedding'] = update_parameter_initialization(
                last_layer['kernel'],
                embed['embedding'],
                node_param_list[i],
                [node_param_list[i]],
                charges,
                (1, 2)
            )

        for i in range(len(global_param_list)):
            mlp = gnn_params['params'][f'GlobalOut_{i}']
            last_layer = MLP.extract_final_linear(mlp)
            last_layer['kernel'], last_layer['bias'] = update_parameter_initialization(
                last_layer['kernel'],
                last_layer['bias'],
                global_param_list[i],
                [global_param_list[i]],
                charges,
                (1, 2)
            )

    # Construct complete parameters
    params = {
        'gnn': gnn_params,
        'ferminet': fermi_params
    }
    if 'constants' in params['gnn']:
        params['gnn']['constants']['axes'] = None
    params['ferminet']['constants']['axes'] = None

    find_axes_fn = functools.partial(find_axes, charges=jnp.array(charges))
    def update_axes(params, atoms, use_frame: bool = True):
        if use_frame:
            axes = find_axes_fn(atoms)
        else:
            axes = jnp.eye(3)
        if 'constants' in params['gnn']:
            params['gnn']['constants']['axes'] = axes
        params['ferminet']['constants']['axes'] = axes
        return params

    def get_fermi_params(params, atoms, use_frame: bool = True):
        params = update_axes(params, atoms, use_frame)
        gnn_params, fermi_params = params['gnn'], params['ferminet']
        node_features, global_features = gnn.apply(
            gnn_params,
            atoms
        )
        fermi_params = node_recover_fn(fermi_params, node_features)
        fermi_params = global_reocver_fn(fermi_params, global_features)
        return fermi_params

    def pesnet(params, electrons, atoms, use_frame: bool = True):
        fermi_params = get_fermi_params(params, atoms, use_frame)
        return ferminet.apply(fermi_params, electrons, atoms)

    def pesnet_orbitals(params, electrons, atoms, use_frame: bool = True):
        fermi_params = get_fermi_params(params, atoms, use_frame)
        return ferminet.apply(fermi_params, electrons, atoms, method=ferminet.orbitals)

    result = PESNet(
        ferminet=ferminet,
        gnn=gnn,
        get_fermi_params=get_fermi_params,
        pesnet_fwd=pesnet,
        pesnet_orbitals=pesnet_orbitals,
        update_axes=update_axes
    )
    return params, result
