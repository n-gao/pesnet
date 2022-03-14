import functools
from collections import namedtuple
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pesnet.ferminet import FermiNet
from pesnet.gnn import GNN, GNNPlaceholder
from pesnet.nn import MLP, ParamTree
from pesnet.utils import get_pca_axes

TreeFilter = Dict[Any, Union[None, Any, 'TreeFilter']]


def ravel_chunked_tree(tree: ParamTree, chunk_size: int = -1) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], ParamTree]]:
    """This function takes a ParamTree and flattens it into a two day array
    with chunk_size as the first dimension. If chunk_size is < 0 the first dimension of
    the first array is taken.

    Args:
        tree (ParamTree): ParamTree to ravel.
        chunk_size (int, optional): Chunk size. Defaults to -1.

    Returns:
        Tuple[jnp.ndarray, Callable[[jnp.ndarray], ParamTree]]: chunked array, unravel function
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

    def unravel(data: jnp.ndarray) -> ParamTree:
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


def extract_from_tree(tree: ParamTree, filters: TreeFilter, chunk_size: int = None) -> Tuple[ParamTree, ParamTree, Callable[[ParamTree, ParamTree], ParamTree]]:
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
        tree (ParamTree): Parameters
        filters (TreeFilter): Filter dictionary
        chunk_size (int, optional): Chunk size - useful to extract node features. Defaults to None.

    Returns:
        Tuple[ParamTree, ParamTree, Callable[[ParamTree, ParamTree], ParamTree]]: parameters without the extracted, extracted, reconstruct function
    """
    if chunk_size is not None:
        ravel_fn = functools.partial(ravel_chunked_tree, chunk_size=chunk_size)
    else:
        ravel_fn = ravel_pytree

    def _extract_from_tree(tree: ParamTree, filters: TreeFilter) -> Tuple[ParamTree, ParamTree]:
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
            tree: ParamTree,
            filters: TreeFilter,
            replacements: Iterable[jnp.ndarray],
            unravel_fn: Iterable[Callable[[jnp.ndarray], jnp.ndarray]]) -> ParamTree:
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

    def reconstruct(tree: ParamTree, replacements: jnp.ndarray, copy: bool = False) -> ParamTree:
        if copy:
            tree = deepcopy(tree)
        if replacements is None:
            return tree
        return _reconstruct(tree, filters, iter(replacements), unravel_fn=iter(unravel_fns))

    return new_tree, extracted, reconstruct


def update_parameter_initialization(
        weight: jnp.ndarray,
        bias: jnp.ndarray,
        flat_original: jnp.ndarray,
        original: List[jnp.ndarray],
        weight_bias_ratio: Tuple[float, float] = (1, 1)) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Rescales the weights and biases such that the distribution of the output matches 
    the distribution of the original parameters. If the original parameters are initialized
    to fixed numbers (e.g., 0 or 1) we try to match them up to some small noise.

    Args:
        weight (jnp.ndarray): Weight matrix
        bias (jnp.ndarray): Bias vector
        flat_original (jnp.ndarray): Original parameters as single array
        original (List[jnp.ndarray]): Original parameters as list
        weight_bias_ratio (Tuple[float, float], optional): Ratio between weights and biases. Defaults to (1, 1).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: new weights, new bias
    """
    n_nodes = flat_original.shape[0] if flat_original.ndim > 1 else 1
    chunks = [o.size//n_nodes for o in original]
    chunks = [0] + [c for c in chunks if c > 0]
    chunks = np.cumsum(chunks)
    for i in range(len(chunks)-1):
        chunk = flat_original[..., chunks[i]:chunks[i+1]]
        index = jax.ops.index[..., chunks[i]:chunks[i+1]]
        if (jnp.mod(chunk, 1) == 0).all():
            # We repalce the biases with the correct terms and set the weights to a
            # very small initialization such that their influence to the result is minor
            bias = jax.ops.index_update(
                bias, index, chunk[0] if chunk.ndim > 1 else chunk)
            weight = jax.ops.index_mul(weight, index, 1e-1)
        else:
            total_weight = sum(weight_bias_ratio)
            weight_weight = jnp.sqrt(total_weight/weight_bias_ratio[0])
            bias_weight = jnp.sqrt(total_weight/weight_bias_ratio[1])
            base_std = bias.std()
            c_std = chunk.std()
            weight = jax.ops.index_mul(weight, index, c_std/weight_weight)
            bias = jax.ops.index_mul(bias, index, c_std/bias_weight)
    return weight, bias


def merge_filters(filter1: dict, filter2: dict) -> dict:
    """Merge to filter dictionaries. If two values coolide,
    filter 2 overrides filter 1.

    Args:
        filter1 (dict): Filter 1
        filter2 (dict): Filter 2

    Returns:
        dict: Merged filter of 1 and 2
    """
    result = deepcopy(filter1)
    for key, val in filter2.items():
        if key not in result:
            result[key] = val
        elif isinstance(val, dict) and isinstance(result[key], dict):
            result[key] = merge_filters(result[key], val)
        elif isinstance(val, list) and isinstance(result[key], list):
            result[key] = merge_filters(result[key], val)
        else:
            result[key] = val
    return result


def get_default_filters(
    fermi_params: ParamTree
):
    """Utility function to select the minimum set of parameters from fermi_params that
    must be extracted to ensure that we capture the right symmetries.

    Args:
        fermi_params (ParamTree): FermiNet parametesr.
        decomposition (ParamTree): FermiNet decomposition of the first player.
        support_vectors (bool, optional): Whether vectors are supported - not used right now. Defaults to False.

    Returns:
        Tuple[dict, dict, dict, dict]: Filters for nodes, edges, global, node vectors and global vetors
    """
    node_filter = {
        'params': {
            'to_orbitals': {
                '$': [
                    {
                        f'IsotropicEnvelope_{i}': {'pi': None}
                        for i in range(2)
                    },
                ],
                '$$': [
                    {
                        f'IsotropicEnvelope_{i}': {'sigma': None}
                        for i in range(2)
                    },
                ],
            },
            'input_construction': {
                'nuc_embedding': None
            }
        }
    }
    return node_filter


def find_axes(atoms: jnp.ndarray, charges: jnp.ndarray) -> jnp.ndarray:
    """Generates equivariant axes based on PCA.

    Args:
        atoms (jnp.ndarray): (..., M, 3)
        charges (jnp.ndarray): (M)

    Returns:
        jnp.ndarray: (3, 3)
    """
    atoms = jax.lax.stop_gradient(atoms)
    # First compute the axes by PCA
    atoms = atoms - atoms.mean(-2, keepdims=True)
    s, axes = get_pca_axes(atoms, charges)
    # Let's check whether we have identical eigenvalues
    # if that's the case we need to work with soem pseudo positions
    # to get unique atoms.
    is_ambiguous = jnp.count_nonzero(
        jnp.unique(s, size=3, fill_value=0)) < jnp.count_nonzero(s)
    # We always compute the pseudo coordinates because it causes some compile errors
    # for some unknown reason on A100 cards with jax.lax.cond.
    # Compute pseudo coordiantes based on the vector inducing the largest coulomb energy.
    distances = atoms[None] - atoms[..., None, :]
    dist_norm = jnp.linalg.norm(distances, axis=-1)
    coulomb = charges[None] * charges[:, None] / dist_norm
    off_diag_mask = ~np.eye(atoms.shape[0], dtype=np.bool)
    coulomb, distances = coulomb[off_diag_mask], distances[off_diag_mask]
    idx = jnp.argmax(coulomb)
    scale_vec = distances[idx]
    scale_vec /= jnp.linalg.norm(scale_vec)
    # Projected atom positions
    proj = atoms@scale_vec[..., None] * scale_vec
    diff = atoms - proj
    pseudo_atoms = proj * (1+1e-4) + diff

    pseudo_s, pseudo_axes = get_pca_axes(pseudo_atoms, charges)

    # Select pseudo axes if it is ambiguous
    s = jnp.where(is_ambiguous, pseudo_s, s)
    axes = jnp.where(is_ambiguous, pseudo_axes, axes)

    order = jnp.argsort(s)[::-1]
    axes = axes[:, order]

    # Compute an equivariant vector
    distances = jnp.linalg.norm(atoms[None] - atoms[..., None, :], axis=-1)
    weights = distances.sum(-1)
    equi_vec = ((weights * charges)[..., None] * atoms).mean(0)

    ve = equi_vec@axes
    flips = ve < 0
    axes = jnp.where(flips[None], -axes, axes)

    right_hand = jnp.stack(
        [axes[:, 0], axes[:, 1], jnp.cross(axes[:, 0], axes[:, 1])], axis=1)
    axes = jnp.where(jnp.abs(ve[-1]) < 1e-7, right_hand, axes)
    return axes


PESNetFunctions = namedtuple(
    'PESNetFunctions', [
        'ferminet',
        'gnn',
        'get_fermi_params',
        'pesnet_fwd',
        'pesnet_orbitals',
        'extract_node_params',
        'extract_global_params',
        'update_axes',
    ])


def make_pesnet(
    key,
    charges: Tuple[int, ...],
    spins: Tuple[int, int],
    gnn_params,
    ferminet_params,
    node_filter: dict = {},
    global_filter: dict = {},
    include_default_filter: bool = True,
    **kwargs
) -> Tuple[ParamTree, PESNetFunctions]:
    """Generate PESNet with parameters and all relevant functions.

    Args:
        key (jnp.ndarray): jax.random.PRNGKey
        charges (Tuple[int, ...]): (M)
        spins (Tuple[int, int]): (spin_up, spin_down)
        gnn_params (dict): GNN initialization parameters 
        ferminet_params (dict): FermiNet initialization parameters
        node_filter (dict, optional): Node filter. Defaults to {}.
        global_filter (dict, optional): Global filter. Defaults to {}.
        include_default_filter (bool, optional): Whether to include the default filters. Defaults to True.

    Returns:
        Tuple[ParamTree, PESNetFucntions]: parameters, PESNet functions
    """
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
        len(charges),
        spins,
        **ferminet_params
    )

    batched_fermi = jax.vmap(ferminet.apply, in_axes=(None, 0, None))
    batched_orbitals = jax.vmap(
        functools.partial(ferminet.apply, method=ferminet.orbitals),
        in_axes=(None, 0, None)
    )

    # Initialization - we have to construct some toy data
    atoms = jnp.ones((len(charges), 3))
    electrons = jnp.zeros((sum(spins)*3))
    key, subkey = jax.random.split(key)
    fermi_params = ferminet.init(subkey, electrons, atoms).unfreeze()

    # Construct default filters
    if include_default_filter:
        default_node_filter = get_default_filters(fermi_params)
        node_filter = merge_filters(default_node_filter, node_filter)

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
        gnn_params = {
            **gnn_params,
            'node_out_dims': n_params,
            'global_out_dims': g_params,
        }
        gnn = GNN(
            charges=charges,
            **gnn_params
        )
    else:
        gnn = GNNPlaceholder()
    key, subkey = jax.random.split(key)
    gnn_params = gnn.init(subkey, atoms).unfreeze()

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
                (1, 2)
            )

    # Construct complete parameters
    params = {
        'gnn': gnn_params,
        'ferminet': fermi_params,
    }

    def update_axes(params, atoms):
        if not use_meta:
            return params
        axes = find_axes(atoms, jnp.array(charges))
        params['gnn']['constants']['axes'] = axes
        params['ferminet']['constants']['axes'] = axes
        return params

    def get_fermi_params(params, atoms):
        params = update_axes(params, atoms)
        gnn_params, fermi_params = params['gnn'], params['ferminet']
        node_features, global_features = gnn.apply(
            gnn_params,
            atoms
        )
        fermi_params = node_recover_fn(fermi_params, node_features)
        fermi_params = global_reocver_fn(fermi_params, global_features)
        return fermi_params

    def pesnet(params, electrons, atoms):
        # We expect electrons to be batched!!!
        fermi_params = get_fermi_params(params, atoms)
        return batched_fermi(fermi_params, electrons, atoms)

    def pesnet_orbitals(params, electrons, atoms):
        # We expect electrons to be batched!!!
        fermi_params = get_fermi_params(params, atoms)
        return batched_orbitals(fermi_params, electrons, atoms)

    def extract_node_params(params):
        new_tree, extracted = extract_from_tree(
            params, node_filter, respect_chunks=True)[:2]
        flat_extracted = ravel_chunked_tree(extracted, n_atoms)[0]
        return new_tree, flat_extracted

    def extract_global_params(params):
        new_tree, extracted = extract_from_tree(params, global_filter)[:2]
        flat_extracted = ravel_pytree(extracted)[0]
        return new_tree, flat_extracted

    result = PESNetFunctions(
        ferminet=ferminet,
        gnn=gnn,
        get_fermi_params=get_fermi_params,
        pesnet_fwd=pesnet,
        pesnet_orbitals=pesnet_orbitals,
        extract_node_params=extract_node_params,
        extract_global_params=extract_global_params,
        update_axes=update_axes,
    )
    return params, result
