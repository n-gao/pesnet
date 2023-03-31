"""
This file contains utility classes to deal with collections
of configurations. There are two types of collections, static
and dynamic configurations.
"""
from functools import cached_property
import numbers
from copy import deepcopy
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from pesnet import systems
from pesnet.utils import merge_dictionaries
from pesnet.utils.jax_utils import replicate


class ConfigCollection:
    def __init__(self, constructor) -> None:
        if isinstance(constructor, str):
            constructor = getattr(systems, constructor)
        self.constructor = constructor

    def update_configs(self):
        return

    def get_current_configs(self):
        return self.sub_configs

    def get_current_conf_vals(self):
        raise NotImplementedError()

    def get_current_systems(self):
        return [
            self.constructor(**c)
            for c in self.get_current_configs()
        ]

    def get_current_atoms(self, n_devices=None):
        result = systems_to_coords(self.get_current_systems())
        if n_devices is not None:
            return replicate(result)
        return result


class StaticConfigs(ConfigCollection):
    """
    A static configuration takes a constructor from `systems/__init__.py`
    and a config dictionary. The config dictionary is scanned for arrays,
    every element is then treated as a seperated configurations.
    If multiple arrays are present, it is assumed that all array values
    with the same index belong to the same configurations.
    Example for a static configurations of LiH containing 3 configs `diatomic`:
    {
        'symbol1': 'H',
        'symbol2': 'Li',
        'R': [
            0.7,
            0.9,
            1.1,
        ]
    }
    """

    def __init__(self, constructor, config, **kwargs) -> None:
        super().__init__(constructor)
        self.config = config
        self.gen_subconfigs()

    def gen_subconfigs(self):
        self.n_configs = -1
        for key, val in self.config.items():
            if isinstance(val, list):
                if self.n_configs == -1:
                    self.n_configs = len(val)
                elif self.n_configs != len(val):
                    raise ValueError()
        if self.n_configs == -1:
            self.n_configs = 1

        self.sub_configs = [{} for _ in range(self.n_configs)]
        for key, val in self.config.items():
            if isinstance(val, list):
                if len(self.sub_configs) == 0:
                    self.sub_configs = [{} for _ in range(len(val))]
                elif len(self.sub_configs) != len(val):
                    raise ValueError()
                for conf, value in zip(self.sub_configs, val):
                    conf[key] = value
            else:
                for conf in self.sub_configs:
                    conf[key] = val

    def get_current_conf_vals(self):
        result = {
            k: v for k, v in self.config.items()
            if isinstance(v, list) and
            all([isinstance(o, numbers.Number) for o in v])
        }
        result = {
            k: np.array(v).reshape(-1)
            for k, v in result.items()
        }
        return result


class DynamicConfigs(ConfigCollection):
    """
    A dynamic configuration takes a constructor from `systems/__init__.py`
    and a config dictionary. The config dictionary is scanned for dictionaries,
    each dictionary has to have 3 entries: `lower`, `upper` and `std`, describing
    the lower, upper bound for valid values as well as a standard deviation for
    random walkers. All dynamic properties then form a grid which is subdivided into
    `n_configs` cells. In each cell a random walker that randomly moves within its cell
    with every call of `update_configs`.
    An example for 16 walkers on the potential energy surface of LiH:
    constructor=diatomic
    config={
        'symbol1': 'H',
        'symbol2': 'LiH',
        'R': {
            'lower': 0.7,
            'upper': 3.5,
            'std': 0.1
        }
    }
    n_configs=16
    """

    def __init__(self, constructor, config, n_configs: int, deterministic: bool = False, **kwargs) -> None:
        super().__init__(constructor)
        self.config = config
        self.n_configs = n_configs
        self.deterministic = deterministic
        self.gen_subconfigs()

    def gen_subconfigs(self):
        lowers = []
        uppers = []
        stds = []
        self.keys = []
        self.static = {}
        for key, val in self.config.items():
            if isinstance(val, dict):
                upper, lower, std = val['upper'], val['lower'], val['std']
                lowers.append(lower)
                uppers.append(upper)
                stds.append(std)
                self.keys.append(key)
            else:
                self.static[key] = val

        self.lowers = np.array(lowers)
        self.uppers = np.array(uppers)
        self.stds = np.array(stds)
        if len(self.keys) == 0:
            self.values = None
            return
        assert round(self.n_configs**(1/len(self.keys))) % 1 == 0
        n_splits = int(np.around(self.n_configs**(1./len(self.keys)), 7)) + 1
        self.cell_limits = np.linspace(self.lowers, self.uppers, n_splits).T
        self.feature_bins = np.stack(np.meshgrid(*self.cell_limits), -1)
        max_slices = tuple(slice(1, None, 1) for _ in self.keys)
        min_slices = tuple(slice(0, -1, 1) for _ in self.keys)
        self.upper_bounds = self.feature_bins[max_slices]
        self.lower_bounds = self.feature_bins[min_slices]
        if self.deterministic:
            self.values = 0.5 * (self.upper_bounds -
                                 self.lower_bounds) + self.lower_bounds
        else:
            self.values = np.random.rand(
                *self.lower_bounds.shape) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def update_configs(self):
        if self.values is not None:
            updates = self.stds[tuple(
                None for _ in self.stds)] * np.random.randn(*self.values.shape)
            ranges = self.upper_bounds-self.lower_bounds
            updates = np.clip(updates, -ranges, ranges)
            self.values += updates
            self.values = np.where(
                self.values < self.lower_bounds,
                2*self.lower_bounds - self.values,
                self.values
            )
            self.values = np.where(
                self.values > self.upper_bounds,
                2*self.upper_bounds - self.values,
                self.values
            )

    @property
    def sub_configs(self):
        result = [deepcopy(self.static) for _ in range(self.n_configs)]
        if self.values is not None:
            vals = self.values.reshape(self.n_configs, len(self.keys))
            for j, k in enumerate(self.keys):
                for i, r in enumerate(result):
                    r[k] = vals[i, j]
        return result

    def get_current_conf_vals(self):
        result = {}
        for i, k in enumerate(self.keys):
            result[k] = self.values[..., i].reshape(-1)
        return result


class JointCollection:
    """
    A collection of collections. The subcollections are merged and treated as one large collection.
    This is useful if one wants to merge multiple static and dynamic configs.

    An example for two ethanol states
    constructor='ethanol'
    # Config is merged with the sub_systems
    config={
        'angle': {
            'lower': 0,
            'upper': 360,
            'std': 2
        }
    }
    # subconfigs are merged with the general config
    sub_systems=[
        {
            'config': {
                'state': 'Gauche'
            }
        },
        {
            'config': {
                'state': 'Trans'
            }
        },
    ]
    n_configs=16
    """
    def __init__(self, collections: Tuple[ConfigCollection, ...]) -> None:
        self.collections = collections

    def update_configs(self):
        for col in self.collections:
            col.update_configs()

    def get_current_configs(self):
        result = []
        for col in self.collections:
            result += col.get_current_configs()
        return result

    def get_current_conf_vals(self):
        result = {}
        for i, col in enumerate(self.collections):
            for k, v in col.get_current_conf_vals().items():
                result[f'{i}/{k}'] = v
        return result

    def get_current_systems(self):
        result = []
        for col in self.collections:
            result += col.get_current_systems()
        return result

    def get_current_atoms(self, n_devices=None):
        result = []
        for col in self.collections:
            result.append(col.get_current_atoms(None))
        result = jnp.concatenate(result, axis=0)
        if n_devices is not None:
            return replicate(result)
        return result
    
    @cached_property
    def sub_system_counts(self):
        return [len(col.get_current_configs()) for col in self.collections]


def make_system_collection(constructor, collection_type=None, **kwargs):
    if collection_type is None or collection_type.lower() == 'dynamic':
        return DynamicConfigs(constructor, **kwargs)
    elif collection_type.lower() == 'static':
        return StaticConfigs(constructor, **kwargs)
    elif collection_type.lower() == 'joint':
        fwd_args = deepcopy(kwargs)
        del fwd_args['sub_systems']
        return JointCollection([make_system_collection(constructor, **merge_dictionaries(fwd_args, sub_conf)) for sub_conf in kwargs['sub_systems']])
    else:
        raise ValueError()


def systems_to_coords(systems):
    return jnp.array(np.stack([
        s.coords
        for s in systems
    ], axis=0))
