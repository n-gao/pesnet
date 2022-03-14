"""
This file contains utility classes to deal with collections
of configurations. There are two types of collections, static
and dynamic configurations.
"""
import numbers
from copy import deepcopy

import jax.numpy as jnp
import numpy as np


class ConfigCollection:
    """
    Interface for groups of molecular configurations.
    """
    def __init__(self, constructor) -> None:
        # Constructor is some function which returns an molecule.
        self.constructor = constructor

    def update_configs(self):
        return

    def get_current_configs(self):
        return self.sub_configs

    def get_current_histograms(self, n_bins):
        raise NotImplementedError()

    def get_current_systems(self):
        return [
            self.constructor(**c)
            for c in self.get_current_configs()
        ]

    def get_current_atoms(self, n_devices=None):
        # Returns the atom positions of all current configs
        # and formats the tensor such that it is directly useable in
        # pmapped functions.
        result = systems_to_coords(self.get_current_systems())
        if n_devices is not None:
            assert self.n_configs % n_devices == 0
            return result.reshape(n_devices, self.n_configs//n_devices, -1, 3)
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

    def get_current_histograms(self, n_bins):
        result = {
            k: v for k, v in self.config.items()
            if isinstance(v, list) and
            all([isinstance(o, numbers.Number) for o in v])
        }
        result = {
            k: {
                'bins': np.linspace(np.min(v), np.max(v), n_bins),
                'values': np.array(v).reshape(-1)
            }
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

    def __init__(self, constructor, config, n_configs: int, **kwargs) -> None:
        super().__init__(constructor)
        self.config = config
        self.n_configs = n_configs
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
        assert (self.n_configs**(1./len(self.keys))) % 1 == 0
        n_splits = int(self.n_configs ** (1./len(self.keys))) + 1
        self.cell_limits = np.linspace(self.lowers, self.uppers, n_splits).T
        self.feature_bins = np.stack(np.meshgrid(*self.cell_limits), -1)
        max_slices = tuple(slice(1, None, 1) for _ in self.keys)
        min_slices = tuple(slice(0, -1, 1) for _ in self.keys)
        self.upper_bounds = self.feature_bins[max_slices]
        self.lower_bounds = self.feature_bins[min_slices]
        self.values = (self.upper_bounds - self.lower_bounds) / \
            2 + self.lower_bounds

    def update_configs(self):
        """
        Performs an update step where we move each config within its grid cell
        by some random pertrubation.
        """
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

    def get_current_histograms(self, n_bins):
        result = {}
        for i, k in enumerate(self.keys):
            bins = np.linspace(self.lowers[i], self.uppers[i], n_bins)
            result[k] = {
                'bins': bins,
                'values':  self.values[..., i].reshape(-1)
            }
        return result


def make_system_collection(constructor, collection_type=None, **kwargs):
    if collection_type is None or collection_type.lower() == 'dynamic':
        return DynamicConfigs(constructor, **kwargs)
    elif collection_type.lower() == 'static':
        return StaticConfigs(constructor, **kwargs)
    else:
        raise ValueError()


def systems_to_coords(systems):
    return jnp.stack([
        s.coords()
        for s in systems
    ], axis=0)
