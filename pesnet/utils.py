import datetime
import json
import math
import os
import shutil
import time
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import jaxboard
import numpy as np
import tqdm
from uncertainties import ufloat

from pesnet.jax_utils import broadcast, replicate
from pesnet.systems.collection import ConfigCollection
from pesnet.training import init_electrons


class MCMCStepSizeScheduler:
    """Utility class to schedule the proposal width of the MCMC.
    """

    def __init__(
            self,
            init_width: jnp.ndarray,
            target_pmove: float = 0.525,
            error: float = 0.025,
            update_interval: int = 20,
            **kwargs) -> None:
        if isinstance(init_width, jnp.ndarray):
            self.width = init_width
        else:
            self.width = jnp.array(init_width)
        self.update_interval = update_interval
        self.target_pmove = target_pmove
        self.error = error
        self.pmoves = np.zeros((update_interval, *self.width.shape))
        self.i = 0

    def __call__(self, pmove: float) -> float:
        if self.i % self.update_interval == 0 and self.i > 0:
            pm_mean = self.pmoves.mean(0)
            self.width = np.where(pm_mean < self.target_pmove - self.error,
                                  self.width / 1.1, self.width)
            self.width = np.where(pm_mean > self.target_pmove + self.error,
                                  self.width * 1.1, self.width)
        self.pmoves[self.i % self.update_interval] = pmove
        self.i += 1
        return self.width


class Logger:
    """Logger utility class. This is essentially a wrapper around
    jaxboard with utility functions to log distributions.F
    """

    def __init__(
            self,
            name: str,
            config: dict = None,
            folder: str = 'logs',
            subfolder: str = None) -> None:
        time_str = datetime.datetime.now().strftime(r'%d-%m-%y_%H:%M:%S:%f')
        self.name = name
        self.folder_name = folder
        if subfolder is not None:
            self.folder_name = os.path.join(self.folder_name, subfolder)
        self.folder_name = os.path.join(
            self.folder_name, f'{name}__{time_str}')
        self.writer = jaxboard.SummaryWriter(self.folder_name)
        if config is not None:
            self.writer.text(
                'config', f'```\n{json.dumps(config, indent=2, sort_keys=True)}\n```')

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)

    def log_params(self, params, path='', n_bins=20, step=None):
        if isinstance(params, dict):
            for k, v in params.items():
                self.log_params(v, f'{path}/{k}', n_bins=n_bins, step=step)
        elif isinstance(params, list):
            for i, v in enumerate(params):
                self.log_params(v, f'{path}/{i}', n_bins=n_bins, step=step)
        elif params is None:
            return
        else:
            data = params[0]
            bins = np.linspace(np.min(data), np.max(data), n_bins)
            self.histogram(
                path, bins=bins, values=data, step=step
            )

    def log_electrons(self, electrons, path='electrons', n_bins=20, step=None):
        electrons = electrons.reshape(-1, 3)
        for i, label in enumerate('xyz'):
            values = electrons[:, i]
            bins = np.linspace(np.min(values), np.max(values), n_bins)
            self.histogram(
                f'{path}/{label}', bins=bins, values=values, step=step
            )


class ExponentiallyMovingAverage:
    """A simple implementation of an exponentially moving average.
    """

    def __init__(self) -> None:
        self._value = 0.
        self._total_weight = 0.

    def update(self, value, decay):
        self._value *= decay
        self._total_weight *= decay
        self._value += value
        self._total_weight += 1.

    @property
    def value(self):
        return self._value / self._total_weight


class EMAPyTree:
    """Exponentially moving averages for JAX Pytrees.
    """

    def __init__(self, tree=None) -> None:
        if tree is not None:
            self._value = jax.tree_map(lambda x: jnp.zeros_like(x), tree)
        else:
            self._value = None
        self._total_weight = 0.

        def update_fn(v1, v2, d):
            return jax.tree_multimap(lambda x, y: x*d + y, v1, v2)
        self.update_fn = jax.jit(update_fn)

        def value_fn(t, w):
            return jax.tree_map(lambda x: x/w, t)
        self.value_fn = jax.jit(value_fn)

    def update(self, value, decay):
        if self._value == None:
            self._value = jax.tree_map(lambda x: x, value)
        else:
            self._value = self.update_fn(self._value, value, decay)
        self._total_weight = self._total_weight * decay + 1

    @property
    def value(self):
        if self._value is None:
            return {}
        else:
            return self.value_fn(self._value, self._total_weight)


class OnlineMean():
    """Compute mean and standard deviations in an online fashion.
    """

    def __init__(self):
        self._mean = 0
        self.n = 0
        self.comoment = 0

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self.comoment/self.n

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def err(self):
        return self.std/np.sqrt(self.n)

    def update(self, x):
        self.n += len(x)
        last_mean = self._mean
        self._mean += (x-self._mean).sum()/self.n
        self.comoment += ((x - self._mean) * (x - last_mean)).sum()

    def __str__(self) -> str:
        return f'{ufloat(self.mean, self.err):S}'


def get_pca_axes(coords: jnp.ndarray, weights: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the (weighted) PCA of the given coordinates.

    Args:
        coords (jnp.ndarray): (N ,D)
        weights (jnp.ndarray, optional): (N). Defaults to None.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: (D), (D, D)
    """
    if weights is None:
        weights = jnp.ones((*coords.shape[:-1],))
    weights /= weights.sum()

    mean = (weights[..., None]*coords).mean()
    centered = coords - mean
    cov = centered.T @ jnp.diag(weights) @ centered

    s, vecs = jnp.linalg.eigh(cov)
    return s, vecs


class Stopwatch:
    """Utility class to measure time deltas.
    """

    def __init__(self) -> None:
        self.last_step = time.time()

    def __call__(self):
        new_time = time.time()
        result = new_time - self.last_step
        self.last_step = new_time
        return result


def eval_energy_parallel(
    key: jnp.ndarray,
    vmc,
    configs: ConfigCollection,
    total_samples: int,
    batch_size: int,
    logger: Logger,
    print_progress: bool,
) -> jnp.ndarray:
    """Evalautes the energy over `total_samples` of the provided configuration
    collection in parallel. This is generally the preferred method as it is
    significantly faster.

    Args:
        key (jnp.ndarray): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jnp.ndarray: Final energies of form (n_configs, total_samples)
    """
    val_atoms = broadcast(configs.get_current_atoms(jax.device_count()))
    sys = configs.get_current_systems()[0]
    key, subkey = jax.random.split(key)
    electrons = init_electrons(
        val_atoms,
        sys.charges(),
        sys.spins(),
        batch_size,
        subkey
    )
    electrons = vmc.thermalize_samples(
        electrons,
        val_atoms,
        5000,
        show_progress=print_progress,
        adapt_step_width=False
    )
    n_repeats = int(
        math.ceil(total_samples/electrons.shape[-2])
    )
    E_l = vmc.eval_energy(electrons, val_atoms, n_repeats, 5, print_progress)
    E_l = E_l.reshape(len(configs.get_current_systems()), -1)
    E_final = np.mean(E_l, axis=-1)
    E_final_std = np.std(E_l, axis=-1)
    E_final_err = E_final_std/np.sqrt(E_l.size/E_final.size)

    if logger is not None:
        for i, (E, std, err) in enumerate(zip(E_final, E_final_std, E_final_err)):
            logger.scalar(f'PES/E', E, step=i)
            logger.scalar(f'PES/E_std', std, step=i)
            logger.scalar(f'PES/E_err', err, step=i)
        logger.flush()
    return E_l, E_final, E_final_std, E_final_err


def eval_energy_sequential(
    key: jnp.ndarray,
    vmc,
    configs: ConfigCollection,
    total_samples: int,
    batch_size: int,
    logger: Logger,
    print_progress: bool,
) -> jnp.ndarray:
    """Evalautes the energy over `total_samples` of the provided configuration
    collection sequentially. The electrons are initially thermalized and then 
    subsequently only thermalized in smaller steps for each geometry.
    This function is useful if the number of configurations does not divide
    without residual by the number of GPUs.

    Args:
        key (jnp.ndarray): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jnp.ndarray: Final energies of form (n_configs, total_samples)
    """
    E_l_final = []
    E_final = []
    E_final_std = []
    E_final_err = []
    val_systems = configs.get_current_systems()
    if print_progress:
        iterator = tqdm.tqdm(
            enumerate(val_systems),
            total=len(val_systems),
            desc='Evaluation'
        )
    else:
        iterator = enumerate(val_systems)
    for i, sys in iterator:
        val_atoms = replicate(sys.coords()[None])

        if i == 0:
            key, subkey = jax.random.split(key)
            electrons = init_electrons(
                val_atoms,
                sys.charges(),
                sys.spins(),
                batch_size,
                subkey
            )
            electrons = broadcast(electrons)
            electrons = vmc.thermalize_samples(
                electrons,
                val_atoms,
                500,
                show_progress=print_progress
            )
        else:
            electrons = vmc.thermalize_samples(
                electrons,
                val_atoms,
                100,
                show_progress=print_progress
            )

        n_repeats = int(
            math.ceil(total_samples['validation']['total_samples'])/batch_size
        )
        E_l = vmc.eval_energy(
            electrons, val_atoms, n_repeats, 5, print_progress
        )
        E_l_final.append(E_l)
        E_final.append(np.mean(E_l))
        E_final_std.append(np.std(E_l))
        E_final_err.append(E_final_std[-1]/np.sqrt(E_l.size))

        if logger is not None:
            logger.scalar(f'PES/E', E_final[-1], step=i)
            logger.scalar(f'PES/E_std', E_final_std[-1], step=i)
            logger.scalar(f'PES/E_err', E_final_err[-1], step=i)
            logger.flush()
        if print_progress:
            iterator.set_postfix({
                'E': E_final[-1],
                'E_std': E_final_std[-1],
                'E_err': E_final_err[-1]
            })
    E_l_final = np.array(E_l_final)
    E_final = np.array(E_final)
    E_final_std = np.array(E_final_std)
    E_final_err = np.array(E_final_err)
    return E_l_final, E_final, E_final_std, E_final_err


def eval_energy(
    key: jnp.ndarray,
    vmc,
    configs: ConfigCollection,
    total_samples: int,
    batch_size: int,
    logger: Logger,
    print_progress: bool,
) -> jnp.ndarray:
    """Selects either `eval_energy_parallel` or `eval_energy_sequential`
    depending on whether the number of configurations divides without
    residual by the number of GPUs.

    Args:
        key (jnp.ndarray): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jnp.ndarray: Final energies of form (n_configs, total_samples)
    """
    if len(configs.get_current_systems()) % jax.device_count() == 0:
        return eval_energy_parallel(
            key,
            vmc,
            configs,
            total_samples,
            batch_size,
            logger,
            print_progress
        )
    else:
        return eval_energy_sequential(
            key,
            vmc,
            configs,
            total_samples,
            batch_size,
            logger,
            print_progress
        )


def to_numpy(pytree):
    return jax.tree_map(lambda x: np.asarray(x), pytree)


def to_jnp(pytree):
    return jax.tree_map(lambda x: jnp.array(x), pytree)
