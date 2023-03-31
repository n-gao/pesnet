import math

import jax
import numpy as np
from chex import PRNGKey
from seml_logger import Logger

from pesnet.systems.collection import ConfigCollection
from pesnet.utils.jax_utils import broadcast, replicate
from pesnet.vmc.update import init_electrons


def eval_energy_parallel(
    key: PRNGKey,
    vmc,
    configs: ConfigCollection,
    total_samples: int,
    batch_size: int,
    logger: Logger,
    print_progress: bool,
) -> jax.Array:
    """Evalautes the energy over `total_samples` of the provided configuration
    collection in parallel. This is generally the preferred method as it is
    significantly faster.

    Args:
        key (jax.Array): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jax.Array: Final energies of form (n_configs, total_samples)
    """
    val_atoms = broadcast(configs.get_current_atoms(jax.device_count()))
    sys = configs.get_current_systems()[0]
    key, subkey = jax.random.split(key)
    electrons = init_electrons(
        subkey,
        val_atoms[0],
        sys.charges,
        sys.spins,
        batch_size
    )
    electrons = broadcast(electrons.reshape(jax.device_count(), -1, *electrons.shape[1:]))
    electrons = vmc.thermalize_samples(
        electrons,
        val_atoms,
        10000,
        show_progress=print_progress,
        adapt_step_width=True
    )
    n_repeats = int(
        math.ceil(total_samples/electrons.shape[-2])
    )
    E_l = vmc.eval_energy(electrons, val_atoms, n_repeats, 10, print_progress)
    E_l = E_l.reshape(len(configs.get_current_systems()), -1)
    E_final = np.mean(E_l, axis=-1)
    E_final_std = np.std(E_l, axis=-1)
    E_final_err = E_final_std/np.sqrt(E_l.size/E_final.size)

    if logger is not None:
        for i, (E, std, err) in enumerate(zip(E_final, E_final_std, E_final_err)):
            logger.add_scalar(f'PES/E', E, step=i)
            logger.add_scalar(f'PES/E_std', std, step=i)
            logger.add_scalar(f'PES/E_err', err, step=i)
    return E_l, E_final, E_final_std, E_final_err


def eval_energy_sequential(
    key: PRNGKey,
    vmc,
    configs: ConfigCollection,
    batch_size: int,
    logger: Logger,
    total_samples: int = 1_000_000,
    init_therm_steps: int = 10_000,
    therm_steps: int = 500,
) -> jax.Array:
    """Evalautes the energy over `total_samples` of the provided configuration
    collection sequentially. The electrons are initially thermalized and then 
    subsequently only thermalized in smaller steps for each geometry.
    This function is useful if the number of configurations does not divide
    without residual by the number of GPUs.

    Args:
        key (jax.Array): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jax.Array: Final energies of form (n_configs, total_samples)
    """
    E_l_final = []
    E_final = []
    E_final_std = []
    E_final_err = []
    val_systems = configs.get_current_systems()
    
    for i, sys in logger.tqdm(enumerate(val_systems), total=len(val_systems), desc='Evaluation'):
        val_atoms = replicate(sys.coords[None])

        if i == 0:
            key, subkey = jax.random.split(key)
            electrons = init_electrons(
                subkey,
                val_atoms[0],
                sys.charges,
                sys.spins,
                batch_size
            )
            electrons = broadcast(electrons.reshape(jax.device_count(), -1, *electrons.shape[1:]))
            electrons = vmc.thermalize_samples(
                electrons,
                val_atoms,
                init_therm_steps,
                show_progress=logger.print_progress,
                adapt_step_width=True
            )
        else:
            electrons = vmc.coordinate_transform(old_atoms, val_atoms, electrons)
            electrons = vmc.thermalize_samples(
                electrons,
                val_atoms,
                therm_steps,
                show_progress=logger.print_progress,
                adapt_step_width=True
            )

        n_repeats = int(
            math.ceil(total_samples/batch_size)
        )
        E_l = vmc.eval_energy(
            electrons, val_atoms, n_repeats, 10, logger.print_progress
        )
        E_l_final.append(E_l)
        E_final.append(np.mean(E_l))
        E_final_std.append(np.std(E_l))
        E_final_err.append(E_final_std[-1]/np.sqrt(E_l.size))

        if logger is not None:
            logger.add_scalar(f'PES/E', E_final[-1], step=i)
            logger.add_scalar(f'PES/E_std', E_final_std[-1], step=i)
            logger.add_scalar(f'PES/E_err', E_final_err[-1], step=i)
        logger.set_postfix({
            'E': E_final[-1],
            'E_std': E_final_std[-1],
            'E_err': E_final_err[-1]
        })
        old_atoms = val_atoms
    E_l_final = np.array(E_l_final)
    E_final = np.array(E_final)
    E_final_std = np.array(E_final_std)
    E_final_err = np.array(E_final_err)
    return E_l_final, E_final, E_final_std, E_final_err


def eval_energy(
    key: PRNGKey,
    vmc,
    configs: ConfigCollection,
    total_samples: int,
    batch_size: int,
    logger: Logger,
    print_progress: bool,
) -> jax.Array:
    """Selects either `eval_energy_parallel` or `eval_energy_sequential`
    depending on whether the number of configurations divides without
    residual by the number of GPUs.

    Args:
        key (jax.Array): jax.random.PRNGKey
        vmc (PesVmc): VMC model
        configs (ConfigCollection): Configurations to evaluate
        total_samples (int): Total numbers of samples per configuration
        batch_size (int): Batch size
        logger (Logger): Logger for logging to tensorboard
        print_progress (bool): Whether to print progress

    Returns:
        jax.Array: Final energies of form (n_configs, total_samples)
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
