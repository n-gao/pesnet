import functools
import logging
import os
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seml
import tqdm.auto as tqdm
from sacred import Experiment
from seml_logger import Logger, add_default_observer_config, automain

from pesnet import nn, systems
from pesnet.systems.collection import (JointCollection, StaticConfigs,
                                       make_system_collection)
from pesnet.systems.constants import HARTREE_TO_KCAL
from pesnet.systems.scf import align_scfs
from pesnet.trainer import VmcTrainer
from pesnet.utils import ExponentiallyMovingAverage, Stopwatch
from pesnet.utils.jax_utils import broadcast, replicate
from pesnet.vmc.eval import eval_energy_sequential
from pesnet.vmc.update import init_electrons

jax.config.update('jax_default_matmul_precision', 'float32')

ex = Experiment()
seml.setup_logger(ex)

ex.add_config('configs/default_config.yaml')
ex.add_config('configs/pesnet.yaml')
add_default_observer_config(ex)


def pretrain(
    key,
    logger: Logger,
    vmc: VmcTrainer,
    system_configs,
    trange,
    batch_size: int,
    steps: int,
    single: bool,
    start_from_last: bool,
    method: Tuple[str],
    checkpoint_dir: str = None,
    align_mo: bool = False,
    distinct_orbitals: bool = True
):
    if steps < 1:
        return None, None
    # Select pretraining system
    current_systems = system_configs.get_current_systems()
    charges = current_systems[0].charges
    spins = current_systems[0].spins
    if single:
        pretrain_system = current_systems[len(current_systems)//2]
        pretrain_atoms = replicate(pretrain_system.coords[None])
        scfs = [pretrain_system.to_scf()]
    else:
        pretrain_atoms = system_configs.get_current_atoms()
        # If we don't train all determinants to all HF solutions, we can save a lot of work by
        # dropping the unused configurations
        if distinct_orbitals:
            k = vmc.pesnet_fns.ferminet.determinants
            if len(current_systems) > k:
                idx = np.linspace(0, len(current_systems) - 1, k).astype(int)
                current_systems = [current_systems[i] for i in idx]
                pretrain_atoms = pretrain_atoms[idx]
        pretrain_atoms = replicate(pretrain_atoms.repeat(len(method), axis=0))
        scfs = [
            s.to_scf(restricted=m=='rhf')
            for s in current_systems
            for m in method
        ]
    
    # Determine SCF file location
    if checkpoint_dir is not None:
        chk_path = os.path.join(checkpoint_dir, '{}.chk')
    else:
        chk_path = os.path.join(logger.log_dir, '{}.chk')
    # Perform SCF calculations
    scfs[0].run(checkfile=chk_path.format(0))
    for i, (scf, last) in enumerate(zip(scfs[1:], scfs[:-1])):
        scf.run(last if start_from_last else None, checkfile=chk_path.format(i+1))
    if align_mo:
        align_scfs(scfs)

    # Init electrons
    key, subkey = jax.random.split(key)
    electrons = init_electrons(subkey, pretrain_atoms[0], charges, spins, batch_size)
    electrons = broadcast(electrons.reshape(jax.device_count(), -1, *electrons.shape[1:]))
    
    logging.info('Pretraining')
    with trange(steps) as iters:
        for step in iters:
            loss, electrons, pmove = vmc.pre_update_step(
                electrons, pretrain_atoms, scfs)
            if step % 10 == 0:
                logger.add_scalar('pretrain/mse', np.mean(loss).item(), step=step)
                logger.add_scalar('pretrain/pmove', np.mean(pmove).item(), step=step)
            iters.set_postfix({
                'MSE': np.mean(loss),
                'pmove': np.mean(pmove)
            })
            if step % 200 == 0:
                with logger.without_aim():
                    if 'gnn' in vmc.params:
                        logger.add_distribution_dict(vmc.params['gnn'], 'pre_gnn', n_bins=100, step=step)
                    fermi_params = vmc.get_fermi_params(pretrain_atoms)
                    logger.add_distribution_dict(fermi_params, 'pre_fermi', n_bins=100, step=step)
    return electrons, [s.energy for s in scfs]


def name_fn(system):
    val_configs = make_system_collection(
        getattr(systems, system['name']),
        **system['validation'],
        deterministic=True
    )
    val_systems = val_configs.get_current_systems()
    return str(val_systems[0])


@automain(ex, name_fn, default_folder='~/logs/pesnet')
def run(
    system: dict,
    pesnet: dict,
    sampling: dict,
    optimization: dict,
    training: dict,
    pretraining: dict,
    surrogate: dict,
    surrogate_optimization: dict,
    evaluation: dict,
    log_every: int,
    comparison_idx: int = None,
    init_method: str = 'default',
    checkpoint: str = None,
    seed=None,
    logger: Logger = None,
    transform_coordinates: bool = False,
    log_energies: bool = False
):
    print_progress = logger.print_progress
    trange = functools.partial(tqdm.trange, disable=not print_progress)

    nn.set_init_method(init_method)
    key = jax.random.PRNGKey(seed)
    n_devices = jax.device_count()
    system_configs = make_system_collection(
        getattr(systems, system['name']),
        **system['training']
    )
    val_configs = make_system_collection(
        getattr(systems, system['name']),
        **system['validation'],
        deterministic=True
    )
    val_systems = val_configs.get_current_systems()
    val_atoms = val_configs.get_current_atoms(n_devices)
    current_systems = system_configs.get_current_systems()
    charges = current_systems[0].charges
    spins = current_systems[0].spins
    
    key, subkey = jax.random.split(key)
    logging.info('Initialization')
    vmc = VmcTrainer(
        subkey,
        charges,
        spins,
        pesnet,
        sampling,
        optimization,
        surrogate,
        surrogate_optimization,
        pretraining
    )

    # Pretraining
    if checkpoint is not None:
        vmc.load_checkpoint(checkpoint)
        electrons = None
        hf_energies = None
    else:
        key, subkey = jax.random.split(key)
        electrons, hf_energies = pretrain(
            subkey,
            logger,
            vmc,
            system_configs,
            trange,
            training['batch_size'],
            **pretraining
        )
        vmc.checkpoint(logger.log_dir, f'pretrained.checkpoint')

    # Thermalize electrons
    atoms = broadcast(system_configs.get_current_atoms(n_devices))
    if electrons is None or atoms.shape[:2] != electrons.shape[:2]:  # Only resample electrons if we have to
        key, subkey = jax.random.split(key)
        electrons = init_electrons(subkey, atoms[0], charges,
                                    spins, training['batch_size'])
        electrons = broadcast(electrons.reshape(jax.device_count(), -1, *electrons.shape[1:]))
    logging.info('Thermalizing')
    electrons = vmc.thermalize_samples(
        electrons,
        atoms,
        training['thermalizing_steps'],
        show_progress=print_progress,
        adapt_step_width=True
    )

    if hf_energies is not None:
        vmc.offset += jnp.mean(jnp.array(hf_energies))
    energies = []
    energy_variances = []
    pmoves = []

    # Early stopping
    ema = ExponentiallyMovingAverage()
    lowest_std = None
    lowest_step = 0
    patience = training['patience']
    eps = training['eps']
    decay = training['ema']

    # Time measurement
    stopwatch = Stopwatch()

    logging.info('Training')
    sub_configs = None
    if isinstance(system_configs, StaticConfigs):
        if system_configs.n_configs <= 8 or comparison_idx is not None:
            sub_configs = [f'{k}_{s}' for k, l in system['training']
                            ['config'].items() if isinstance(l, list) for s in l]
    
    if log_energies:
        a_dset = logger.create_dataset(
            'atoms',
            (training['max_steps'], *atoms.shape),
            atoms.dtype,
            chunks=(10, *atoms.shape)
        )
        e_dset = logger.create_dataset(
            'energies',
            (training['max_steps'], *electrons.shape[:-1]),
            electrons.dtype,
            chunks=(10, *electrons.shape[:-1])
        )

    with trange(training['max_steps']) as iters:
        for step in iters:
            # Update configs
            system_configs.update_configs()
            new_atoms = broadcast(system_configs.get_current_atoms(n_devices))
            if transform_coordinates:
                electrons = vmc.coordinate_transform(atoms, new_atoms, electrons)
            atoms = new_atoms

            # Do update step
            result, aux_data = vmc.update_step(electrons, atoms)
            electrons = result.electrons
            E_by_config = np.array(result.energies).reshape(-1)
            E_var_by_config = np.array(result.energy_variances).reshape(-1)

            if log_energies:
                a_dset[step] = np.asarray(atoms)
                e_dset[step] = np.asarray(result.local_energies)

            # Compute metrics
            E = np.mean(E_by_config)
            E_var = np.mean(E_var_by_config)
            E_std = np.mean(np.sqrt(E_var_by_config))
            pmove = np.mean(result.pmove)
            energies.append(E)
            energy_variances.append(E_var)
            pmoves.append(pmove)
            ema.update(E_std, decay)

            # NaN check
            if np.isnan(E).any():
                raise ValueError(f"Detected NaN during training in step {step}!")

            # Log everything
            if step % log_every == 0:
                logger.add_scalar_dict({
                    'E': E,
                    'E_std': E_std,
                    'E_var': E_var,
                    'pmove': pmove,
                    't_per_step': stopwatch()/log_every
                }, 'train', step=step)

                # Config histograms
                with logger.without_aim():
                    logger.add_distribution_dict(system_configs.get_current_conf_vals(), 'config', n_bins=20, step=step)
                
                if sub_configs is not None:
                    for i in range(len(sub_configs)):
                        logger.add_scalar(
                            f'train_sub/{sub_configs[i]}/E', E_by_config[i], step=step)
                        logger.add_scalar(
                            f'train_sub/{sub_configs[i]}/E_std', E_var_by_config[i]**0.5, step=step)
                
                if comparison_idx is not None:
                    E_comp = E_by_config[comparison_idx]
                    for i, val in enumerate(E_by_config):
                        if i == comparison_idx:
                            continue
                        if sub_configs is None:
                            logger.add_scalar(f'train_comp/{i}-{comparison_idx}/E', val - E_comp, step=step)
                        else:
                            logger.add_scalar(f'train_comp/{sub_configs[i]}-{sub_configs[comparison_idx]}/E', (val - E_comp) * HARTREE_TO_KCAL, step=step)
                
                # Log aux data
                logger.add_scalar_dict(aux_data, step=step)

            # Log parameters
            if step % (log_every*10) == 0:
                E_gnn = vmc.surrogate_energies(val_atoms)[0]
                logger.add_scalar('train/val_E', E_gnn.mean(), step=step)

                if step < 1000 or step % 1000 == 0:
                    # plot val energies
                    if isinstance(val_systems, JointCollection):
                        splits = np.cumsum(val_systems.sub_system_counts)[:-1]
                        for i, x in enumerate(np.split(E_gnn, splits)):
                            plt.plot(np.arange(len(x)), x)
                            logger.add_figure('val/PES/{i}', plt.gcf(), step=step)
                    else:
                        plt.plot(np.arange(len(E_gnn)), E_gnn)
                        logger.add_figure('val/PES', plt.gcf(), step=step)
                    # log distributions
                    with logger.without_aim():
                        logger.add_distribution_dict(
                            {
                                'electrons': {
                                    'x': electrons[..., 0],
                                    'y': electrons[..., 1],
                                    'z': electrons[..., 2],
                                },
                                'gnn_params': vmc.params['gnn'],
                                'fermi_params': vmc.get_fermi_params(val_atoms),
                                'surr_params': vmc.surr_state.params,
                            },
                            n_bins=100,
                            step=step
                        )
            
            iters.set_postfix({
                'E': E,
                'E_std': E_std,
                'E_var': E_var,
                'pmove': pmove
            })
            if step % training['checkpoint_every'] == 0:
                logging.info(f'[{step}] creating checkpoint')
                vmc.checkpoint(logger.log_dir, f'{step}.checkpoint')
                logger.store_dict(f'data_{step}', electrons=electrons, atoms=atoms)
            if step % 100 == 0:
                if lowest_std is None or ema.value <= lowest_std:
                    vmc.checkpoint(logger.log_dir, 'best.checkpoint')
                if lowest_std is None or (ema.value < lowest_std and abs(lowest_std - ema.value)/lowest_std > eps):
                    lowest_std = ema.value
                    lowest_step = step
            if step - lowest_step > patience:
                logging.info('Stopping training due to convergence.')
                break

    if logger._h5py is not None:
        logger._h5py.close()

    vmc.checkpoint(logger.log_dir, 'last.checkpoint')
    logger.store_dict(f'data_last', electrons=electrons, atoms=atoms)

    vmc.load_checkpoint(os.path.join(logger.log_dir, 'best.checkpoint'))

    energies = np.asarray(energies).reshape(-1)
    energy_variances = np.asarray(energy_variances).reshape(-1)
    pmoves = np.asarray(pmoves).reshape(-1)

    logging.info('Evaluating final energy')
    key, subkey = jax.random.split(key)
    E_l_final, E_final, E_final_std, E_final_err = eval_energy_sequential(
        subkey,
        vmc,
        val_configs,
        training['val_batch_size'],
        logger,
        **evaluation
    )

    logging.info('Plotting')
    E_gnn = vmc.surrogate_energies(val_atoms)[0]
    plt.errorbar(np.arange(len(E_final)), E_final, yerr=E_final_err, label='MC')
    plt.plot(np.arange(len(E_gnn)), E_gnn, label='GNN')
    plt.legend()
    logger.add_figure('PES', plt.gcf())
    
    gnn_mae = np.abs(E_final - E_gnn).mean()
    logger.add_scalar('PES/MAE', gnn_mae)
    for i, val in enumerate(E_gnn):
        logger.add_scalar('PES/GNN', val, step=i)

    logger.store_array('e_l_final.npy', E_l_final)
    
    if energies.size > 10000:
        idx = np.linspace(0, energies.size-1, 10000).astype(np.int32)
        energies = energies[idx]
        energy_variances = energy_variances[idx]
        pmoves = pmoves[idx]

    result_dict = {
        'E_final': E_final.tolist(),
        'E_final_std': E_final_std.tolist(),
        'E_final_err': E_final_err.tolist(),
        'E_gnn': E_gnn.tolist(),
        'GNN_MAE': gnn_mae.item()
    }
    logging.info('Finished')
    return result_dict
