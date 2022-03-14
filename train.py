import functools
import logging
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
import traceback

import jax
import matplotlib.pyplot as plt
import numpy as np
import seml
import tqdm.auto as tqdm
from sacred import Experiment

from pesnet import systems
from pesnet.jax_utils import broadcast, replicate
from pesnet.pretrain import determine_hf_signs
from pesnet.systems.collection import StaticConfigs, make_system_collection
from pesnet.systems.scf import Scf
from pesnet.training import init_electrons
from pesnet.utils import (ExponentiallyMovingAverage, Logger, Stopwatch,
                          eval_energy)
from pesnet.vmc import PesVmc

ex = Experiment()
seml.setup_logger(ex)

ex.add_config('configs/default_config.yaml')
ex.add_config('configs/pesnet.yaml')


@ex.config
def config():
    overwrite = None
    db_collection = None
    print_progress = False

    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(
            db_collection, overwrite=overwrite))


@ex.automain
def run(
    system: dict,
    pesnet: dict,
    sampling: dict,
    optimization: dict,
    training: dict,
    pretraining: dict,
    print_progress: bool,
    log_every: int,
    db_collection: str = None,
    comparison_idx: int = None,
    seed=None,
):
    config = locals()
    logger = None
    try:
        trange = functools.partial(tqdm.trange, disable=not print_progress)

        key = jax.random.PRNGKey(seed)
        n_devices = jax.device_count()
        system_configs = make_system_collection(
            getattr(systems, system['name']),
            **system['training']
        )
        current_systems = system_configs.get_current_systems()
        charges = current_systems[0].charges()
        spins = current_systems[0].spins()

        key, subkey = jax.random.split(key)
        logging.info('Initialization')
        vmc = PesVmc(
            subkey,
            charges,
            spins,
            pesnet,
            sampling,
            optimization,
        )

        logger = Logger(f'{current_systems[0]}',
                        config, subfolder=db_collection)

        # Pretraining
        if pretraining['steps'] > 0:
            if pretraining['single']:
                if isinstance(system_configs, StaticConfigs):
                    pretrain_system = current_systems[0]
                else:
                    pretrain_system = current_systems[len(current_systems)//2]
                pretrain_atoms = replicate(pretrain_system.coords()[None])
                scfs = [Scf(pretrain_system.to_pyscf(verbose=3))]
            else:
                pretrain_atoms = system_configs.get_current_atoms(n_devices)
                scfs = [
                    Scf(s.to_pyscf(verbose=3),
                        restricted=pretraining['restricted'])
                    for s in current_systems
                ]
            scfs[0].run()
            for scf, last in zip(scfs[1:], scfs[:-1]):
                scf.run(last if pretraining['start_from_last'] else None)

            key, subkey = jax.random.split(key)
            electrons = init_electrons(pretrain_atoms, charges,
                                       spins, training['batch_size'], subkey)
            electrons = broadcast(electrons)

            signs = determine_hf_signs(scfs, electrons, spins)

            logging.info('Pretraining')
            with trange(pretraining['steps']) as iters:
                vmc.initialize_pretrain_optimizer(
                    train_gnn=pretraining['train_gnn'])
                for step in iters:
                    loss, electrons, pmove = vmc.pre_update_step(
                        electrons, pretrain_atoms, scfs, signs)
                    logger.scalar('pretrain/mse', np.mean(loss), step=step)
                    logger.scalar('pretrain/pmove', np.mean(pmove), step=step)
                    iters.set_postfix({
                        'MSE': np.mean(loss),
                        'pmove': np.mean(pmove)
                    })

        # Thermalize electrons
        atoms = broadcast(system_configs.get_current_atoms(n_devices))
        # Only resample electrons if we have to
        if pretraining['steps'] <= 0 or electrons.shape[:2] != atoms.shape[:2]:
            key, subkey = jax.random.split(key)
            electrons = init_electrons(atoms, charges,
                                       spins, training['batch_size'], subkey)
            electrons = broadcast(electrons)
        logging.info('Thermalizing')
        electrons = vmc.thermalize_samples(
            electrons,
            atoms,
            training['thermalizing_steps'],
            show_progress=print_progress
        )

        vmc.initialize_optimizer()
        energies = []
        energy_variances = []
        pmoves = []

        # Early stopping
        ema = ExponentiallyMovingAverage()
        lowest_std = None
        lowest_step = 0
        best_params = vmc.params
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

        with trange(training['max_steps']) as iters:
            for step in iters:
                # Update configs
                system_configs.update_configs()
                atoms = broadcast(system_configs.get_current_atoms(n_devices))

                # Do update step
                electrons, (E_by_config, E_var_by_config,
                            pmove) = vmc.update_step(electrons, atoms)
                E_by_config = E_by_config.reshape(-1)
                E_var_by_config = E_var_by_config.reshape(-1)

                # Compute metrics
                E = np.mean(E_by_config)
                E_var = np.mean(E_var_by_config)
                E_std = np.mean(np.sqrt(E_var_by_config))
                pmove = np.mean(pmove)
                energies.append(E)
                energy_variances.append(E_var)
                pmoves.append(pmove)
                ema.update(E_std, decay)

                # NaN check
                if np.isnan(E).any():
                    raise ValueError("Detected NaN during training.")

                # Log histograms
                histograms = system_configs.get_current_histograms(20)
                for name, stats in histograms.items():
                    logger.histogram(f'train/{name}', **stats, step=step)

                # Log everything
                if step % log_every == 0:
                    logger.scalar('train/E', E, step=step)
                    logger.scalar('train/E_std', E_std, step=step)
                    logger.scalar('train/E_var', E_var, step=step)
                    logger.scalar('train/pmove', pmove, step=step)
                    logger.scalar('train/t_per_step',
                                  stopwatch()/log_every, step=step)
                    if 'damping' in vmc.train_state.value:
                        logger.scalar('train/damping',
                                      vmc.train_state.value['damping'][0], step=step)
                    if sub_configs is not None:
                        for i in range(len(sub_configs)):
                            logger.scalar(
                                f'train_sub/{sub_configs[i]}/E', E_by_config[i], step=step)

                    if comparison_idx is not None:
                        E_comp = E_by_config[comparison_idx]
                        for i, val in enumerate(E_by_config):
                            if i == comparison_idx:
                                continue
                            if sub_configs is None:
                                logger.scalar(
                                    f'train_comp/{i}-{comparison_idx}/E', val - E_comp, step=step)
                            else:
                                logger.scalar(
                                    f'train_comp/{sub_configs[i]}-{sub_configs[comparison_idx]}/E', val - E_comp, step=step)

                # Log parameters
                if step % 100 == 0:
                    logger.log_electrons(
                        electrons, 'electrons', n_bins=100, step=step)
                    logger.log_params(vmc.params, 'params',
                                      n_bins=100, step=step)
                    logger.log_params(vmc.get_fermi_params(
                        atoms), 'fermi_params', n_bins=100, step=step)
                iters.set_postfix({
                    'E': E,
                    'E_std': E_std,
                    'E_var': E_var,
                    'pmove': pmove
                })
                if step % training['checkpoint_every'] == 0:
                    logging.info(f'[{step}] creating checkpoint')
                    vmc.checkpoint(logger.folder_name, f'checkpoint_{step}')
                    np.savez(os.path.join(logger.folder_name,
                             f'data_{step}'), electrons=electrons, atoms=atoms)
                if lowest_std is None or (ema.value < lowest_std and abs(lowest_std - ema.value)/lowest_std > eps):
                    lowest_std = ema.value
                    lowest_step = step
                    best_params = vmc.params
                if step - lowest_step > patience:
                    logging.info('Stopping training due to convergence.')
                    break

        vmc.checkpoint(logger.folder_name, 'checkpoint_last')
        np.savez(os.path.join(logger.folder_name, 'data_last'),
                 electrons=electrons, atoms=atoms)

        vmc.params = best_params
        vmc.checkpoint(logger.folder_name, 'checkpoint_final')

        energies = np.asarray(energies).reshape(-1)
        energy_variances = np.asarray(energy_variances).reshape(-1)
        pmoves = np.asarray(pmoves).reshape(-1)

        logging.info('Evaluating final energy')
        val_configs = make_system_collection(
            getattr(systems, system['name']),
            **system['validation']
        )

        key, subkey = jax.random.split(key)
        E_l_final, E_final, E_final_std, E_final_err = eval_energy(
            subkey,
            vmc,
            val_configs,
            system['validation']['total_samples'],
            training['val_batch_size'],
            logger,
            print_progress
        )

        logging.info('Plotting')
        plt.errorbar(np.arange(len(E_final)), E_final, yerr=E_final_err)
        logger.plot('PES', plt)

        np.save(logger.folder_name + '/e_l_final.npy', np.array(E_l_final))

        if energies.size > 10000:
            idx = np.linspace(0, energies.size-1, 10000).astype(np.int32)
            energies = energies[idx]
            energy_variances = energy_variances[idx]
            pmoves = pmoves[idx]

        return {
            'E': energies,
            'E_var': energy_variances,
            'pmove': pmoves,
            'E_final': E_final.tolist(),
            'E_final_std': E_final_std.tolist(),
            'E_final_err': E_final_err.tolist(),
            'tensorboard': logger.folder_name
        }
    except:
        if logger is not None:
            logger.close()
        traceback.print_exc()
        exit()
