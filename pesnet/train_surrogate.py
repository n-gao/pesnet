import logging
import os

os.environ['JAX_DEFAULT_DTYPE_BITS'] = '32'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import json
import pickle
from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
import optax
import seml
import tqdm.auto as tqdm
from sacred import Experiment
from seml_logger import Logger, add_default_observer_config, automain

import pesnet.systems
from pesnet.nn.dimenet import DimeNet
from pesnet.surrogate.training import (make_many_surrogate_steps,
                                       make_surrogate_training_step)
from pesnet.systems.collection import make_system_collection
from pesnet.trainer import VmcTrainer
from pesnet.utils import (p_ema_make, ema_make, ema_value, jax_utils,
                          make_schedule, to_numpy)
from pesnet.utils.jax_utils import pmap, instance
from pesnet.vmc.eval import *

ex = Experiment()
seml.setup_logger(ex)
add_default_observer_config(ex, notify_on_completed=False)


def get_chkp_and_data(path):
    chkpoints = [os.path.join(path, p) for p in os.listdir(path) if '.checkpoint' in p and '0' in p]
    order = np.argsort([int(Path(p).stem) for p in chkpoints])
    chkpoints = np.array(chkpoints)[order]

    data = [os.path.join(path, p) for p in os.listdir(path) if 'data' in p and '0' in p]
    order = np.argsort([int(Path(p).stem.split('_')[1]) for p in data])
    data = np.array(data)[order]
        
    return chkpoints, data, np.sort([int(Path(p).stem.split('_')[1]) for p in data])


def get_final_chkpt(path):
    if os.path.exists(os.path.join(path, 'best.checkpoint')):
        return os.path.join(path, 'best.checkpoint')
    else:
        return get_chkp_and_data(path)[0][-1]


def get_config(path):
    with open(os.path.join(path, 'config.json')) as inp:
        return json.load(inp)


def get_results(path):
    if os.path.exists(os.path.join(path, 'result_sequential.pickle')):
        with open(os.path.join(path, 'result_sequential.pickle'), 'rb') as inp:
            return pickle.load(inp)
    elif os.path.exists(os.path.join(path, 'result.pickle')):
        with open(os.path.join(path, 'result.pickle'), 'rb') as inp:
            return pickle.load(inp)
    else:
        return None


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def eval_energies(key, path, collection, logger):
    systems = collection.get_current_systems()
    charges = systems[0].charges()
    spins = systems[0].spins()
    config = get_config(path)
    key, subkey = jax.random.split(key)
    vmc = VmcTrainer(
        subkey,
        charges,
        spins,
        config['pesnet'],
        config['sampling'],
        config['optimization'],
        config['surrogate'],
        config['surrogate_optimization']
    )
    vmc.load_checkpoint(get_final_chkpt(path))
    key, subkey = jax.random.split(key)
    E_l_final, E_final, E_final_std, E_final_err = eval_energy_sequential(
        subkey,
        vmc,
        collection,
        config['system']['validation']['total_samples'] if 'total_size' in config['system']['validation'] else 1_000_000,
        config['training']['val_batch_size'],
        logger,
        True)
    with open(os.path.join(path, 'result.pickle'), 'wb') as out:
        pickle.dump({
            'E_final': E_final.tolist(),
            'E_final_std': E_final_std.tolist(),
            'E_final_err': E_final_err.tolist()
        }, out)
    return E_final


@ex.config
def config():
    lr = dict(
        init=1e-4,
        delay=10000,
        decay=1.0
    )
    n_steps = 5
    ema_min = 0.99
    ema_max = 0.9999
    loss_decay = 0.999
    threshold = 1.05
    loss = 'rmse'
    dimenet = dict(
        emb_size=128,
        out_emb_size=256,
        int_emb_size= 64,
        basis_emb_size=8,
        num_blocks=4,
        
        directional=False,
        concat_before_out=False,

        num_spherical=7,
        num_radial=6,
        num_rbf=6,
        cutoff=10,
        envelope_exponent=1,
        envelope_type='none',

        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,

        activation='silu',
    )

def naming_fn(path, **_):
    path = os.path.expanduser(path)
    config = get_config(path)

    collection = make_system_collection(
        getattr(pesnet.systems, config['system']['name']),
        **config['system']['validation'],
        deterministic=True
    )
    val_systems = collection.get_current_systems()
    return str(val_systems[0])


@automain(ex, naming_fn, default_folder='~/tensorboard/planet_sep')
def main(
        path,
        dimenet,
        lr,
        ema_min,
        ema_max,
        loss_decay,
        threshold,
        loss,
        n_steps,
        seed=None,
        logger: Logger = None):
    path = os.path.expanduser(path)
    key = jax.random.PRNGKey(seed)
    config = get_config(path)

    collection = make_system_collection(
        getattr(pesnet.systems, config['system']['name']),
        **config['system']['validation'],
        deterministic=True
    )
    systems = collection.get_current_systems()
    charges = systems[0].charges()
    val_atoms = collection.get_current_atoms(jax.device_count())

    try:
        h5 = h5py.File(os.path.join(path, 'energies.h5py'), 'r')
    except Exception:
        h5 = h5py.File(os.path.join(path, 'data.h5py'), 'r')
    
    ds_atoms = h5['atoms'][:]
    ds_energies = h5['energies'][:]
    ds_atoms = ds_atoms.reshape(ds_atoms.shape[0], jax.device_count(), -1, *ds_atoms.shape[3:])
    ds_energies = ds_energies.reshape(ds_energies.shape[0], jax.device_count(), -1, *ds_energies.shape[3:])
    offset = ds_energies[0].mean()

    surrogate = DimeNet([], [1], charges, **dimenet)
    proper_fwd = lambda params, atoms: surrogate.apply(params, atoms)[1][0].squeeze()
    potential_fn = pmap(jax.vmap(proper_fwd, in_axes=(None, 0)))
    
    key, subkey = jax.random.split(key)
    params = jax_utils.replicate(surrogate.init(subkey, val_atoms[0, 0]).unfreeze())

    optimizer = optax.adamw(make_schedule(lr))
    opt_state = pmap(optimizer.init)(params)

    train_step = make_surrogate_training_step(
        energy_net=jax.vmap(proper_fwd, in_axes=(None, 0)),
        opt_update=optimizer.update,
        loss=loss,
        n_steps=n_steps,
        ema_decay_min=ema_min,
        ema_decay_max=ema_max,
        threshold=threshold,
        loss_decay=loss_decay
    )
    train_step = pmap(make_many_surrogate_steps(train_step))

    emas = {
        'loss': jax_utils.replicate(ema_make(-1 * jnp.ones(()))),
        'err': jax_utils.replicate(ema_make(-1 * jnp.ones(()))),
        'params': p_ema_make(params)
    }

    try:
        vmc_energies = np.array(get_results(path)['E_final'])
    except Exception:
        logging.info('final energies are missing. Recomputing them...')
        key, subkey = jax.random.split(key)
        vmc_energies = eval_energies(subkey, path, collection, logger)

    step = 0
    ones = replicate(jnp.ones(()))

    batch_size = 10
    with tqdm.trange(len(ds_energies)//batch_size) as ds_iter:
        for idx in ds_iter:
            index = slice(idx*batch_size, (idx+1)*batch_size)
            (params, opt_state), emas, aux_data = train_step(
                params,
                ds_atoms[index].swapaxes(0, 1),
                ds_energies[index].swapaxes(0, 1),
                opt_state,
                emas,
                ones*offset
            )
            keys = aux_data.keys()
            for vals in zip(*[x[0] for x in aux_data.values()]):
                for k, v in zip(keys, vals):
                    logger.add_scalar(f'train/{k}', v, global_step=step)
                step += 1
            ds_iter.set_postfix({'loss': aux_data['loss'].reshape(-1)[-1]})
    
    surr_energies = np.array(potential_fn(ema_value(emas['params']), val_atoms).reshape(-1)) + offset
    mae = np.abs(vmc_energies - surr_energies).mean()
    mse = np.mean((vmc_energies - surr_energies)**2)
    rmse = mse ** 0.5

    logger.add_scalar('eval/MAE', mae)
    logger.add_scalar('eval/MSE', mse)
    logger.add_scalar('eval/RMSE', rmse)
    plt.plot(vmc_energies)
    plt.plot(surr_energies)
    logger.add_figure('eval/PES', plt.gcf())
    logger.add_scalar('eval/E', surr_energies.mean())

    result_dict = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'vmc_energies': vmc_energies.tolist(),
        'surr_energies': surr_energies.tolist()
    }
    logger.store_dict('params.npz', params=to_numpy(instance(params)), ema_params=to_numpy(instance(ema_value(emas['params']))), offset=offset)
    return result_dict
