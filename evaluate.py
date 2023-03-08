import json
import os

import jax
import matplotlib.pyplot as plt
import seml
from sacred import Experiment
from seml_logger import add_default_observer_config, automain

import pesnet.systems
from pesnet.systems.collection import make_system_collection
from pesnet.trainer import VmcTrainer
from pesnet.vmc.eval import *

jax.config.update('jax_default_matmul_precision', 'float32')
jax.config.update('jax_array', False)


ex = Experiment()
seml.setup_logger(ex)
add_default_observer_config(ex)


def get_final_chkpt(path):
    return os.path.join(path, 'best.checkpoint')


def get_config(path):
    with open(os.path.join(path, 'config.json')) as inp:
        return json.load(inp)


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
        config['training']['val_batch_size'],
        logger,
        1_000_000)
    return E_final, E_final_std, E_final_err


def naming_fn(path, name='', **_):
    path = os.path.expanduser(path)
    config = get_config(path)
    collection = make_system_collection(
        getattr(pesnet.systems, config['system']['name']),
        **config['system']['training'],
        deterministic=True
    )
    val_systems = collection.get_current_systems()
    return name + str(val_systems[0])


@automain(ex, naming_fn, default_folder='~/tensorboard/planet_eval')
def main(
        path,
        name: str,
        system: dict = None,
        seed=None,
        logger=None):
    path = os.path.expanduser(path)
    key = jax.random.PRNGKey(seed)
    config = get_config(path)

    if system is None:
        collection = make_system_collection(
            getattr(pesnet.systems, config['system']['name']),
            **config['system']['validation'],
            deterministic=True
        )
    else:
        collection = make_system_collection(
            getattr(pesnet.systems, system['name']),
            **system['validation'],
            deterministic=True
        )

    key, subkey = jax.random.split(key)
    E_final, E_final_std, E_final_err = eval_energies(subkey, path, collection, logger)
    result = {
        'E_final': E_final,
        'E_final_std': E_final_std,
        'E_final_err': E_final_err
    }
    plt.errorbar(np.arange(len(E_final)), E_final, yerr=E_final_err, label='MC')
    plt.legend()
    logger.add_figure('PES', plt.gcf())
    return result
