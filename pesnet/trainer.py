import functools
import gzip
import os
import pickle
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from seml.utils import flatten as flat_dict, unflatten as unflat_dict
import tqdm.auto as tqdm

from pesnet.nn import ParamTree
from pesnet.nn.dimenet import DimeNet
from pesnet.nn.pesnet import make_pesnet
from pesnet.surrogate.training import (make_joint_training_step,
                                       make_surrogate_training_step)
from pesnet.systems.scf import Scf
from pesnet.utils import (MCMCStepSizeScheduler, OnlineMean,
                          ema_make, p_ema_make, p_ema_value, jax_utils, make_schedule,
                          to_jnp, to_numpy)
from pesnet.utils.jax_utils import pmap, replicate, instance
from pesnet.vmc.damping import make_damping_fn
from pesnet.vmc.hamiltonian import make_local_energy_function
from pesnet.vmc.mcmc import make_mcmc
from pesnet.vmc.pretrain import eval_orbitals, make_pretrain_step
from pesnet.vmc.training import (make_loss, make_loss_and_natural_gradient_fn,
                                 make_training_step)


class VmcTrainer:
    """This class contains a lot of utility functions to ease
    the training of models on the potential energy surface.
    It takes care of initializing the network, sampler,
    energy computation and optimization. Further, it includes
    utility functions to thermalize samples, evaluate the
    energies and checkpointing.
    """

    def __init__(
        self,
        key: jax.Array,
        charges: Tuple[int, int],
        spins: Tuple[int, int],
        pesnet_config: Dict,
        sampler_config: Dict,
        optimization_config: Dict,
        surrogate_config: Dict,
        surrogate_optimization: Dict,
    ) -> None:
        """Constructor.

        Args:
            key (jax.Array): jax.random.PRNGKey
            charges (jax.Array): (M) charges of the system.
            spins (Tuple[int, int]): spins of the system.
            pesnet_config (Dict): Wfnet parameters, see 'pesnet.py'.
            sampler_config (Dict): Sampler parameters, see 'mcmc.py'.
            optimization_config (Dict): Optimization configuration.
            surrogate_config (Dict): Surrogate model hyperparameters.
            surrogate_optimization (Dict): Surrogate training hyperparameters.
        """

        self.key = key

        self.num_devices = jax.device_count()

        self.charges = charges
        self.spins = spins

        self.pesnet_config = pesnet_config
        self.sampler_config = sampler_config
        self.optimization_config = optimization_config
        self.surrogate_config = surrogate_config
        self.surrogate_optimization = surrogate_optimization

        self.key, subkey = jax.random.split(key)
        self.params, self.pesnet_fns = make_pesnet(
            subkey,
            charges=charges,
            spins=spins,
            **pesnet_config
        )
        self.params = jax_utils.replicate(self.params)

        # Vmap are all over the number of configurations
        self.batch_gnn = jax.vmap(
            self.pesnet_fns.gnn.apply, in_axes=(None, 0))
        # For FermiNet first vmap over electrons then vmap over configurations!
        self.batch_fermi = jax.vmap(
            jax.vmap(self.pesnet_fns.ferminet.apply, in_axes=(None, 0, None)), in_axes=(0, 0, 0))
        self.batch_get_fermi_params = jax.vmap(
            self.pesnet_fns.get_fermi_params, in_axes=(None, 0))
        self.pm_get_fermi_params = pmap(self.batch_get_fermi_params)

        # PESNet already uses a vmapped ferminet interallly, so we don't need to vmap over the electrons here
        self.batch_pesnet = jax.vmap(
            self.pesnet_fns.pesnet_fwd, in_axes=(None, 0, 0))
        self.pm_pesnet = pmap(self.batch_pesnet)
        self.batch_pesnet_orbitals = jax.vmap(
            self.pesnet_fns.pesnet_orbitals, in_axes=(None, 0, 0)
        )
        self.batch_fermi_orbitals = jax.vmap(jax.vmap(
            functools.partial(
                self.pesnet_fns.ferminet.apply,
                method=self.pesnet_fns.ferminet.orbitals
            ),
            in_axes=(None, 0, None)
        ), in_axes=(0, 0, 0))
        self.pm_gnn = pmap(self.batch_gnn)

        # GNN Energy computation
        self.surrogate = DimeNet([], [1], charges, **surrogate_config)
        self.key, subkey = jax.random.split(self.key)
        self.s_params = jax_utils.replicate(self.surrogate.init(subkey, jnp.ones((len(charges), 3))).unfreeze())
        self.surrogate_fwd = lambda params, nuclei: self.surrogate.apply(params, nuclei)[1][0].squeeze()
        self.batch_energy = jax.vmap(self.surrogate_fwd, in_axes=(None, 0))
        self.pm_energy = pmap(self.batch_energy)

        # Sampling
        # Here we need a seperate width per atom (since the atom configuration changes)
        self.width = jax_utils.broadcast(
            jnp.ones((self.num_devices,)) *
            sampler_config['init_width']
        )
        self.width_scheduler = MCMCStepSizeScheduler(self.width)
        self.key, subkey = jax.random.split(self.key)

        self.sampler = make_mcmc(self.batch_fermi, **sampler_config)
        self.pm_sampler = pmap(self.sampler)
        # We need to wrap the sampler to first produce the parameters of the ferminet
        # otherwise we would have to execute the GNN for every sampling iteration

        def pesnet_sampler(params, electrons, atoms, key, width):
            fermi_params = self.batch_get_fermi_params(params, atoms)
            return self.sampler(fermi_params, electrons, atoms, key, width)
        self.pesnet_sampler = pesnet_sampler
        self.pm_pesnet_sampler = pmap(self.pesnet_sampler)

        # Prepare random keys
        self.key, *subkeys = jax.random.split(self.key, self.num_devices+1)
        subkeys = jnp.stack(subkeys)
        self.shared_key = jax_utils.broadcast(subkeys)

        # Prepare energy computation
        # We first need to compute the parameters and feed them into the energy computation
        self.local_energy_fn = make_local_energy_function(
            self.pesnet_fns.ferminet.apply,
            atoms=None,
            charges=charges
        )
        self.batch_local_energy = jax.vmap(jax.vmap(
            self.local_energy_fn, in_axes=(None, 0, None)
        ), in_axes=(0, 0, 0))
        self.pm_local_energy = pmap(self.batch_local_energy)

        def local_energy(params, electrons, atoms):
            fermi_params = self.batch_get_fermi_params(params, atoms)
            return self.batch_local_energy(fermi_params, electrons, atoms)
        self.batch_pesnet_local_energy = local_energy
        self.pm_pesnet_local_energy = pmap(local_energy)

        # Prepare optimizer
        self.lr_schedule = make_schedule(optimization_config['lr'])

        ###########################################
        # Prepare VMC loss and gradient function
        self.use_cg = optimization_config['gradient'] == 'natural'
        self.opt_alg = optimization_config['optimizer']
        self.initialize_optimizer()

        self.train_state = None

        ###########################################
        # Pretraining
        self.initialize_pretrain_optimizer()

    def initialize_optimizer(self, optimizer: str = None, atoms: jax.Array = None):
        """Initializes the optimizer and training step.

        Args:
            optimizer (str, optional): Overwrites the optimizer in the training config. Defaults to None.
            atoms (jax.Array, optional): (..., M, 3) if specified an auxilliary loss is added
                which forces the parameters to stay close to the initial distribution. Defaults to None.
        """
        if optimizer is None:
            optimizer = self.opt_alg

        # Init optimizer
        lr_schedule = [
            optax.scale_by_schedule(self.lr_schedule),
            optax.scale(-1.)
        ]
        if optimizer == 'adam':
            self.optimizer = optax.chain(
                optax.scale_by_adam(),
                *lr_schedule
            )
        elif optimizer == 'sgd':
            self.optimizer = optax.chain(
                *lr_schedule
            )
        elif optimizer == 'sgd+clip':
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(
                    self.optimization_config['max_norm']),
                *lr_schedule
            )
        elif optimizer == 'rmsprop':
            self.optimizer = optax.chain(
                optax.scale_by_rms(),
                *lr_schedule
            )
        elif optimizer == 'lamb':
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.scale_by_adam(eps=1e-7),
                optax.scale_by_trust_ratio(),
                *lr_schedule
            )
        else:
            raise ValueError()

        # Initialize loss function
        if self.optimization_config['gradient'] == 'euclidean':
            self.loss = make_loss(
                self.batch_pesnet,
                normalize_gradient=True,
                **self.optimization_config)
            self.loss_and_grads = jax.value_and_grad(self.loss)
        elif self.optimization_config['gradient'] == 'natural':
            self.loss_and_grads = make_loss_and_natural_gradient_fn(
                self.batch_pesnet,
                **self.optimization_config,
                **self.optimization_config['cg'],
            )
            self.loss_and_grads = make_damping_fn(
                self.optimization_config['cg']['damping']['method'],
                self.loss_and_grads,
                self.optimization_config['cg']['damping']
            )
        else:
            raise ValueError(self.optimization_config['gradient'])

        # Initialize training step
        self.opt_state = jax.pmap(self.optimizer.init)(self.params)
        self.train_step = pmap(make_training_step(
            self.pesnet_sampler,
            self.loss_and_grads,
            self.batch_pesnet_local_energy,
            self.optimizer.update,
            uses_cg=self.use_cg
        ))

        # Initialize EMA and epoch counter
        self.epoch = jax_utils.replicate(jnp.zeros([]))
        self.train_state = None
        # Let's initialize it exactly like cg in scipy:
        # https://github.com/scipy/scipy/blob/edb50b09e92cc1b493242076b3f63e89397032e4/scipy/sparse/linalg/isolve/utils.py#L95
        if self.optimization_config['gradient'] == 'natural':
            self.train_state = {
                'last_grad': jtu.tree_map(lambda x: jnp.zeros_like(x), self.params),
                'damping': jax_utils.replicate(jnp.ones([])*self.optimization_config['cg']['damping']['init'])
            }
        

        # Surrogate training
        self.s_optimizer = optax.adamw(make_schedule(self.surrogate_optimization['lr']))
        self.s_emas = {
            'loss': jax_utils.replicate(ema_make(-1 * jnp.ones(()))),
            'err': jax_utils.replicate(ema_make(-1 * jnp.ones(()))),
            'params': p_ema_make(self.s_params)
        }
        self.s_opt_state = jax.pmap(self.s_optimizer.init)(self.s_params)
        self.s_train_step = pmap(make_surrogate_training_step(
            self.batch_energy,
            self.s_optimizer.update,
            **self.surrogate_optimization
        ))
        self.joint_train_step = make_joint_training_step(
            self.train_step,
            self.s_train_step
        )
        self.offset = replicate(jnp.zeros(()))

    def initialize_pretrain_optimizer(self, gnn_lr: float = 0, distinct_orbitals: bool = True):
        """Initializes the pretraining optimizer and update function.

        Args:
            train_gnn (bool, optional): Whether to train the GNN. Defaults to False.
        """
        trust_ratio_mask = flat_dict(jtu.tree_map(lambda _: True, self.params))
        for k in trust_ratio_mask:
            if any(v in k for v in ('BesselRBF', 'bias')):
                trust_ratio_mask[k] = False
        gnn_mask = flat_dict(jtu.tree_map(lambda _: True, self.params))
        for k in gnn_mask:
            if 'gnn' in k and 'Out' in k:
                if 'embedding' in k:
                    gnn_mask[k] = False
                if f'Dense_{str(self.pesnet_fns.gnn.out_mlp_depth - 1)}.bias' in k:
                    gnn_mask[k] = False
            if 'ferminet' in k:
                gnn_mask[k] = False
        self.pre_opt_init, self.pre_opt_update = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.masked(optax.scale_by_trust_ratio(), unflat_dict(trust_ratio_mask)),
            optax.masked(optax.scale(gnn_lr), unflat_dict(gnn_mask)),
            optax.scale(-3e-3)
        )
        self.pre_opt_state = jax.pmap(self.pre_opt_init)(self.params)
        self.pretrain_epoch = replicate(jnp.zeros(()))

        self.pretrain_step = pmap(make_pretrain_step(
            lambda p, e, a, k, w: make_mcmc(self.batch_fermi, **{**self.sampler_config, 'steps': 1})(self.batch_get_fermi_params(p, a), e, a, k, w),
            self.batch_fermi_orbitals,
            self.batch_get_fermi_params,
            self.pre_opt_update,
            full_det=self.pesnet_fns.ferminet.full_det,
            distinct_orbitals=distinct_orbitals
        ))

    def get_fermi_params(self, atoms: jax.Array) -> ParamTree:
        """Returns the full Ansatz parameters for the given molecular structures.

        Args:
            atoms (jax.Array): (..., M, 3)

        Returns:
            ParamTree: Ansatz parameters
        """
        return self.pm_get_fermi_params(self.params, atoms)

    def thermalize_samples(
            self,
            electrons: jax.Array,
            atoms: jax.Array,
            n_iter: int,
            show_progress: bool = True,
            adapt_step_width: bool = False) -> jax.Array:
        """Thermalize electrons.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)
            n_iter (int): Number of thermalizing steps to take.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.
            adapt_step_width (bool, optional): Whether to adapt the step width. Defaults to False.

        Returns:
            jax.Array: thermalized electrons
        """
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        with tqdm.trange(n_iter, desc='Thermalizing',
                            leave=False, disable=not show_progress) as iters:
            try:
                for _ in iters:
                    self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
                    electrons, pmove = self.pm_sampler(
                        fermi_params,
                        electrons,
                        atoms,
                        subkeys,
                        self.width
                    )
                    if adapt_step_width:
                        self.width = self.width_scheduler(pmove.mean())
                    iters.set_postfix({
                        'pmove': pmove[0, 0],
                        'width': self.width[0]
                    })
            except KeyboardInterrupt:
                return electrons
        return electrons

    def update_step(self, electrons: jax.Array, atoms: jax.Array):
        """Does an parameter update step.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)

        Returns:
            (jax.Array, (jax.Array, jax.Array, jax.Array)): (new electrons, (energy, energy variance, pmove))
        """
        # Do update step
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        (electrons, self.params, self.opt_state, e_l, E, E_var, pmove, self.train_state), (self.s_params, self.s_opt_state, self.s_emas), aux_data = self.joint_train_step(
            self.epoch,
            atoms,
            dict(
                params=self.params,
                electrons=electrons,
                opt_state=self.opt_state,
                key=subkeys,
                mcmc_width=self.width,
                **self.train_state
            ),
            dict(
                params=self.s_params,
                opt_state=self.s_opt_state,
                emas=self.s_emas,
                offset=self.offset
            )
        )
        pmove = pmove.item()
        self.width = self.width_scheduler(pmove)

        self.epoch += 1
        return electrons, (e_l, E, E_var, pmove), aux_data
    
    def surrogate_energies(self, atoms):
        """
        Infer energy for a given geometry based on the surrogate model.
        """
        return self.pm_energy(p_ema_value(self.s_emas['params']), atoms) + self.offset[:, None]

    def pre_update_step(self, electrons: jax.Array, atoms: jax.Array, scfs: List[Scf]):
        """Performs an pretraining update step.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)
            scfs (List[Scf]): List of SCF solutions for the provided atom configurations.

        Returns:
            (jax.Array, jax.Array, jax.Array): (loss, new electrons, move probability)
        """
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        targets = eval_orbitals(scfs, electrons, self.spins)

        self.params, electrons, self.pre_opt_state, loss, pmove = self.pretrain_step(
            self.pretrain_epoch,
            self.params,
            electrons,
            atoms,
            targets,
            self.pre_opt_state,
            subkeys,
            self.width,
            jnp.arange(jax.device_count())
        )
        self.width = self.width_scheduler(pmove.mean())
        self.pretrain_epoch += 1
        return loss, electrons, pmove

    def eval_energy(
            self,
            electrons: jax.Array,
            atoms: jax.Array,
            n_repeats: int,
            thermalize_steps: int,
            show_progress: bool = True) -> jax.Array:
        """Evaluates the energy for the given molecular structure.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)
            n_repeats (int): How often we sample and compute the energy
            thermalize_steps (int): Thermalizing steps between energy computations
            show_progress (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            jax.Array: evaluated energies
        """
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        n_configs = np.prod(atoms.shape[:2])
        means = [OnlineMean() for _ in range(n_configs)]

        def iter_step(key, electrons):
            for _ in range(thermalize_steps):
                key, subkey = jax_utils.p_split(key)
                electrons, pmove = self.pm_sampler(
                    fermi_params,
                    electrons,
                    atoms,
                    subkey,
                    self.width
                )
            energies = self.pm_local_energy(
                fermi_params, electrons, atoms
            )
            return key, energies, electrons

        total_energies = []
        with tqdm.trange(n_repeats, desc='Computing Energy', disable=not show_progress) as iters:
            for _ in iters:
                self.shared_key, energies, electrons = iter_step(
                    self.shared_key, electrons)
                total_energies.append(energies.reshape(n_configs, -1))
                for e, mean in zip(energies.reshape(n_configs, -1), means):
                    mean.update(e)
                iters.set_postfix({
                    'E': '\t'.join(map(str, means))
                })
        total_energies = np.concatenate(total_energies, -1)
        return total_energies

    def checkpoint(self, folder: str, name: str):
        """Store checkpoint.

        Args:
            folder (str): Folder to store the checkpoint
            name (str): Checkpoint name
        """
        with gzip.open(os.path.join(folder, name), 'wb') as out:
            pickle.dump(jtu.tree_map(instance, dict(
                params=to_numpy(self.params),
                opt_state=to_numpy(self.opt_state),
                s_params=to_numpy(self.s_params),
                s_emas=to_numpy(self.s_emas),
                s_opt_state=to_numpy(self.s_opt_state),
                width=to_numpy(self.width),
                train_state=to_numpy(self.train_state),
                epoch=to_numpy(self.epoch),
                offset=to_numpy(self.offset)
            )), out)

    def load_checkpoint(self, file_path):
        """Load checkpoint

        Args:
            file_path (str): Path to checkpoint file
        """
        try:
            with gzip.open(file_path, 'rb') as inp:
                data = jtu.tree_map(replicate, to_jnp(pickle.load(inp)))
        except:
            with open(file_path, 'rb') as inp:
                data = {
                    k: v.item() if v.size == 1 else v
                    for k, v in dict(np.load(inp, allow_pickle=True)).items()
                }
                for k in ['params', 's_params', 's_emas', 'train_state']:
                    data[k] = jax.tree_map(lambda x: replicate(x[0]), to_jnp(data[k]))
        del self.params
        del self.s_params
        del self.s_emas
        if hasattr(self, 's_opt_state'):
            del self.s_opt_state
        if hasattr(self, 'opt_state'):
            del self.opt_state
        del self.width
        del self.epoch
        if hasattr(self, 'train_state'):
            del self.train_state
        for k, v in data.items():
            setattr(self, k, v)
        self.width_scheduler = MCMCStepSizeScheduler(self.width)
        
    @property
    def parameters(self):
        return {
            **self.params,
            'surrogate': self.s_params,
            'surrogate_val': self.s_emas['params'],
        }
    
    @parameters.setter
    def parameters(self, value):
        self.params = {
            'gnn': value['gnn'],
            'ferminet': value['ferminet']
        }
        self.s_params = value['surrogate']
        self.s_emas['params'] = value['surrogate_val']
