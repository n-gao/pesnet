import functools
import os
from typing import Dict, List, Tuple

import jax
import jax.experimental.host_callback
import jax.numpy as jnp
import numpy as np
import optax
import tqdm.auto as tqdm

from pesnet import jax_utils
from pesnet.damping import make_damping_fn
from pesnet.hamiltonian import make_local_energy_function
from pesnet.jax_utils import pmap
from pesnet.mcmc import make_mcmc
from pesnet.nn import ParamTree
from pesnet.optim import accumulate
from pesnet.pesnet import make_pesnet
from pesnet.pretrain import eval_orbitals, make_pretrain_step
from pesnet.systems.scf import Scf
from pesnet.training import (make_loss, make_loss_and_natural_gradient_fn,
                             make_schedule, make_training_step)
from pesnet.utils import (EMAPyTree, MCMCStepSizeScheduler, OnlineMean, to_jnp, to_numpy)


class PesVmc:
    """This class contains a lot of utility functions to ease
    the training of models on the potential energy surface.
    It takes care of initializing the network, sampler,
    energy computation and optimization. Further, it includes
    utility functions to thermalize samples, evaluate the
    energies and checkpointing.
    """

    def __init__(
        self,
        key: jnp.ndarray,
        charges: Tuple[int, ...],
        spins: Tuple[int, int],
        pesnet_config: Dict,
        sampler_config: Dict,
        training_config: Dict,
        train_state_deacy: float = 0.0
    ) -> None:
        """Constructor.

        Args:
            key (jnp.ndarray): jax.random.PRNGKey
            charges (Tuple[int, ...]): (M) charges of the system.
            spins (Tuple[int, int]): spins of the system.
            pesnet_config (Dict): Wfnet parameters, see 'pesnet.py'.
            sampler_config (Dict): Sampler parameters, see 'mcmc.py'.
            training_config (Dict): Optimization configuration.
            train_state_deacy (float, optional): EWM decay factor for the training state. Defaults to 0.0.
        """

        self.key = key

        self.num_devices = jax.device_count()

        self.charges = charges
        self.spins = spins

        self.pesnet_config = pesnet_config
        self.sampler_config = sampler_config
        self.training_config = training_config
        self.train_state_deacy = train_state_deacy

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

        # WfNet already uses a vmapped ferminet interallly, so we don't need to vmap over the electrons here
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

        # Sampling
        self.width = jax_utils.broadcast(
            jnp.ones((self.num_devices,)) *
            sampler_config['init_width']
        )
        self.width_scheduler = MCMCStepSizeScheduler(self.width, update_interval=100)
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
        self.lr_gnn_prefactor = training_config['lr']['gnn_prefactor'] if 'gnn_prefactor' in training_config else 1.
        self.lr_schedule = make_schedule(training_config['lr'])

        ###########################################
        # Prepare VMC loss and gradient function
        self.use_cg = training_config['gradient'] == 'natural'
        self.opt_alg = training_config['optimizer']
        self.initialize_optimizer()

        self.train_state = EMAPyTree()

        ###########################################
        # Pretraining
        self.initialize_pretrain_optimizer()

    def initialize_optimizer(self, optimizer: str = None):
        """Initializes the optimizer and training step.

        Args:
            optimizer (str, optional): Overwrites the optimizer in the training config. Defaults to None.
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
            if self.training_config['accumulate_n'] > 1:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(
                        self.training_config['max_norm']),
                    accumulate(self.training_config['accumulate_n']),
                    *lr_schedule
                )
            else:
                self.optimizer = optax.chain(
                    optax.clip_by_global_norm(
                        self.training_config['max_norm']),
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
        if self.training_config['gradient'] == 'euclidean':
            self.loss = make_loss(
                self.batch_pesnet,
                normalize_gradient=True,
                **self.training_config)
            self.loss_and_grads = jax.value_and_grad(self.loss)
        elif self.training_config['gradient'] == 'natural':
            self.loss_and_grads = make_loss_and_natural_gradient_fn(
                self.batch_pesnet,
                **self.training_config,
                **self.training_config['cg']
            )
            self.loss_and_grads = make_damping_fn(
                self.training_config['cg']['damping']['method'],
                self.loss_and_grads,
                self.training_config['cg']['damping']
            )
        else:
            raise ValueError(self.training_config['gradient'])

        # Initialize training step
        self.opt_state = jax.pmap(self.optimizer.init)(self.params)
        self.train_step = pmap(make_training_step(
            self.pesnet_sampler,
            self.loss_and_grads,
            self.batch_pesnet_local_energy,
            self.optimizer.update,
            uses_cg=self.use_cg,
        ))

        # Initialize EMA and epoch counter
        self.epoch = jax_utils.replicate(jnp.zeros([]))
        self.train_state = EMAPyTree()
        # Initialize CG init guess exactly like cg in scipy:
        # https://github.com/scipy/scipy/blob/edb50b09e92cc1b493242076b3f63e89397032e4/scipy/sparse/linalg/isolve/utils.py#L95
        if self.training_config['gradient'] == 'natural':
            self.train_state.update({
                'last_grad': jax.tree_map(lambda x: jnp.zeros_like(x), self.params),
                'damping': jax_utils.replicate(jnp.ones([])*self.training_config['cg']['damping']['init'])
            }, 0)

    def initialize_pretrain_optimizer(self, train_gnn: bool = False):
        """Initializes the pretraining optimizer and update function.

        Args:
            train_gnn (bool, optional): Whether to train the GNN. Defaults to False.
        """
        self.pre_opt_init, self.pre_opt_update = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_trust_ratio(),
            optax.scale(-3e-3),
        )
        self.pre_opt_state = jax.pmap(self.pre_opt_init)(self.params)

        self.pretrain_step = pmap(make_pretrain_step(
            self.pesnet_sampler,
            self.batch_fermi_orbitals,
            self.batch_get_fermi_params,
            self.pre_opt_update,
            train_gnn=train_gnn
        ))

    def get_fermi_params(self, atoms: jnp.ndarray) -> ParamTree:
        """Returns the full Ansatz parameters for the given molecular structures.

        Args:
            atoms (jnp.ndarray): (..., M, 3)

        Returns:
            ParamTree: Ansatz parameters
        """
        return self.pm_get_fermi_params(self.params, atoms)

    def thermalize_samples(
            self,
            electrons: jnp.ndarray,
            atoms: jnp.ndarray,
            n_iter: int,
            show_progress: bool = True,
            adapt_step_width: bool = False) -> jnp.ndarray:
        """Thermalize electrons.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            n_iter (int): Number of thermalizing steps to take.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.
            adapt_step_width (bool, optional): Whether to adapt the step width. Defaults to False.

        Returns:
            jnp.ndarray: thermalized electrons
        """
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        with tqdm.trange(n_iter, desc='Thermalizing',
                         leave=False, disable=not show_progress) as iters:
            try:
                for _ in iters:
                    self.shared_key, subkeys = jax_utils.p_split(
                        self.shared_key)
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

    def update_step(self, electrons: jnp.ndarray, atoms: jnp.ndarray):
        """Does an parameter update step.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)

        Returns:
            (jnp.ndarray, (jnp.ndarray, jnp.ndarray, jnp.ndarray)): (new electrons, (energy, energy variance, pmove))
        """
        # Do update step
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        (electrons, self.params, self.opt_state, E, E_var, pmove), train_state = self.train_step(
            self.epoch,
            self.params,
            electrons,
            atoms,
            self.opt_state,
            subkeys,
            self.width,
            **self.train_state.value
        )
        self.train_state.update(train_state, self.train_state_deacy)
        self.width = self.width_scheduler(np.mean(pmove))

        self.epoch += 1
        return electrons, (E, E_var, pmove)

    def pre_update_step(self, electrons: jnp.ndarray, atoms: jnp.ndarray, scfs: List[Scf], signs: Tuple[np.ndarray, np.ndarray] = None):
        """Performs an pretraining update step.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            scfs (List[Scf]): List of SCF solutions for the provided atom configurations.

        Returns:
            (jnp.ndarray, jnp.ndarray, jnp.ndarray): (loss, new electrons, move probability)
        """
        self.shared_key, subkeys = jax_utils.p_split(self.shared_key)
        targets = eval_orbitals(scfs, electrons, self.spins, signs)

        self.params, electrons, self.pre_opt_state, loss, pmove = self.pretrain_step(
            self.params,
            electrons,
            atoms,
            targets,
            self.pre_opt_state,
            subkeys,
            self.width
        )
        self.width = self.width_scheduler(pmove.mean())
        return loss, electrons, pmove

    def eval_energy(
            self,
            electrons: jnp.ndarray,
            atoms: jnp.ndarray,
            n_repeats: int,
            thermalize_steps: int,
            show_progress: bool = True) -> jnp.ndarray:
        """Evaluates the energy for the given molecular structure.

        Args:
            electrons (jnp.ndarray): (..., N, 3)
            atoms (jnp.ndarray): (..., M, 3)
            n_repeats (int): How often we sample and compute the energy
            thermalize_steps (int): Thermalizing steps between energy computations
            show_progress (bool, optional): Whether to print progress. Defaults to True.

        Returns:
            jnp.ndarray: evaluated energies
        """
        fermi_params = self.pm_get_fermi_params(self.params, atoms)
        n_configs = np.prod(atoms.shape[:2])
        means = [OnlineMean() for _ in range(n_configs)]

        def iter_step(key, electrons):
            for j in range(thermalize_steps):
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
        with open(os.path.join(folder, name), 'wb') as out:
            np.savez(out,
                     params=to_numpy(self.params),
                     opt_state=to_numpy(self.opt_state),
                     width=to_numpy(self.width),
                     train_state=to_numpy(self.train_state.value),
                     epoch=to_numpy(self.epoch)
                     )

    def load_checkpoint(self, file_path):
        """Load checkpoint

        Args:
            file_path (str): Path to checkpoint file
        """
        with open(file_path, 'rb') as inp:
            data = dict(np.load(inp, allow_pickle=True))
        self.params = jax_utils.broadcast(to_jnp(data['params'].item()))
        self.opt_state = jax_utils.broadcast(to_jnp(list(data['opt_state'])))
        self.width = jax_utils.broadcast(to_jnp(data['width']))
        self.epoch = jax_utils.broadcast(to_jnp(data['epoch']))
        self.train_state = EMAPyTree()
        self.train_state.update(jax_utils.broadcast(
            to_jnp(data['train_state'].item())), 0)
