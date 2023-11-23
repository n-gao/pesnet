import functools
import gzip
import os
import pickle

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import tqdm.auto as tqdm
from chex import ArrayTree, PRNGKey
from flax.core import unfreeze

from pesnet.nn.dimenet import DimeNet
from pesnet.nn.pesnet import make_pesnet
from pesnet.surrogate.training import make_surrogate_training_step
from pesnet.systems.scf import Scf
from pesnet.utils import (MCMCStepSizeScheduler, OnlineMean, p_ema_value,
                          to_jnp, to_numpy)
from pesnet.utils.jax_utils import (broadcast, instance, p_split, pmap,
                                    replicate)
from pesnet.utils.optim import (make_natural_gradient_preconditioner,
                                make_optimizer,
                                scale_by_trust_ratio_embeddings)
from pesnet.utils.typing import (EnergyFn, JointTrainingStep, McmcFn,
                                 OrbitalFunction, ParameterFunction,
                                 Parameters, Surrogate, SurrogateUpdateFn,
                                 VMCTrainingStep, WaveFunction)
from pesnet.vmc.hamiltonian import make_local_energy_function
from pesnet.vmc.mcmc import make_mcmc
from pesnet.vmc.pretrain import eval_orbitals, make_pretrain_step
from pesnet.vmc.update import (VMCState, VMCTempResult, coordinate_transform,
                               make_gradient_fn, make_training_step)


def make_joint_training_step(
        vmc_step: VMCTrainingStep,
        surrogate_step: SurrogateUpdateFn,
    ) -> JointTrainingStep:
    """
    Creates a joint training step function that combines variational Monte Carlo (VMC) and surrogate training.

    Args:
        vmc_step: A function that performs a single step of variational Monte Carlo (VMC) calculation.
        surrogate_step: A function that performs a single step of surrogate training.

    Returns:
        A joint training step function that performs both VMC and surrogate training.
    """
    def step(
        atoms: jax.Array,
        vmc_args: ArrayTree,
        surrogate_args: ArrayTree,
        ):
        aux_data = {}
        vmc_tmp, vmc_result, aux_data['cg'] = vmc_step(
            atoms=atoms,
            **vmc_args)
        surr_result, aux_data['surrogate'] = surrogate_step(
            E_l=vmc_tmp.local_energies,
            atoms=atoms,
            **surrogate_args
        )
        # Remove n_gpu axis
        aux_data = jtu.tree_map(lambda x: jnp.mean(x), aux_data)
        return vmc_tmp, vmc_result, surr_result, aux_data
    return step


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
        key: PRNGKey,
        charges: tuple[int, ...],
        spins: tuple[int, int],
        pesnet_config: dict,
        sampler_config: dict,
        optimization_config: dict,
        surrogate_config: dict,
        surrogate_optimization: dict,
        pretrain_config: dict
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
            pretrain_config (Dict): Pretraining configuration.
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
        self.pretrain_config = pretrain_config

        self.key, subkey = jax.random.split(key)
        params, self.pesnet_fns = make_pesnet(
            subkey,
            charges=charges,
            spins=spins,
            **pesnet_config
        )
        params = replicate(params)

        # Vmap are all over the number of configurations
        conf_axes_pes = (None, 0, 0)
        conf_axes_fermi = (0, 0, 0)
        elec_axes = (None, 0, None)

        self.batch_gnn = jax.vmap(
            self.pesnet_fns.gnn.apply,
            in_axes=(None, 0)
        )
        # For FermiNet first vmap over electrons then vmap over configurations!
        self.batch_fermi: WaveFunction = jax.vmap(jax.vmap(
            self.pesnet_fns.ferminet.apply,
            in_axes=conf_axes_fermi),
            in_axes=elec_axes
        )
        self.batch_get_fermi_params: ParameterFunction = jax.vmap(
            self.pesnet_fns.get_fermi_params,
            in_axes=(None, 0)
        )
        self.pm_get_fermi_params = pmap(self.batch_get_fermi_params)

        # PESNet functions
        self.batch_pesnet: WaveFunction = jax.vmap(jax.vmap(
            self.pesnet_fns.pesnet_fwd,
            in_axes=conf_axes_pes),
            in_axes=elec_axes
        )
        self.pm_pesnet = pmap(self.batch_pesnet)
        # We disable the equivariant frame for pretraining
        self.batch_pesnet_orbitals: OrbitalFunction = jax.vmap(jax.vmap(
            self.pesnet_fns.pesnet_orbitals,
            in_axes=conf_axes_pes),
            in_axes=elec_axes
        )

        # Coordinate transformation
        self.coordinate_transform = pmap(
            jax.vmap(jax.vmap(
            coordinate_transform,
            in_axes=(0, 0, 0)), # configurations
            in_axes=(None, None, 0)) # electrons
        )

        # GNN Energy computation
        self.surrogate = DimeNet([], [1], charges, **surrogate_config)
        self.key, subkey = jax.random.split(self.key)
        s_params = replicate(unfreeze(self.surrogate.init(subkey, jnp.ones((len(charges), 3)))))
        self.surrogate_fwd: Surrogate = lambda params, nuclei: self.surrogate.apply(params, nuclei)[1][0].squeeze()
        self.batch_energy = jax.vmap(self.surrogate_fwd, in_axes=(None, 0))
        self.pm_energy = pmap(self.batch_energy)

        # Sampling
        # Here we need a seperate width per atom (since the atom configuration changes)
        self.width = replicate(jnp.ones(()) * sampler_config['init_width'])
        self.width_scheduler = MCMCStepSizeScheduler(self.width)
        self.key, subkey = jax.random.split(self.key)

        self.sampler = make_mcmc(self.batch_fermi, **sampler_config)
        self.pm_sampler = pmap(self.sampler)

        # We need to wrap the sampler to first produce the parameters of the ferminet
        # otherwise we would have to execute the GNN for every sampling iteration
        def pesnet_sampler(key, params, electrons, atoms, width):
            fermi_params = self.batch_get_fermi_params(params, atoms)
            return self.sampler(key, fermi_params, electrons, atoms, width)
        self.pesnet_sampler: McmcFn = pesnet_sampler
        self.pm_pesnet_sampler = pmap(self.pesnet_sampler)

        # Prepare random keys
        self.key, *subkeys = jax.random.split(self.key, self.num_devices+1)
        subkeys = jnp.stack(subkeys)
        self.shared_key = broadcast(subkeys)

        # Prepare energy computation
        # We first need to compute the parameters and feed them into the energy computation
        self.local_energy_fn = make_local_energy_function(
            self.pesnet_fns.ferminet.apply,
            atoms=None,
            charges=charges
        )
        self.batch_local_energy = jax.vmap(jax.vmap(
            self.local_energy_fn,
            in_axes=conf_axes_fermi),
            in_axes=elec_axes
        )
        self.pm_local_energy = pmap(self.batch_local_energy)

        def pesnet_local_energy(params, electrons, atoms):
            fermi_params = self.pesnet_fns.get_fermi_params(params, atoms)
            return self.local_energy_fn(fermi_params, electrons, atoms)
        self.pesnet_local_energy: EnergyFn = pesnet_local_energy
        self.pesnet_batch_local_energy = jax.vmap(jax.vmap(
            self.pesnet_local_energy,
            in_axes=conf_axes_pes,),
            in_axes=elec_axes
        )
        self.pm_pesnet_local_energy = pmap(self.pesnet_batch_local_energy)

        ###########################################
        # Prepare VMC loss and gradient function        
        # Init optimizer
        self.optimizer = make_optimizer(**self.optimization_config['optimizer_args'])

        # Initialize loss function
        self.nat_grad = make_natural_gradient_preconditioner(
            self.batch_pesnet,
            **self.optimization_config['cg']
        )
        self.grad_fn = make_gradient_fn(
            self.batch_pesnet,
            self.nat_grad.precondition,
            self.optimization_config['clip_local_energy'],
            self.optimization_config['clip_stat']
        )

        # Initialize training step
        self.train_step = pmap(make_training_step(
            self.pesnet_sampler,
            self.grad_fn,
            self.pesnet_batch_local_energy,
            self.optimizer.update
        ))
        self.vmc_state = VMCState(
            params,
            jax.pmap(self.optimizer.init)(params),
            jax.pmap(self.nat_grad.init)(params)
        )

        # Initialize epoch counter
        self.epoch = replicate(jnp.zeros([]))        

        # Surrogate training
        self.s_optimizer = make_optimizer(**self.surrogate_optimization['optimizer_args'])
        self.s_trainer = make_surrogate_training_step(
            self.batch_energy,
            self.s_optimizer,
            **self.surrogate_optimization
        )
        self.joint_train_step = make_joint_training_step(
            self.train_step,
            pmap(self.s_trainer.update)
        )
        self.surr_state = pmap(self.s_trainer.init)(s_params)

        ###########################################
        # Pretraining
        embedding_mask = jtu.tree_map_with_path(
            lambda p, _: 'Embedding' in jtu.keystr(p),
            params
        )
        kernel_mask = jtu.tree_map_with_path(
            lambda p, _: 'kernel' in jtu.keystr(p),
            params
        )
        def is_intermediate_gnn(p, _):
            p = jtu.keystr(p)
            if "'ferminet'" in p:
                return False
            if "'gnn'" in p and "Out" in p:
                    if 'embedding' in p:
                        return False
                    if f'Dense_{self.pesnet_fns.gnn.out_mlp_depth - 1}' and 'bias' in p:
                        return False
            return True

        self.pre_opt_init, self.pre_opt_update = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.masked(optax.scale_by_trust_ratio(), kernel_mask),
            optax.masked(scale_by_trust_ratio_embeddings(), embedding_mask),
            optax.scale_by_schedule(lambda t: -1e-3 * 1/(1+t/1000)),
        )
        self.pre_opt_state = jax.pmap(self.pre_opt_init)(self.params)
        self.pretrain_epoch = replicate(jnp.zeros(()))

        self.pretrain_step = pmap(make_pretrain_step(
            lambda k, p, e, a, w: make_mcmc(self.batch_fermi, **{**self.sampler_config, 'steps': 1})(k, self.batch_get_fermi_params(p, a), e, a, w),
            self.batch_pesnet_orbitals,
            self.pre_opt_update,
            full_det=self.pesnet_fns.ferminet.full_det,
            distinct_orbitals=pretrain_config['distinct_orbitals']
        ))
    
    @property
    def params(self) -> Parameters:
        return self.vmc_state.params
    
    @params.setter
    def params(self, params: Parameters):
        self.vmc_state = self.vmc_state._replace(params=params)

    @property
    def offset(self) -> jax.Array:
        return instance(self.surr_state.offset)
    
    @offset.setter
    def offset(self, offset: jax.Array):
        self.surr_state = self.surr_state._replace(
            offset=replicate(jnp.array(offset))
        )

    def get_fermi_params(self, atoms: jax.Array) -> Parameters:
        """Returns the full Ansatz parameters for the given molecular structures.

        Args:
            atoms (jax.Array): (..., M, 3)

        Returns:
            ArrayTree: Ansatz parameters
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
                    self.shared_key, subkeys = p_split(self.shared_key)
                    electrons, pmove = self.pm_sampler(
                        subkeys,
                        fermi_params,
                        electrons,
                        atoms,
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

    def update_step(self, electrons: jax.Array, atoms: jax.Array) -> tuple[VMCTempResult, ArrayTree]:
        """Does an parameter update step.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)

        Returns:
            (jax.Array, (jax.Array, jax.Array, jax.Array)): (new electrons, (energy, energy variance, pmove))
        """
        # Do update step
        self.shared_key, subkeys = p_split(self.shared_key)
        vmc_tmp, self.vmc_state, self.surr_state, aux_data = self.joint_train_step(
            atoms,
            dict(
                key=subkeys,
                state=self.vmc_state,
                electrons=electrons,
                mcmc_width=self.width
            ),
            dict(state=self.surr_state)
        )
        pmove = np.mean(vmc_tmp.pmove).item()
        self.width = self.width_scheduler(pmove)

        self.epoch += 1
        return vmc_tmp, aux_data
    
    def surrogate_energies(self, atoms: jax.Array) -> jax.Array:
        """
        Infer energy for a given geometry based on the surrogate model.
        """
        return self.pm_energy(p_ema_value(self.surr_state.emas.params), atoms) + self.surr_state.offset.flatten()[0]

    def pre_update_step(self, electrons: jax.Array, atoms: jax.Array, scfs: list[Scf]) -> list[jax.Array, jax.Array, jax.Array]:
        """Performs an pretraining update step.

        Args:
            electrons (jax.Array): (..., N, 3)
            atoms (jax.Array): (..., M, 3)
            scfs (List[Scf]): List of SCF solutions for the provided atom configurations.

        Returns:
            (jax.Array, jax.Array, jax.Array): (loss, new electrons, move probability)
        """
        self.shared_key, subkeys = p_split(self.shared_key)
        targets = eval_orbitals(scfs, electrons, self.spins)

        self.params, electrons, self.pre_opt_state, loss, pmove = self.pretrain_step(
            subkeys,
            self.params,
            electrons,
            atoms,
            targets,
            self.pre_opt_state,
            self.width
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
                key, subkey = p_split(key)
                electrons, _ = self.pm_sampler(
                    subkey,
                    fermi_params,
                    electrons,
                    atoms,
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
            pickle.dump(instance(dict(
                vmc_state=to_numpy(self.vmc_state),
                surr_state=to_numpy(self.surr_state),
                width=to_numpy(self.width),
                epoch=to_numpy(self.epoch),
            )), out)

    def load_checkpoint(self, file_path: str):
        """Load checkpoint

        Args:
            file_path (str): Path to checkpoint file
        """
        with gzip.open(file_path, 'rb') as inp:
            data = replicate(to_jnp(pickle.load(inp)))
        for k, v in data.items():
            setattr(self, k, v)
        self.width_scheduler = MCMCStepSizeScheduler(self.width)
