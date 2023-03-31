from jax import Array
from chex import PRNGKey, ArrayTree, Shape
from typing import Callable, NamedTuple, Protocol

from pesnet.utils import EMA


Parameters = ArrayTree
AuxData = ArrayTree


class WaveFunction(Protocol):
    def __call__(self, params: Parameters, electrons: Array, atoms: Array) -> Array:
        pass


class OrbitalFunction(Protocol):
    def __call__(self, params: Parameters, electrons: Array, atoms: Array) -> Array:
        pass


class ParameterFunction(Protocol):
    def __call__(self, params: Parameters, atoms: Array) -> Parameters:
        pass


class NaturalGradientInit(Protocol):
    def __call__(self, params: Parameters) -> "NaturalGradientState":
        pass


class NaturalGradientPreconditioner(Protocol):
    def __call__(
        self,
        params: Parameters,
        norm: Array,
        input: ArrayTree,
        damping_input: ArrayTree,
        gradient: ArrayTree,
        natgrad_state: "NaturalGradientState"
    ) -> tuple[ArrayTree, "NaturalGradientState"]:
        pass


class NaturalGradientState(NamedTuple):
    last_grad: Parameters
    damping_state: NamedTuple


class NaturalGradient(NamedTuple):
    precondition: NaturalGradientPreconditioner
    init: NaturalGradientInit


class McmcFn(Protocol):
    def __call__(
        self,
        key: PRNGKey,
        params: Parameters,
        electrons: Array,
        atoms: Array,
        width: Array
    ) -> tuple[Array, Array]:
        pass


class GradientFn(Protocol):
    def __call__(
        self,
        params: Parameters,
        electrons: Array,
        atoms: Array,
        local_energy: Array,
        natgrad_state: NaturalGradientState
    ) -> tuple[Parameters, NaturalGradientState, AuxData]:
        pass


class EnergyFn(Protocol):
    def __call__(
        self,
        params: Parameters,
        electrons: Array,
        atoms: Array
    ) -> Array:
        pass


####################################################################################
# VMC
####################################################################################
class VMCTempResult(NamedTuple):
    electrons: Array
    local_energies: Array
    energies: Array
    energy_variances: Array
    pmove: Array


class VMCState(NamedTuple):
    params: Parameters
    opt_state: ArrayTree
    natgrad_state: NaturalGradientState


class VMCTrainingStep(Protocol):
    def __call__(
        self,
        key: PRNGKey,
        state: VMCState,
        electrons: Array,
        atoms: Array,
        mcmc_width: Array
    ) -> tuple[VMCTempResult, VMCState, AuxData]:
        pass


class PretrainStep(Protocol):
    def __call__(
        self,
        key: PRNGKey,
        params: Parameters,
        electrons: Array,
        atoms: Array,
        targets: tuple[Array, ...],
        opt_state: ArrayTree,
        mcmc_width: Array
    ) -> tuple[Parameters, Array, ArrayTree, Array, Array]:
        pass


####################################################################################
# Surrogate
####################################################################################
class SurrogateEMAs(NamedTuple):
    loss: EMA
    err: EMA
    params: EMA


class SurrogateState(NamedTuple):
    params: ArrayTree
    opt_state: ArrayTree
    emas: SurrogateEMAs
    offset: Array


class Surrogate(Protocol):
    def __call__(
        self,
        params: Parameters,
        atoms: Array
    ) -> Array:
        pass


SurrogateInitFn = Callable[[Parameters], SurrogateState]


class SurrogateUpdateFn(Protocol):
    def __call__(
        self,
        state: SurrogateState,
        atoms: Array,
        local_energy: Array
    ) -> tuple[SurrogateState, AuxData]:
        pass


class SurrogateTraining(NamedTuple):
    init: SurrogateInitFn
    update: SurrogateUpdateFn


class JointTrainingStep(Protocol):
    def __call__(
        self,
        atoms: Array,
        vmc_args: ArrayTree,
        surrogate_args: ArrayTree
    ) -> tuple[VMCTempResult, VMCState, SurrogateState, AuxData]:
        pass
