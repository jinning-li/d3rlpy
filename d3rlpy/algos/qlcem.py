from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    QFuncArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_q_func,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from ..models.q_functions import QFunctionFactory
from .base import AlgoBase
from .torch.qlcem_impl import QLCEMImpl


class QLCEM(AlgoBase):
    r"""Q Learning with Cross Entropy Method policy.

    This algorithm only learns Q functions. No policy is learned. 
    """

    _critic_learning_rate: float
    _critic_optim_factory: OptimizerFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _impl: Optional[QLCEMImpl]

    def __init__(
        self,
        *,
        critic_learning_rate: float = 3e-4,
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        critic_encoder_factory: EncoderArg = "default",
        q_func_factory: QFuncArg = "mean",
        batch_size: int = 100,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_critics: int = 1,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        impl: Optional[QLCEMImpl] = None,
        **kwargs: Any
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._critic_learning_rate = critic_learning_rate
        self._critic_optim_factory = critic_optim_factory
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._q_func_factory = check_q_func(q_func_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = check_use_gpu(use_gpu)
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = QLCEMImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            critic_learning_rate=self._critic_learning_rate,
            critic_optim_factory=self._critic_optim_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
        critic_loss = self._impl.update_critic(batch)
        self._impl.update_critic_target()

        return {"critic_loss": critic_loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
