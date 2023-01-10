import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_continuous_q_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    EnsembleContinuousQFunction,
    EnsembleQFunction,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api
from .base import TorchImplBase
from .utility import ContinuousQFunctionMixin
from .cem import CEM


class QLCEMBaseImpl(ContinuousQFunctionMixin, TorchImplBase, metaclass=ABCMeta):

    _critic_learning_rate: float
    _critic_optim_factory: OptimizerFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _tau: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleContinuousQFunction]
    _targ_q_func: Optional[EnsembleContinuousQFunction]
    _critic_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        critic_learning_rate: float,
        critic_optim_factory: OptimizerFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._critic_learning_rate = critic_learning_rate
        self._critic_optim_factory = critic_optim_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._critic_optim = None
        self._cem = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()

        self._build_cem()

    def _build_cem(self) -> None:
        assert len(self.observation_shape) == 1
        assert self._q_func is not None
        self._cem = CEM(
            None, 
            self._q_func,  
            self.observation_shape[0], 
            self.action_size, 
            num_samples=50, 
            num_iterations=50,
            num_elite=5,
            horizon=1, 
            device=self._device, 
            u_max=torch.tensor(1, dtype=torch.double, device=self.device), 
            u_min=torch.tensor(-1, dtype=torch.double, device=self.device),
            init_cov_diag=1,
        )

    def _build_critic(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )


    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )


    @abstractmethod
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._cem is not None
        with torch.no_grad():
            best_ac = self._cem.command(x)
        return best_ac


    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Cross Entropy policy does not support sample!"
        )

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)


    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._critic_optim
        return self._critic_optim


class QLCEMImpl(QLCEMBaseImpl):


    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)
