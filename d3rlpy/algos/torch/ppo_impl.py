import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import (
    create_squashed_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import (
    SquashedNormalPolicy,
    Policy,
    ValueFunction,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api, eval_api
from .base import TorchImplBase
from .utility import ContinuousVFunctionMixin


class PPOBaseImpl(ContinuousVFunctionMixin, TorchImplBase, metaclass=ABCMeta):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _v_func: Optional[ValueFunction]
    _policy: Optional[Policy]
    _old_v_func: Optional[ValueFunction]
    _old_policy: Optional[Policy]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        eps_clip: float, 
        policy_entropy_weight: float, 
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
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._eps_clip = eps_clip
        self._policy_entropy_weight = policy_entropy_weight
        self._use_gpu = use_gpu

        # initialized in build
        self._v_func = None
        self._policy = None
        self._old_v_func = None
        self._old_policy = None
        self._actor_optim = None
        self._critic_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self._old_v_func = copy.deepcopy(self._v_func)
        self._old_policy = copy.deepcopy(self._policy)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self) -> None:
        self._v_func = create_value_function(
            self._observation_shape, self._critic_encoder_factory)

    def _build_critic_optim(self) -> None:
        assert self._v_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._v_func.parameters(), lr=self._critic_learning_rate
        )

    @abstractmethod
    def _build_actor(self) -> None:
        pass

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        loss = self.compute_critic_loss(batch)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor: 
        assert self._v_func is not None
        return self._v_func.compute_error(batch.observations, batch.rewards)

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._v_func is not None
        assert self._actor_optim is not None
        assert self._critic_optim is not None

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @abstractmethod
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    def update_critic_target(self) -> None:
        assert self._v_func is not None
        assert self._old_v_func is not None
        hard_sync(self._old_v_func, self._v_func)

    def update_actor_target(self) -> None:
        assert self._policy is not None
        assert self._old_policy is not None
        hard_sync(self._old_policy, self._policy)

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._actor_optim
        return self._actor_optim

    @property
    def v_function(self) -> ValueFunction:
        assert self._v_func
        return self._v_func

    @property
    def v_function_optim(self) -> Optimizer:
        assert self._critic_optim
        return self._critic_optim


class PPOImpl(PPOBaseImpl):

    _policy: Optional[SquashedNormalPolicy]
    _old_policy: Optional[SquashedNormalPolicy]

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-6.0,
            max_logstd=0.0,
            use_std_parameter=True,
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._old_v_func is not None
        
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        dist_entropy = log_probs
        assert log_probs.shape == dist_entropy.shape

        old_dist = self._old_policy.dist(batch.observations)
        old_log_probs = old_dist.log_prob(batch.actions).detach()
        old_values = self._old_v_func(batch.observations)
        advantages = (batch.rewards - old_values).detach()  

        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self._eps_clip, 1 + self._eps_clip) * advantages

        loss = - torch.min(surr1, surr2) - self._policy_entropy_weight * dist_entropy
        return loss.mean()

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)
