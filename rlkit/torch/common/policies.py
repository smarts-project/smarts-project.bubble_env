import math
import torch
from torch import nn as nn

from rlkit.policies.base import ExplorationPolicy, Policy
from rlkit.torch.common.networks import Mlp
from rlkit.torch.common.distributions import ReparamTanhMultivariateNormal
from rlkit.torch.common.distributions import ReparamMultivariateNormalDiag

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ReparamTanhMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = ReparamTanhMultivariateGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        init_w=1e-3,
        max_act=1.0,
        conditioned_std: bool = False,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.max_act = max_act
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    @torch.jit.ignore
    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    @torch.jit.ignore
    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    @torch.jit.ignore
    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_tanh_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """

        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))

        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)

            if return_log_prob:
                action, pre_tanh_value = tanh_normal.sample(return_pretanh_value=True)
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
            else:
                action = tanh_normal.sample()

        # XXX: doing it like this for now for backwards compatibility
        if return_tanh_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                pre_tanh_value,
                tanh_normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
            pre_tanh_value,
        )

    @torch.jit.export
    def jit_forward(self, obs):
        """torch.jit does not support condition control
        :param obs: Observation
        """
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        action = torch.tanh(mean)
        return action

    @torch.jit.ignore
    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )

        return log_prob, entropy

    @torch.jit.ignore
    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        tanh_normal = ReparamTanhMultivariateNormal(mean, log_std)
        log_prob = tanh_normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob


class ReparamMultivariateGaussianPolicy(Mlp, ExplorationPolicy):
    def __init__(
        self,
        hidden_sizes,
        obs_dim,
        action_dim,
        conditioned_std=False,
        init_w=1e-3,
        **kwargs,
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs,
        )
        self.conditioned_std = conditioned_std

        if self.conditioned_std:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.last_fc.weight.data.mul_(0.1)
        self.last_fc.bias.data.mul_(0.0)

    @torch.jit.ignore
    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    @torch.jit.ignore
    def get_actions(self, obs_np, deterministic=False):
        return self.eval_np(obs_np, deterministic=deterministic)[0]

    @torch.jit.ignore
    def forward(
        self, obs, deterministic=False, return_log_prob=False, return_normal=False
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        if deterministic:
            action = mean
        else:
            normal = ReparamMultivariateNormalDiag(mean, log_std)
            action = normal.sample()
            if return_log_prob:
                log_prob = normal.log_prob(action)

        # XXX: Doing it like this for now for backwards compatibility
        if return_normal:
            return (
                action,
                mean,
                log_std,
                log_prob,
                expected_log_prob,
                std,
                mean_action_log_prob,
                normal,
            )
        return (
            action,
            mean,
            log_std,
            log_prob,
            expected_log_prob,
            std,
            mean_action_log_prob,
        )

    @torch.jit.export
    def jit_forward(self, obs):
        """
        :param obs: Observation
        """
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        action = self.last_fc(h)
        return action

    @torch.jit.ignore
    def get_log_prob_entropy(self, obs, acts):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(
            dim=-1, keepdim=True
        )
        return log_prob, entropy

    @torch.jit.ignore
    def get_log_prob(self, obs, acts, return_normal_params=False):
        h = obs
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)

        if self.conditioned_std:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        else:
            log_std = self.action_log_std.expand_as(mean)

        normal = ReparamMultivariateNormalDiag(mean, log_std)
        log_prob = normal.log_prob(acts)

        if return_normal_params:
            return log_prob, mean, log_std
        return log_prob
