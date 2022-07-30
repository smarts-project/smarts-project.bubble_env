import inspect
import os
import numpy as np
import torch
from typing import Tuple, List, Dict

import rlkit
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy

# SOCIAL_AGENT_TERMINAL_FEATURES = ["collisions", "off_road", "off_route", "on_shoulder"]
SOCIAL_AGENT_TERMINAL_FEATURES = ["collisions", "off_road", "off_route"]
EPS = np.finfo(np.float32).eps.item()

rlkit_dir = os.path.dirname(rlkit.__file__)
non_cutin_model_path = f"{rlkit_dir}/model/non-cutin_model"
left_model_path = f"{rlkit_dir}/model/leftmodel"
right_model_path = f"{rlkit_dir}/model/rightmodel"


def split_list_dict_to_ego_and_soical(raw_list_dict: List[Dict]):
    ego_list_dict = []
    social_list_dict = []
    for raw_dict in raw_list_dict:
        ego_dict, social_dict = split_dict_to_ego_and_social(raw_dict)
        ego_list_dict.append(ego_dict)
        social_list_dict.append(social_dict)

    return ego_list_dict, social_list_dict


def split_dict_to_ego_and_social(raw_dict: Dict):
    ego_dict = {}
    social_dict = {}
    for agent_id, agent_value in raw_dict.items():
        if agent_id[:5] == "agent":
            ego_dict[agent_id] = agent_value
        else:
            social_dict[agent_id] = agent_value
    return ego_dict, social_dict


class SocialAgent(object):
    """This is just a place holder for the actual agent."""

    def __init__(
        self,
        model_path,
    ) -> None:
        self.policy = ReparamTanhMultivariateGaussianPolicy(
            hidden_sizes=[256] * 3,
            obs_dim=56,
            action_dim=2,
        )
        self.policy.load_state_dict(torch.load(model_path + "/model_parameter.pkl"))
        self.obs_mean = np.load(model_path + "/mean.npy")
        self.obs_std = np.load(model_path + "/std.npy")

    def norm_obs(self, obs):
        return (obs - self.obs_mean) / (self.obs_std + EPS)

    def act(self, observation) -> Tuple[float, float]:
        full_obs = self.norm_obs(observation)
        action = self.policy.get_action(full_obs, deterministic=True)[0]
        return (action[0], action[1])


class SocialAgentInitialization:
    @staticmethod
    def initial_agents(mapping_mode):
        func = getattr(SocialAgentInitialization, mapping_mode)
        return func()

    @staticmethod
    def default():
        return [SocialAgent(model_path=non_cutin_model_path)]

    @staticmethod
    def traffic_A():
        return [
            SocialAgent(model_path=non_cutin_model_path),
            SocialAgent(model_path=left_model_path),
        ]


class SocialAgentMapping:
    @staticmethod
    def mapping_index(ego_obs, vehicle_obs, mapping_mode):
        func = getattr(SocialAgentMapping, mapping_mode)
        return func(ego_obs, vehicle_obs)

    @staticmethod
    def default(ego_obs, vehicle_obs):
        return 0

    @staticmethod
    def traffic_A(ego_obs, vehicle_obs):
        ego_lane_index = ego_obs.ego_vehicle_state.lane_index
        vehicle_lane_index = vehicle_obs.ego_vehicle_state.lane_index

        return 0 if vehicle_lane_index >= ego_lane_index else 1
