import numpy as np
from .feature_group import FeatureGroup
from .agent import get_agent_spec
from .social_utils import SocialAgentInitialization


class SocialAgentsWrapper:
    def __init__(
        self,
        env,
        action_range: np.ndarray,
        feature_type: str = "egolane",
        closest_neighbor_num: int = 6,
        neighbor_max_distance: float = 80.0,
    ):
        self._env = env
        self.social_agent_policys = SocialAgentInitialization.initial_agents(
            self._env.social_agent_mapping_mode
        )
        self._action_range = action_range
        assert (
            action_range.shape == (2, 2) and (action_range[1] >= action_range[0]).all()
        ), action_range
        self.feature_type = feature_type
        self.closest_neighbor_num = closest_neighbor_num
        self.neighbor_max_distance = neighbor_max_distance

        self.feature_list = FeatureGroup[feature_type]
        self.agent_spec = get_agent_spec(
            self.feature_list,
            closest_neighbor_num,
            neighbor_max_distance,
        )  # agent_spec for social agents
        self.last_social_agent_observation_n = {}
        self.social_agent_mapping = {}

    def _convert_obs(self, raw_observations):
        full_obs_n = {}
        for agent_id in raw_observations.keys():
            all_states = []
            observation = self.agent_spec.observation_adapter(
                raw_observations[agent_id]
            )
            for feat in self.feature_list:
                all_states.append(observation[feat])
            full_obs = np.concatenate(all_states, axis=-1).reshape(-1)
            full_obs_n[agent_id] = full_obs
        return full_obs_n

    def action_mapping(self, action):
        scaled_action = np.clip(action, -1, 1)
        # Transform the normalized action back to the original range
        # *** Formula for transformation from x in [xmin, xmax] to [ymin, ymax]
        # *** y = (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin
        scaled_action = (self._action_range[1] - self._action_range[0]) * (
            scaled_action + 1
        ) / 2 + self._action_range[0]
        return (scaled_action[0], scaled_action[1])

    def reset(self):
        self.last_social_agent_observation_n = {}
        self.social_agent_mapping = {}
        return self._env.reset()

    def step(self, action_n):
        full_action_n = action_n
        social_last_observation_n = self._convert_obs(
            self.last_social_agent_observation_n
        )
        for social_agent_id, obs in social_last_observation_n.items():
            social_agent_policy = self.social_agent_policys[
                self.social_agent_mapping[social_agent_id]
            ]
            action = social_agent_policy.act(obs)
            scaled_action = self.action_mapping(action)
            assert social_agent_id not in full_action_n
            full_action_n[social_agent_id] = scaled_action
        full_obs_n, reward_n, full_done_n, info_n = self._env.step(full_action_n)
        self.last_social_agent_observation_n = {}
        ego_obs_n = {}
        ego_done_n = {}
        for agent_id, raw_obs in full_obs_n.items():
            if agent_id[:5] == "agent":
                ego_obs_n[agent_id] = raw_obs
                ego_done_n[agent_id] = full_done_n[agent_id]
            else:
                if not full_done_n[agent_id]:
                    self.last_social_agent_observation_n[agent_id] = raw_obs

        for social_agent_id, mapping_index in info_n["social_agent_mapping"].items():
            self.social_agent_mapping[social_agent_id] = mapping_index
        info_n.pop("social_agent_mapping")

        return ego_obs_n, reward_n, ego_done_n, info_n

    def seed(self, **kwargs):
        return self._env(**kwargs)

    def close(self):
        self._env.close()
