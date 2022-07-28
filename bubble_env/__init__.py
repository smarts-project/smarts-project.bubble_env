from typing import Any, Dict
import gym

from envs import get_bubble_env

class BubbleTrafficEnv(gym.Env):
    def __init__(self, agent_interface, traffic_mode="default") -> None:
        self._env = get_bubble_env(agent_interface, traffic_mode)

    def seed(self, **kwargs: Dict[str, Any]):
        self._env.seed(**kwargs)

    def step(self, action: Any):
        return self._env.step(action)

    def reset(self, observation: Any):
        return self._env.reset(observation)

    def close(self):
        self._env.close()