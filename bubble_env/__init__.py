from typing import Any, Dict
import gym

from smarts.core.agent_interface import AgentInterface
from smarts.env.multi_scenario_v0 import resolve_agent_interface

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

def entry_point(config: Dict[str, Any]):
    agent_interface = resolve_agent_interface(**config)
    return BubbleTrafficEnv(agent_interface=agent_interface)

gym.register("bubble_traffic-v0", entry_point=entry_point)