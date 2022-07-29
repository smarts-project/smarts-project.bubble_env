from typing import Any, Dict
import gym


class BubbleTrafficGymEnv(gym.Env):
    def __init__(self, env) -> None:
        self._env = env

    def seed(self, **kwargs: Dict[str, Any]):
        self._env.seed(**kwargs)

    def step(self, action: Any):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()
