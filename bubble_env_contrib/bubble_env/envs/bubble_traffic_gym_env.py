from typing import Any, Dict
import gym


class BubbleTrafficGymEnv(gym.Env):
    def __init__(self, env) -> None:
        self._env = env

    def seed(self, *args, **kwargs):
        self._env.seed(*args, **kwargs)

    def step(self, action: Any):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        """Forward everything else to the non-standard environment."""
        attribute = getattr(self._env, name)
        return attribute
