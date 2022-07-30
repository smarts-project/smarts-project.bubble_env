from typing import Any, Dict
import gym


def entry_point(traffic_mode="traffic_A", action_space="Direct", **kwargs):
    from bubble_env_contrib.bubble_env.envs.bubble_traffic_gym_env import (
        BubbleTrafficGymEnv,
    )
    from bubble_env_contrib.bubble_env.envs import get_bubble_env
    from smarts.env.multi_scenario_v0_env import resolve_agent_interface

    agent_interface = resolve_agent_interface(action_space=action_space, **kwargs)

    return BubbleTrafficGymEnv(
        get_bubble_env(agent_interface, traffic_mode=traffic_mode)
    )


gym.register("bubble_env-v0", entry_point=entry_point)
