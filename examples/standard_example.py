import inspect
import os
import sys
import numpy as np

import bubble_env_contrib.bubble_env.envs as envs
import gym


class EgoVehicle:
    def act(self, obs):
        return 0.0, 0.0


def main(episode_num=20):
    env = gym.make(
        "bubble_env_contrib:bubble_env-v0",
        traffic_mode="traffic_A",
        action_space="Direct",
        img_meters=50,
        img_pixels=200,
    )
    env.seed(24)

    ego_policy = EgoVehicle()
    for episode in range(episode_num):
        ego_obs = env.reset()
        steps = 0
        while True:
            ego_action = {}
            for agent_id, obs in ego_obs.items():
                ego_action[agent_id] = ego_policy.act(obs)
            ego_obs, _, done_n, _ = env.step(ego_action)
            steps += 1
            if np.any(list(done_n.values())):
                break

        print(f"episode {episode} finished! steps: {steps}")

    print("all finished")


if __name__ == "__main__":
    main()
