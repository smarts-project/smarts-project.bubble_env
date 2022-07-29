import os
import sys
import inspect
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from bubble_env_contrib.bubble_env.envs import get_bubble_env
import numpy as np

print(sys.path)


class EgoVehicle:
    def act(self, obs):
        return 0.0, 0.0


if __name__ == "__main__":
    env = get_bubble_env(traffic_mode="traffic_A")
    ego_policy = EgoVehicle()

    episode_num = 20
    for episode in range(episode_num):
        ego_obs = env.reset()
        steps = 0
        while True:
            steps += 1
            ego_action = {}
            for agent_id, obs in ego_obs.items():
                ego_action[agent_id] = ego_policy.act(obs)
            ego_obs, _, done_n, _ = env.step(ego_action)
            if np.any(list(done_n.values())):
                break

        print(f"episode {episode} finished! steps: {steps}")

    print("all finished")
