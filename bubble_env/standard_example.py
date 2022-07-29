import inspect
import os
import sys
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import envs


class EgoVehicle:
    def act(self, obs):
        return 0.0, 0.0


def main():
    env = envs.entry_point(config=dict(traffic_mode="traffic_A", action_space="Direct"))
    ego_policy = EgoVehicle()

    episode_num = 20
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
