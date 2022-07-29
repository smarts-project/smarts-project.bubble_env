import os
import sys
import inspect

# import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from bubble_env_contrib.bubble_env.envs import get_bubble_env
import numpy as np

print(sys.path)


class EgoVehicle:
    def act(self, obs):
        return 0.0, 0.0


json_save_path = f"{os.path.abspath(__file__)}/models/legal_vehicle_infos.json"
scenario_search_path = (
    f"{os.path.abspath(__file__)}/models/true_legal_vehicle_infos.json"
)
search_scenario = []

if __name__ == "__main__":
    env = get_bubble_env(traffic_mode="traffic_A")
    ego_policy = EgoVehicle()

    episode_num = 882
    for episode in range(episode_num):
        ego_obs = env.reset()
        if episode % 10 == 0:
            print(f"episode:{episode} finished")
        steps = 0
        while True:
            steps += 1
            ego_action = {}
            for agent_id, obs in ego_obs.items():
                ego_action[agent_id] = ego_policy.act(obs)
            ego_obs, _, done_n, _ = env.step(ego_action)
            if np.any(list(done_n.values())):
                break

        if steps >= 150:
            search_scenario.append(env._env.legal_vehicle_info["vehicle_infos"][-1])
            print(
                f"legal num:{len(search_scenario)} legal vehicle:{search_scenario[-1]}"
            )

            # with open(scenario_search_path, "w") as f:
            #     json.dump(search_scenario, f)

        print(f"episode {episode} finished! steps: {steps}")
    # with open(json_save_path, "w") as f:
    #     json.dump(env._env.legal_vehicle_info, f)

    print("all finished")
