from .core import SMARTSBubbleEnv
from .env_wrapper import SocialAgentsWrapper
import numpy as np
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from .vehicle_info import VehicleInfo
import json
import os


class SocialAgentPolicy:
    def __call__(self, observation):
        return 0.0, 0.0


def get_bubble_env(agent_interface=None, traffic_mode="default"):
    if agent_interface is None:
        agent_interface = AgentInterface.from_type(
            AgentType.Direct,
            done_criteria=DoneCriteria(),
        )
    vehicles = []
    current_path = os.path.dirname(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    with open(
        os.path.join(father_path, "models/true_legal_vehicle_infos.json"), "r"
    ) as f:
        all_vehicle_infos = json.load(f)
    print("vehicle_num:{}".format(len(all_vehicle_infos)))
    for json_info in all_vehicle_infos:
        vehicle = VehicleInfo(
            vehicle_id=json_info["vehicle_id"],
            start_time=json_info["start_time"],
            end_time=json_info["end_time"],
            scenario_name=json_info["scenario_name"],
            traffic_name=json_info["traffic_name"],
        )
        vehicles.append(vehicle)
    core_env = SMARTSBubbleEnv(
        agent_interface=agent_interface,
        vehicles=vehicles,
        scenario_path=os.path.join(father_path, "scenarios/"),
        social_agent_mapping_mode=traffic_mode,
        social_agent_interface_mode=False,
        collision_done=False,
        envision=True,
        control_vehicle_num=1,
    )
    bubble_env = SocialAgentsWrapper(
        env=core_env,
        action_range=np.array(
            [
                [-3.0, -2.0],
                [3.0, 2.0],
            ]
        ),
    )
    return bubble_env
