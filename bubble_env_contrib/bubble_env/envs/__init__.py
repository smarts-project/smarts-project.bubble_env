from typing import Any, Dict
from .core import SMARTSBubbleEnv
from .env_wrapper import SocialAgentsWrapper
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from .vehicle_info import VehicleInfo
import json
import os

from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
)


class SocialAgentPolicy:
    def __call__(self, observation):
        return 0.0, 0.0


def get_bubble_env(agent_interface=None, traffic_mode="default"):
    if agent_interface is None:
        agent_interface = AgentInterface.from_type(
            AgentType.Direct,
            done_criteria=DoneCriteria(
                collision=False,
                off_road=True,
                off_route=False,
                on_shoulder=False,
            ),
            waypoints=True,  # Only for RL training
        )
    vehicles = []
    file_directory = os.path.dirname(__file__)
    root_path = os.path.abspath(os.path.dirname(file_directory))
    with open(
        os.path.join(root_path, "models/list_true_legal_vehicle_infos.json"), "r"
    ) as f:
        all_vehicle_infos = json.load(f)

    for json_info in all_vehicle_infos:
        vehicle = VehicleInfo(
            vehicle_id=json_info["vehicle_id"],
            start_time=json_info["start_time"],
            end_time=json_info["start_time"] + 15,
            scenario_name=json_info["scenario_name"],
            traffic_name=json_info["traffic_name"],
        )
        vehicles.append(vehicle)
    core_env = SMARTSBubbleEnv(
        agent_interface=agent_interface,
        vehicles=vehicles,
        scenario_path=os.path.join(root_path, "scenarios/"),
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
