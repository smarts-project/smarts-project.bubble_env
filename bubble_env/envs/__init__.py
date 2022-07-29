from typing import Any, Dict
from .core import SMARTSBubbleEnv
from .bubble_traffic_gym_env import BubbleTrafficGymEnv
from .env_wrapper import SocialAgentsWrapper
import numpy as np
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from .vehicle_info import VehicleInfo
import json
import os
import gym

from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    NeighborhoodVehicles,
    RoadWaypoints,
    Waypoints,
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


def _resolve_agent_interface(
    img_meters: int = 64, img_pixels: int = 256, action_space="TargetPose", **kwargs
):
    done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=False,
        on_shoulder=False,
        wrong_way=False,
        not_moving=False,
        agents_alive=None,
    )
    max_episode_steps = 800
    neighbor_radius = 50
    road_waypoint_horizon = 50
    waypoints_lookahead = 50
    return AgentInterface(
        accelerometer=True,
        action=ActionSpaceType[action_space],
        done_criteria=done_criteria,
        drivable_area_grid_map=DrivableAreaGridMap(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        lidar=True,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicles=NeighborhoodVehicles(neighbor_radius),
        ogm=OGM(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        rgb=RGB(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
        waypoints=Waypoints(lookahead=waypoints_lookahead),
    )


def entry_point(
    config: Dict[str, Any] = dict(traffic_mode="traffic_A", action_space="Direct")
):
    try:
        from smarts.env.multi_scenario_v0 import resolve_agent_interface

        agent_interface = resolve_agent_interface(**config)
    except:
        agent_interface = _resolve_agent_interface(**config)

    return BubbleTrafficGymEnv(
        get_bubble_env(agent_interface, traffic_mode=config["traffic_mode"])
    )


gym.register("bubble_traffic-v0", entry_point=entry_point)
