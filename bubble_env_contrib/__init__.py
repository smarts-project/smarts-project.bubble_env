from typing import Any, Dict
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

import gym

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
    from bubble_env_contrib.bubble_env.envs.bubble_traffic_gym_env import BubbleTrafficGymEnv
    from bubble_env_contrib.bubble_env.envs import get_bubble_env
    try:
        from smarts.env.multi_scenario_v0 import resolve_agent_interface

        agent_interface = resolve_agent_interface(**config)
    except:
        agent_interface = _resolve_agent_interface(**config)

    return BubbleTrafficGymEnv(
        get_bubble_env(agent_interface, traffic_mode=config["traffic_mode"])
    )
    
gym.register("bubble_env-v0", entry_point=entry_point)