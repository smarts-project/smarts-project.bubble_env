import numpy as np
from collections import deque, defaultdict
from typing import Dict, Tuple

from envision.client import Client as Envision
from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.local_traffic_provider import LocalTrafficProvider
from functools import partial

from .vehicle_info import VehiclePosition
from .social_utils import (
    SocialAgentMapping,
    SOCIAL_AGENT_TERMINAL_FEATURES,
)
from smarts.core.plan import (
    Mission,
    TraverseGoal,
    VehicleSpec,
    default_entry_tactic,
)
from smarts.sstudio.types import Bubble, MapZone, PositionalZone, SocialAgentActor
from smarts.core.agent_manager import AgentManager
from smarts.core.agent import Agent
from smarts.core.sensors import Observation
from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.zoo.agent_spec import AgentSpec


class ObservationState:
    def __init__(self) -> None:
        self.last_observations: Dict[str, Observation] = None

    def observation_callback(self, obs: Observation):
        self.last_observations = obs


def register_dummy_locator(interface, name="dummy_agent-v0"):
    class DummyAgent(Agent):
        """This is just a place holder that is used for the default agent used by the bubble."""

        def act(self, obs: Observation) -> Tuple[float, float]:
            acceleration = 0.0
            angular_velocity = 0.0
            return acceleration, angular_velocity

    # Register the dummy agent for use by the bubbles
    #  referenced as `"<module>:<locator>"` (i.e. `"examples:dummy_agent-v0"`)
    register(
        name,
        entry_point=lambda **kwargs: AgentSpec(
            interface=interface,
            agent_builder=DummyAgent,
        ),
    )


def create_moving_bubble(
    follow_vehicle_id=None, follow_agent_id=None, social_agent_name="dummy_agent-v0"
):
    assert follow_vehicle_id or follow_agent_id, "Must follow a vehicle or agent"
    assert not (
        follow_vehicle_id and follow_agent_id
    ), "Must follow only one of vehicle or agent"

    exclusion_prefixes = ()
    follow = {}
    if follow_vehicle_id:
        follow = dict(follow_vehicle_id=follow_vehicle_id)
        exclusion_prefixes = (follow_vehicle_id,)
    else:
        follow = dict(follow_actor_id=follow_agent_id)

    bubble = Bubble(
        zone=PositionalZone(pos=(0, 0), size=(10, 40)),
        actor=SocialAgentActor(
            name="keep_lane0", agent_locator=f"examples:{social_agent_name}"
        ),
        exclusion_prefixes=exclusion_prefixes,
        follow_offset=(0, 0),
        margin=5,
        **follow,
    )
    return bubble


class SMARTSBubbleEnv:
    def __init__(
        self,
        agent_interface,
        vehicles: list,
        scenario_path: str,
        social_agent_mapping_mode="default",  # social_agent_mapping_mode
        social_agent_interface_mode=False,  # define terminal mode
        control_vehicle_num: int = 1,
        collision_done: bool = True,
        envision: bool = False,
        envision_sim_name: str = None,
        envision_record_data_replay_path: str = None,
        headless: bool = False,
        **kwargs,
    ):
        """
        Args:
            vehicles: Dict[scenario_name: Dict[traffic_name: List[vehicle_info, ...]]]
                vehicle_info: v_info = namedtuple(
                        vehicle_id=vehicle_id,
                        start_time=None,
                        end_time=None,
                        scenario_name="xxx",
                        traffic_name="xxx",
                        ttc=None,
                    )
                    start_time (Optional): Specify the start_time of the vehicle (can not be
                        "first-seen-time"). Default None (use the first-seen-time).
                    end_time (Optional): Specify the end_time of the vehicle (can not be
                        "last-seen-time"). Default None (use the last-seen-time).
        """
        print("Now Create Bubble Env")

        assert control_vehicle_num == 1  # Bubble should only support 1 ego_vehicle
        self.collision_done = collision_done
        self.padding_vehicle = "0"  # imply vacant vehicle,use for neighbor replace only

        self.control_vehicle_num = self.n_agents = control_vehicle_num
        self.vehicles = vehicles
        if vehicles is None:
            print("Use All Vehicles")

        self.scenarios_list = list(
            Scenario.scenario_variations(
                [scenario_path], [], shuffle_scenarios=False, circular=False
            )
        )
        self._init_scenario()
        # Num of all combinations of different controlled vehicles used.
        self.episode_num = len(self.vehicle_ids) - self.control_vehicle_num + 1

        self.aid_to_vindex = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.default_agent_id = "agent_0"

        self.agent_interface = agent_interface

        self.social_agent_mapping_mode = social_agent_mapping_mode
        print(f"social mapping mode:{social_agent_mapping_mode}")

        envision_client = None
        if envision:
            envision_client = Envision(
                sim_name=envision_sim_name,
                output_dir=envision_record_data_replay_path,
                headless=headless,
            )

        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sims=[LocalTrafficProvider()],
            envision=envision_client,
            fixed_timestep_sec=0.1,
        )
        self.social_agent_interface_mode = social_agent_interface_mode
        self.social_interface = AgentInterface.from_type(
            AgentType.Direct,
            done_criteria=DoneCriteria(
                collision=self.social_agent_interface_mode,
                off_road=False,
                off_route=False,
                on_shoulder=False,
            ),
            waypoints=True,
        )
        register_dummy_locator(self.social_interface)

        self.obs_state = ObservationState()

        self.agent_manager: AgentManager = self.smarts.agent_manager
        self.agent_manager.add_social_agent_observations_callback(
            self.obs_state.observation_callback, "bubble_watcher"
        )

        self.legal_vehicle_info = {"vehicle_infos": []}

    def seed(self, seed):
        np.random.seed(seed)

    def check_social_agent_terminals(self, raw_observations):
        social_dones = {}
        for social_agent_id, social_agent_obs in raw_observations.items():
            # if not self.social_agent_interface_mode:
            #     social_dones[social_agent_id] = False
            #     continue
            done = False
            for feat in SOCIAL_AGENT_TERMINAL_FEATURES:
                if feat == "collisions":
                    if len(getattr(social_agent_obs.events, feat)) == 0 or isinstance(
                        social_agent_obs.events.collisions[0].collidee_id, type(None)
                    ):
                        continue
                    if (
                        social_agent_obs.events.collisions[0].collidee_id[:6]
                        == "BUBBLE"
                    ):
                        print(
                            f"social_done with:{social_agent_obs.events.collisions[0].collidee_id}"
                        )
                        done = True
                else:
                    if getattr(social_agent_obs.events, feat):
                        print(f"social_done with:{feat} True")
                    done = done | getattr(social_agent_obs.events, feat)
            social_dones[social_agent_id] = done
        return social_dones

    def step(self, action_n):
        ego_action_n = {}
        social_action_n = {}
        for agent_id in action_n.keys():
            if self.done_n[agent_id]:
                continue
            action = action_n[agent_id]
            if agent_id.startswith("agent"):
                ego_action_n[agent_id] = action
            else:
                social_action_n[agent_id] = action

        for social_agent_id in self.obs_state.last_observations:
            if social_agent_id in social_action_n:
                self.agent_manager.reserve_social_agent_action(
                    social_agent_id, social_action_n[social_agent_id]
                )
            else:
                # print(f"WARNING: NO ACTION ASSIGNED FOR SOCIAL VEHICLE: {social_agent_id}")
                self.agent_manager.reserve_social_agent_action(
                    social_agent_id, (0.0, 0.0)
                )

        raw_observation_n, reward_n, self.done_n, _ = self.smarts.step(ego_action_n)
        ego_obs_n = raw_observation_n
        social_dones = self.check_social_agent_terminals(
            self.obs_state.last_observations
        )
        full_obs_n = dict(raw_observation_n, **self.obs_state.last_observations)
        self.done_n = dict(self.done_n, **social_dones)

        info_n = {}

        info_n["social_agent_mapping"] = {}
        for (
            social_agent_id,
            social_agent_observation,
        ) in self.obs_state.last_observations.items():
            if social_agent_id not in self.used_history_ids:
                if social_agent_observation.ego_vehicle_state.lane_index == None:
                    self.done_n[social_agent_id] = True
                    continue
                info_n["social_agent_mapping"][
                    social_agent_id
                ] = SocialAgentMapping.mapping_index(
                    raw_observation_n[self.default_agent_id],
                    social_agent_observation,
                    self.social_agent_mapping_mode,
                )
                self.used_history_ids.add(social_agent_id)

        for agent_id in ego_obs_n.keys():
            vehicle_index = self.aid_to_vindex[agent_id]
            vehicle_id = self.vehicle_ids[vehicle_index]
            info_n[agent_id] = {}
            info_n[agent_id]["reached_goal"] = raw_observation_n[
                agent_id
            ].events.reached_goal
            info_n[agent_id]["collision"] = (
                len(raw_observation_n[agent_id].events.collisions) > 0
            )
            info_n[agent_id]["car_id"] = vehicle_id
            info_n[agent_id]["lane_index"] = raw_observation_n[
                agent_id
            ].ego_vehicle_state.lane_index
            raw_position = raw_observation_n[agent_id].ego_vehicle_state.position
            info_n[agent_id]["raw_position"] = raw_position

        if self.time_slice:
            """vehicle_end_times are specified by user."""
            for agent_id in ego_obs_n.keys():
                vehicle_index = self.aid_to_vindex[agent_id]
                if self.smarts.elapsed_sim_time > self.vehicle_end_times[vehicle_index]:
                    if not self.done_n[agent_id]:
                        info_n[agent_id]["over_max_time"] = True
                    self.done_n[agent_id] = True

        return (
            full_obs_n,
            reward_n,
            self.done_n,
            info_n,
        )

    def reset(self):
        self.raw_obs_queues = defaultdict(partial(deque, maxlen=2))
        self.agents_initial_lane_index = {}

        if self.episode_count == self.episode_num:
            self.episode_count = 0
            if self.control_vehicle_num > 1 and self.time_slice == False:
                self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
            else:
                self.vehicle_itr = 0
        if self.vehicle_itr + self.n_agents > len(self.vehicle_ids):
            self.vehicle_itr = 0

        self.active_vehicles_index = self.vehicles_index[
            self.vehicle_itr : self.vehicle_itr + self.n_agents
        ]
        self.aid_to_vindex = {
            f"agent_{i}": self.active_vehicles_index[i] for i in range(self.n_agents)
        }
        self._set_traffic(self.vehicles[self.active_vehicles_index[0]].traffic_name)

        agent_interfaces = {}
        # Find the earliest start time among all selected vehicles.
        history_start_time = np.inf
        for agent_id in self.agent_ids:
            vehicle_index = self.aid_to_vindex[agent_id]
            if self.vehicle_ids[vehicle_index] != self.padding_vehicle:
                # agent_interfaces[agent_id] = self.social_interface
                agent_interfaces[agent_id] = self.agent_interface
                history_start_time = min(
                    history_start_time, self.vehicle_start_times[vehicle_index]
                )

        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle_index = self.aid_to_vindex[agent_id]
            if self.vehicle_ids[vehicle_index] != self.padding_vehicle:
                assert self.vehicle_missions[vehicle_index] is not None
                ego_missions[agent_id] = self.vehicle_missions[vehicle_index]

        self.scenario.bubbles.clear()
        self.scenario.bubbles.extend(
            [create_moving_bubble(follow_agent_id=a_id) for a_id in ego_missions]
        )
        self.agent_vehicles = {
            mission.vehicle_spec.veh_id for mission in ego_missions.values()
        }
        self.used_history_ids = set(self.agent_vehicles)

        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)

        raw_observation_n = self.smarts.reset(self.scenario, history_start_time)
        for agent_id, raw_observation in raw_observation_n.items():
            self.agents_initial_lane_index[
                agent_id
            ] = raw_observation.ego_vehicle_state.lane_index
            if self.agents_initial_lane_index[agent_id] > 1:
                vehicle_index = self.aid_to_vindex[agent_id]
                self.legal_vehicle_info["vehicle_infos"].append(
                    {
                        "vehicle_id": self.vehicle_ids[vehicle_index],
                        "start_time": self.vehicle_start_times[vehicle_index],
                        "end_time": self.vehicle_end_times[vehicle_index],
                        "scenario_name": self.scenario_names[vehicle_index],
                        "traffic_name": self.traffic_names[vehicle_index],
                    }
                )

        self.done_n = {a_id: False for a_id in self.agent_ids}
        self.vehicle_itr += self.n_agents
        self.episode_count += 1
        return raw_observation_n

    def close(self):
        """Closes the environment and releases all resources."""
        self.destroy()

    def _set_traffic(self, traffic_name):
        self.scenario = None
        for scenario in self.scenarios_list:
            if scenario._traffic_history.name == traffic_name:
                self.scenario = scenario
                break
        assert self.scenario is not None

    def _init_scenario(self):
        self._init_vehicle_missions()
        """
        Prepare for multiple start/end_times for same vehicle
        Use list to store relative info instead of dict
        """
        self.vehicles_index = [id for id in range(len(self.vehicle_ids))]

        if self.control_vehicle_num == 1:
            np.random.shuffle(self.vehicles_index)
            self.vehicle_itr = 0
        elif not self.time_slice:
            # Sort vehicle id by starting time, so that we can get adjacent vehicles easily.
            self.vehicles_index = np.argsort([self.vehicle_start_times])
            self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
        else:
            self.vehicle_itr = 0
            print("Replace Neighbor with Time Slice")

        self.episode_count = 0

    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()
            self.smarts = None

    def _vehicle_pos_between(self, vehicle_id: str, start_time: float, end_time: float):
        """Find the vehicle states between the given history times."""
        query = """SELECT T.position_x, T.position_y, T.sim_time
                   FROM Trajectory AS T
                   WHERE T.vehicle_id = ? AND T.sim_time > ? AND T.sim_time <= ? 
                   ORDER BY T.sim_time DESC"""
        rows = self.scenario._traffic_history._query_list(
            query, (vehicle_id, start_time, end_time)
        )
        return (VehiclePosition(*row) for row in rows)

    def _discover_sliced_vehicle_missions(self):
        """Retrieves the missions of traffic history vehicles given the start_time of each vehicle.

        The function is used to get the vehicle_mission when the vehicle_start_times is
        "User-defined" (not the first-seen-times of the vehicles, is often used to define some
        special cases, such as cut-in, fail-cases, etc.), since Scenario.discover_missions_of_traffic_histories()
        can only get the missions at fist-seen-time of the vehicles.
        """
        vehicle_missions = []
        for v in self.vehicles:
            self._set_traffic(v.traffic_name)
            vehicle_id = v.vehicle_id
            if vehicle_id == self.padding_vehicle:
                vehicle_missions.append(None)
                continue
            start_time = v.start_time
            rows = self._vehicle_pos_between(
                vehicle_id,
                start_time=start_time - 0.1,
                end_time=start_time,
            )
            rows = list(rows)
            assert len(rows) > 0, rows
            sim_time = rows[0].sim_time
            start, speed = self.scenario.get_vehicle_start_at_time(vehicle_id, sim_time)
            entry_tactic = default_entry_tactic(speed)
            veh_config_type = self.scenario._traffic_history.vehicle_config_type(
                vehicle_id
            )
            veh_dims = self.scenario._traffic_history.vehicle_dims(vehicle_id)
            vehicle_missions.append(
                Mission(
                    start=start,
                    entry_tactic=entry_tactic,
                    goal=TraverseGoal(self.scenario.road_map),
                    start_time=sim_time,
                    vehicle_spec=VehicleSpec(
                        veh_id=vehicle_id,
                        veh_config_type=veh_config_type,
                        dimensions=veh_dims,
                    ),
                )
            )

        return vehicle_missions

    def _init_vehicle_missions(self):
        """
        Get the initial conditions of controlled vehicles (position, speed, heading, etc.)
        """

        if self.vehicles[0].start_time is not None:
            """Use the "start_times" and "end_times" of the vehicles specified by user."""
            self.time_slice = True
            self.scenario_names = [v.scenario_name for v in self.vehicles]
            self.traffic_names = [v.traffic_name for v in self.vehicles]
            self.vehicle_ids = [v.vehicle_id for v in self.vehicles]
            self.vehicle_start_times = [v.start_time for v in self.vehicles]
            self.vehicle_end_times = [v.end_time for v in self.vehicles]
            self.vehicle_missions = self._discover_sliced_vehicle_missions()
        else:
            raise NotImplementedError
