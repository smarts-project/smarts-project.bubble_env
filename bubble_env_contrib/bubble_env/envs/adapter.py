from .common import get_obs_adapter, subscribe_features
import gym.spaces


def get_observation_adapter(feature_list, closest_neighbor_num, neighbor_max_distance):
    img_resolution = 40
    observe_lane_num = 3

    observation_space = gym.spaces.Dict(
        subscribe_features(feature_list, closest_neighbor_num=closest_neighbor_num)
    )
    observation_adapter = get_obs_adapter(
        observation_space,
        observe_lane_num=observe_lane_num,
        resize=(img_resolution, img_resolution),
        closest_neighbor_num=closest_neighbor_num,
        neighbor_max_distance=neighbor_max_distance,
    )
    return observation_adapter


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return model_action[0], model_action[1]

    return action_adapter
