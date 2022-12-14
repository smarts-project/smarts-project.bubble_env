#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
from collections import deque
import numpy as np
import paddle
import gym
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from storage import RolloutStorage
from parl.algorithms import PPO
from parl.utils import summary
import argparse

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
import bubble_env.envs

LR = 3e-4
GAMMA = 0.99
EPS = 1e-5  # Adam optimizer epsilon (default: 1e-5)
GAE_LAMBDA = 0.95  # Lambda parameter for calculating N-step advantage
ENTROPY_COEF = 0.0  # Entropy coefficient (ie. c_2 in the paper)
VALUE_LOSS_COEF = 0.5  # Value loss coefficient (ie. c_1 in the paper)
MAX_GRAD_NROM = 0.5  # Max gradient norm for gradient clipping
NUM_STEPS = 64  # data collecting time steps (ie. T in the paper)
PPO_EPOCH = 10  # number of epochs for updating using each T data (ie K in the paper)
CLIP_PARAM = 0.2  # epsilon in clipping loss (ie. clip(r_t, 1 - epsilon, 1 + epsilon))
BATCH_SIZE = 8

# Logging Params
LOG_INTERVAL = 1


def evaluate(agent, env_id):
    eval_env = gym.make("bubble_env_contrib:bubble_env-v0")
    # eval_env = wrap_rms(eval_env, GAMMA, test=True, ob_rms=ob_rms)
    eval_episode_rewards = []
    obs = eval_env.reset()
    obs = eval_env._convert_obs(obs)["agent_0"]

    while len(eval_episode_rewards) < 3:
        action = agent.predict(obs)

        # Observe reward and next obs
        obs, _, done, info = eval_env.step({"agent_0": (action[0], action[1])})
        obs = eval_env._convert_obs(obs)["agent_0"]
        done = done["agent_0"]
        # get validation rewards from info['episode']['r']
        if done:
            eval_episode_rewards.append(0)
            obs = eval_env.reset()
            obs = eval_env._convert_obs(obs)["agent_0"]

    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)
        )
    )
    return np.mean(eval_episode_rewards)


def main(env_id):
    paddle.seed(args.seed)
    env = gym.make("bubble_env_contrib:bubble_env-v0")

    model = MujocoModel(56, 2)

    algorithm = PPO(
        model, CLIP_PARAM, VALUE_LOSS_COEF, ENTROPY_COEF, LR, EPS, MAX_GRAD_NROM
    )

    agent = MujocoAgent(algorithm)

    rollouts = RolloutStorage(NUM_STEPS, 56, 2)

    obs = env.reset()
    obs = env._convert_obs(obs)["agent_0"]
    rollouts.obs[0] = np.copy(obs)

    episode_rewards = deque(maxlen=10)

    num_updates = int(args.train_total_steps) // NUM_STEPS
    for j in range(num_updates):
        for step in range(NUM_STEPS):
            # Sample actions
            value, action, action_log_prob = agent.sample(rollouts.obs[step])

            # Observe reward and next obs
            obs, reward, done, info = env.step(
                {"agent_0": (action[0, 0], action[0, 1])}
            )
            obs = env._convert_obs(obs)["agent_0"]
            reward = reward["agent_0"]
            done = done["agent_0"]
            info = info["agent_0"]
            # get training rewards from info['episode']['r']
            if done:
                episode_rewards.append(0)
                obs = env.reset()
                obs = env._convert_obs(obs)["agent_0"]

            # If done then clean the history of observations.
            masks = paddle.to_tensor([[0.0]] if done else [[1.0]], dtype="float32")
            bad_masks = paddle.to_tensor(
                [[0.0]] if "bad_transition" in info.keys() else [[1.0]], dtype="float32"
            )
            rollouts.append(
                obs, action, action_log_prob, value, reward, masks, bad_masks
            )

            print(f"finish steps:{step}")

        next_value = agent.value(rollouts.obs[-1])

        value_loss, action_loss, dist_entropy = agent.learn(
            next_value, GAMMA, GAE_LAMBDA, PPO_EPOCH, BATCH_SIZE, rollouts
        )

        rollouts.after_update()

        if j % LOG_INTERVAL == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * NUM_STEPS
            print(
                "Updates {}, num timesteps {},\n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

        if (
            args.test_every_steps is not None
            and len(episode_rewards) > 1
            and j % args.test_every_steps == 0
        ):
            eval_mean_reward = evaluate(agent, env_id)

            summary.add_scalar(
                "ppo/mean_validation_rewards", eval_mean_reward, (j + 1) * NUM_STEPS
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--seed", type=int, default=616, help="random seed (default: 616)"
    )
    parser.add_argument(
        "--test_every_steps", type=int, default=10, help="eval interval (default: 10)"
    )
    parser.add_argument(
        "--train_total_steps",
        type=int,
        default=10e5,
        help="number of total time steps to train (default: 10e5)",
    )
    parser.add_argument(
        "--env",
        default="bubble_env-v0",
        help="environment to train on (default: bubble_env-v0)",
    )
    args = parser.parse_args()

    main(env_id=args.env)
