import sys, os
import time
import numpy as np
import random
import traceback
import argparse
import gym
sys.path.append(os.pardir)
import gym_sfm.envs.env as envs

import torch
import torch.optim as optim

from models.ddpg import *

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='d_kan')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=1800)
parser.add_argument('-mepi', '--max_episodes', type=int, default=300)
args = parser.parse_args()

env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

max_episodes = 300
memory_capacity = 1e6  # バッファの容量
gamma = 0.99  # 割引率
tau = 1e-3  # ターゲットの更新率
epsilon = 1.0  # ノイズの量をいじりたい場合、多分いらない
batch_size = 32
lr_actor = 1e-4
lr_critic = 1e-3
logger_interval = 10
weight_decay = 1e-2

num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = args.max_t

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actorNet = ActorNetwork(num_state, num_action).to(device)
criticNet = CriticNetwork(num_state, num_action).to(device)
optimizer_actor = optim.Adam(actorNet.parameters(), lr=lr_actor)
optimizer_critic = optim.Adam(criticNet.parameters(), lr=lr_critic, weight_decay=weight_decay)
replay_buffer = ReplayBuffer(capacity=memory_capacity)
agent = DDPG(actorNet, criticNet, optimizer_actor, optimizer_critic, replay_buffer, device, gamma, tau, epsilon, batch_size)

for i_episode in range(args.max_episodes):
    observation = env.reset()
    done = False
    epidode_reward_sum = 0
    total_reward = 0
    # start = time.time()
    for t in range(args.max_t):
        # action = np.array([0, 0], dtype=np.float64)
        action = agent.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        agent.add_memory(observation, action, next_observation, reward, done)

        agent.train()

        observatin = next_observation

        env.render()
    env.close()
    print("total reward: {}".format(total_reward))
print('Finished all episode.')
