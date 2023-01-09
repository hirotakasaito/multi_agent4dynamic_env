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

from models.ppo_lstm import *

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='d_kan')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=1800)
parser.add_argument('-mepi', '--max_episodes', type=int, default=300)
parser.add_argument('--agent-num', type=int, default=1)
parser.add_argument('--action-std', type=float, default=0.6)
parser.add_argument('--action-std-decay-rate', type=float, default=0.05)
parser.add_argument('--min-action-std', type=float, default=0.1)
parser.add_argument('--action-std-decay-freq', type=int, default=2.5e5)
parser.add_argument('--update-timestep', type=float, default=1800*1)
parser.add_argument('--k-epochs', type=int, default=2)
parser.add_argument('--eps-clip', type=int, default=0.2)
parser.add_argument('--gamma', type=int, default=0.99)
parser.add_argument('--lr-actor', type=int, default=3e-4)
parser.add_argument('--lr-critic', type=int, default=1e-3)
parser.add_argument('--random-seed', type=int, default=0)
args = parser.parse_args()

env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")

time_step = 0
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.k_epochs, args.eps_clip, device, args.action_std)

for i_episode in range(args.max_episodes):
    observation = env.reset()
    done = False
    epidode_reward_sum = 0
    total_reward = 0

    for t in range(args.max_t):
        actions = []
        if args.agent_num > 1:
            print("multi agent")
        else:
            action = agent.get_action(observation[0])
            actions.append(action)
        next_observation, reward, done, _ = env.step(actions)
        total_reward += reward
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)

        time_step += 1

        if time_step % args.update_timestep == 0:
            print("update")
            agent.update()

        if time_step % args.action_std_decay_freq == 0:
            agent.decay_action_std(action_action_std_decay_rate, min_action_std)

        observatin = next_observation

        env.render()
    env.close()
    print("total reward: {}".format(total_reward))
