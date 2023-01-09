import sys, os
import time
import numpy as np
import random
import traceback
import argparse
import gym
import gym_sfm.envs.env as envs

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='d_kan')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=1800)
parser.add_argument('-mepi', '--max_episodes', type=int, default=5)
args = parser.parse_args()

env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

for i_episode in range(args.max_episodes):
    observation = env.reset()
    # env.agent.pose = np.array([0.0, 0.0])
    done = False
    epidode_reward_sum = 0
    actions = np.random.rand(2,2)
    for t in range(args.max_t):
        scans, _, _,agents,yaw ,_ =self.env.step(actions)
        env.render()
    env.close()
print('Finished all episode.')
