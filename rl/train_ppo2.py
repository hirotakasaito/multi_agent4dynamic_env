import sys, os
import time
import numpy as np
import random
import traceback
import argparse
import gym
import datetime
import json

sys.path.append(os.pardir)
import gym_sfm.envs.env as envs
import torch
import torch.optim as optim
from torch import nn
import pfrl
from torch.utils.tensorboard import SummaryWriter
from pfrl import utils
from pfrl.agents import PPO
from tqdm import tqdm
from collections import OrderedDict

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', help='Specify map setting folder.', default='midas_env1')
    parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1000)
    parser.add_argument('-mt', '--max_t', type=int, default=500)
    parser.add_argument('-mepi', '--max_episodes', type=int, default=10000)
    parser.add_argument('--agent-num', type=int, default=1)
    parser.add_argument('--update-interval', type=float, default=500*10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--eps-clip', type=int, default=0.2)
    parser.add_argument('--gamma', type=int, default=0.995)
    parser.add_argument('--lr', type=int, default=1e-5)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--render-interval', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--dynamic-env', type=bool, default=False)
    parser.add_argument('--midas-env', type=bool, default=True)
    parser.add_argument('--can-obs-people', type=int, default=2)

    parser.add_argument('--save-dir', type=str, default="/share/private/27th/hirotaka_saito/logs/ppo_lstm_midas")
    args = parser.parse_args()
    return args

def make_save_dir(save_dir):
    dt_now = datetime.datetime.now()
    save_dir = os.path.join(save_dir, dt_now.strftime('%Y%m%d%H%M%S'))
    os.makedirs(os.path.join(save_dir, "weights"))
    return save_dir

def min_pooling(obs, k):
    n_obs = np.empty(0)

    for idx, v in enumerate(obs):
        d_obs = []
        d_obs.append(v)
        if (idx+1) % k == 0:
            min_v = min(d_obs)
            n_obs = np.append(n_obs, min_v)
    return n_obs

def max_pooling(obs, k):
    n_obs = np.empty(0)

    for idx, v in enumerate(obs):
        d_obs = []
        d_obs.append(v)
        if (idx+1) % k == 0:
            min_v = max(d_obs)
            n_obs = np.append(n_obs, min_v)
    return n_obs

def calc_reward(reward, t):
    reward += -0.001*t
    return reward

def normalize(observation):
    max_v = np.amax(observation)
    min_v = np.amin(observation)
    return np.abs(((observation - min_v) / (max_v - min_v)) - 1 )

class Network(nn.Module):
    def __init__(self, obs_size, hidden_size):
        super(Network, self).__init__()

        self.common_layer = nn.Sequential(
            nn.Linear(obs_size, args.hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
        )

        self.fc1 = nn.Linear(hidden_size + 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1) #calc v
        self.fc3 = nn.Linear(hidden_size, 1) #calc w
        self.fc4 = nn.Linear(hidden_size, 1) #calc value

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.action = pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=2,
            var_type="diagonal",
            var_func=nn.Sigmoid(),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )

    def forward(self, x):
        obs = x[0]
        goal = x[1]
        h = self.common_layer(obs)
        h = torch.cat([h, goal],dim=-1)
        h = self.elu(self.fc1(h))

        value = self.elu(self.fc4(h))
        hv = self.sigmoid(self.fc2(h))
        hw = self.tanh(self.fc3(h))
        # ha = self.tanh(self.fc3(h))
        ha = torch.cat([hv,hw], dim=-1)
        action = self.action(ha)

        return action, value

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x[0].shape)
        return x


if __name__ == "__main__":

    args = args_parse()

    env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

    save_dir = make_save_dir(args.save_dir)
    with open(os.path.join(save_dir,'args.json'), 'w') as f:
        json.dump(vars(args),f)

    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        print("Device set to : cpu")

    if args.dynamic_env:
        obs_size = int(env.observation_space[0].size / args.k) + 3 + args.can_obs_people*5 #each human has five states
    else:
        obs_size = int(env.observation_space[0].size / args.k) # 3 is delta goal pose
    action_size = env.action_space.shape[0]

    model = Network(obs_size, args.hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=None,
        gpu=0,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=args.gamma,
        lambd=0.97,
        )

    writer = SummaryWriter(os.path.join(save_dir, "summary"))
    max_reward = 0.0

    with tqdm(range(args.max_episodes)) as pbar:
        for i_episode in pbar:
            if args.dynamic_env:
                observation, people = env.reset()
            else:
                observation, _ = env.reset()

            if args.midas_env:
                observation = normalize(observation[0])
                observation = max_pooling(observation, args.k)
            else:
                observation = observation[0]
                observation = min_pooling(observation, args.k)

            if args.dynamic_env:
                observation = np.concatenate([observation, people[0]], 0)
            action = np.array([0.0,0.0])
            done = False
            epidode_reward_sum = 0
            total_reward = 0
            t = 0

            for t in range(args.max_t+1):
                actions = []
                if t == 0: #for calc delta goal pose
                    actions.append(action)
                    _, _, _, _, delta_goal_pose = env.step(actions)
                else:
                    # obs = np.concatenate([observation, delta_goal_pose], 0)
                    tensor_obs = torch.from_numpy(observation.astype(np.float32)).clone()
                    tensor_delta_goal_pose = torch.from_numpy(delta_goal_pose.astype(np.float32)).clone()
                    action = agent.act((tensor_obs,tensor_delta_goal_pose))
                    actions.append(action)
                    # action = np.array([0.0,0.0])
                    # actions.append(action)
                    observation, _, reward, done, delta_goal_pose =  env.step(actions)

                    if args.midas_env:
                        observation = normalize(observation[0])
                        observation = max_pooling(observation, args.k)

                    else:
                        observation = min_pooling(observation[0], args.k)

                    reward  = calc_reward(reward, t)
                    total_reward += reward

                    reset = t == args.max_t

                    if done == 1: done = True
                    else: done = False

                    if args.dynamic_env:
                        observation = np.concatenate([observation, people[0]], 0)

                    tensor_obs = torch.from_numpy(observation.astype(np.float32)).clone()
                    tensor_delta_goal_pose = torch.from_numpy(delta_goal_pose.astype(np.float32)).clone()
                    agent.observe((tensor_obs, tensor_delta_goal_pose), reward, done, reset)

                    if done or reset: break
                    t+=1

                    if (1 + i_episode) % args.render_interval == 0:
                        env.render()

            if i_episode % 5 == 0:
                writer.add_scalar("avg reward", total_reward/t, i_episode)
            env.close()
            pbar.set_postfix(OrderedDict(avg_reward=total_reward/t))

            if max_reward < (total_reward/t):
                torch.save(model.state_dict(), os.path.join(save_dir, "weights", "model.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "weights", "final_model.pth"))

        writer.close()
