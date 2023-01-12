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
from torch import nn
import pfrl
from pfrl import utils
from pfrl.agents import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='d_kan')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=10000)
parser.add_argument('-mepi', '--max_episodes', type=int, default=1000)
parser.add_argument('--agent-num', type=int, default=1)
parser.add_argument('--action-std', type=float, default=0.6)
parser.add_argument('--action-std-decay-rate', type=float, default=0.05)
parser.add_argument('--min-action-std', type=float, default=0.1)
parser.add_argument('--action-std-decay-freq', type=int, default=2.5e5)
parser.add_argument('--update-interval', type=float, default=1000*4)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=32)
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

obs_size = env.observation_space[0].size - 3 + 5 * 5
action_size = env.action_space.shape[0]

obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_size, clip_threshold=5
    )
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

policy = pfrl.nn.RecurrentSequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.LSTM(input_size=64, hidden_size=32),
    nn.Tanh(),
    nn.Linear(32, action_size),
    pfrl.policies.GaussianHeadWithStateIndependentCovariance(
        action_size=action_size,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    ),
)
# ortho_init(policy[0], gain=1)
# ortho_init(policy[2], gain=1)
# ortho_init(policy[4], gain=1e-2)
# ortho_init(vf[0], gain=1)
# ortho_init(vf[2], gain=1)
# ortho_init(vf[4], gain=1)

vf = pfrl.nn.RecurrentSequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.LSTM(input_size=64, hidden_size=32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

# model = pfrl.nn.Branched(policy, vf)
model = pfrl.nn.RecurrentBranched(policy, vf)
opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=0,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=0,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
        recurrent=True,
        max_recurrent_sequence_len=1000
    )

for i_episode in range(args.max_episodes):
    observation, obs_people  = env.reset()
    done = False
    epidode_reward_sum = 0
    total_reward = 0
    t = 0
    for t in range(args.max_t):
        actions = []
        if args.agent_num > 1:
            print("multi agent")
        else:
            concat_obs = np.hstack((observation[0][:1080], obs_people[0]))
            concat_obs = torch.from_numpy(concat_obs.astype(np.float32)).clone().to(device)
            action = agent.act(concat_obs)
            actions.append(action)
            observation, obs_people, reward, done, _ = env.step(actions)
            reward += -0.01*t
            total_reward += reward

            concat_obs = np.hstack((observation[0][:1080], obs_people[0]))
            concat_obs = torch.from_numpy(concat_obs.astype(np.float32)).clone().to(device)
            reset = t == args.max_t
            if done == 1: done = True
            else: done = False
            agent.observe(concat_obs, reward, done, reset)
            if done or reset: break
        t+=1
        env.render()
    env.close()
    print("total reward: {}".format(total_reward/t))
