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
from pfrl import utils
from pfrl.agents import PPO
from collections import OrderedDict
from matplotlib import animation
import matplotlib.pyplot as plt

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', help='Specify map setting folder.', default='narrow_env')
    parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1000)
    parser.add_argument('-mt', '--max_t', type=int, default=1000)
    parser.add_argument('-mepi', '--max_episodes', type=int, default=10)
    parser.add_argument('--agent-num', type=int, default=1)
    parser.add_argument('--update-interval', type=float, default=1000*10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eps-clip', type=int, default=0.2)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--lr', type=int, default=5e-4)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--lstm-hidden-size', type=int, default=64)
    parser.add_argument('--render-interval', type=int, default=50)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dynamic-env', type=bool, default=True)
    parser.add_argument('--can-obs-people', type=int, default=5)


    parser.add_argument('--pretrained-dir', type=str, default="/share/private/27th/hirotaka_saito/logs/ppo_lstm_dynamic/20230123102218")
    parser.add_argument('--save-gif-dir', type=str, default="../outputs/")
    args = parser.parse_args()
    return args

def min_pooling(obs, k):
    n_obs = np.empty(0)

    for idx, v in enumerate(obs):
        d_obs = []
        d_obs.append(v)
        if (idx+1) % k == 0:
            min_v = min(d_obs)
            n_obs = np.append(n_obs, min_v)
    return n_obs

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=10)
    anim.save(os.path.join(path, filename), writer='imagemagick', fps=110)

def calc_reward(reward, t):
    reward += -0.001*t
    return reward

if __name__ == "__main__":

    args = args_parse()

    env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

    os.makedirs(os.path.join(args.save_gif_dir,"gif"), exist_ok=True)

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
        obs_size = int(env.observation_space[0].size / args.k) + 3 # 3 is delta goal pose
    action_size = env.action_space.shape[0]

    policy = pfrl.nn.RecurrentSequential(
        nn.Linear(obs_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ReLU(),
        nn.LSTM(input_size=args.hidden_size, hidden_size=args.lstm_hidden_size, num_layers=args.num_layers),
        nn.ReLU(),
        nn.Linear(args.lstm_hidden_size, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = pfrl.nn.RecurrentSequential(
        nn.Linear(obs_size, args.hidden_size),
        nn.ReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ReLU(),
        nn.LSTM(input_size=args.hidden_size, hidden_size=args.lstm_hidden_size),
        nn.ReLU(),
        nn.Linear(args.lstm_hidden_size, 1),
    )

    policy.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'weights', 'policy.pth')))
    vf.load_state_dict(torch.load(os.path.join(args.pretrained_dir, 'weights', 'vf.pth')))

    model = pfrl.nn.RecurrentBranched(policy, vf)
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
        gamma=0.995,
        lambd=0.97,
        recurrent=True,
        max_recurrent_sequence_len=1000
        )

    max_reward = 0.0

    with agent.eval_mode():
        frames = []
        for i_episode in range(args.max_episodes):
            if args.dynamic_env:
                observation, people = env.reset()
            else:
                observation, _ = env.reset()
            observation = min_pooling(observation[0], args.k)
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
                    obs = np.concatenate([observation, delta_goal_pose], 0)
                    x = torch.from_numpy(obs.astype(np.float32)).clone()
                    action = agent.act(x)
                    actions.append(action)
                    observation, _, reward, done, delta_goal_pose =  env.step(actions)
                    observation = min_pooling(observation[0], args.k)
                    reward  = calc_reward(reward, t)
                    total_reward += reward

                    reset = t == args.max_t

                    if done == 1: done = True
                    else: done = False

                    if args.dynamic_env:
                        observation = np.concatenate([observation, people[0]], 0)

                    obs = np.concatenate([observation, delta_goal_pose], 0)
                    x = torch.from_numpy(obs.astype(np.float32)).clone()
                    agent.observe(x, reward, done, reset)

                    if done or reset: break
                    t+=1

                    env.render()
                    frames.append(env.render(mode="rgb_array"))

            env.close()
    save_frames_as_gif(frames, path=os.path.join(args.save_gif_dir, "gif"))
