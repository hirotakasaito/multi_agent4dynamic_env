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
    parser.add_argument('--map', help='Specify map setting folder.', default='midas_env')
    parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1000)
    parser.add_argument('-mt', '--max_t', type=int, default=1000)
    parser.add_argument('-mepi', '--max_episodes', type=int, default=15000)
    parser.add_argument('--agent-num', type=int, default=1)
    parser.add_argument('--update-interval', type=float, default=1000*20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=64*2)
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
    parser.add_argument('--random-noise', type=bool, default=True)

    parser.add_argument('--save-dir', type=str, default="/share/private/27th/hirotaka_saito/logs/ppo_midas2")
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

def obs_random_noise(obs):
    obs_size = len(obs)
    obs =  obs + np.random.normal(loc = 0.0, scale = 0.1, size = obs_size)
    return obs

def action_random_noise(action):
    action =  action + np.random.normal(loc = 0.0, scale = 0.01, size = 2)
    return action

def calc_reward(reward, t):
    reward += -0.001*t
    return reward

def normalize(observation):
    max_v = np.amax(observation)
    min_v = np.amin(observation)
    return np.abs(((observation - min_v) / (max_v - min_v)) - 1 )

def convert_world2robot_frame(delta_goal_pose):
    x = delta_goal_pose[0]
    y = delta_goal_pose[1]
    yaw = delta_goal_pose[2]
    if yaw > 2*np.pi:
        yaw = yaw % 2*np.pi
    trans_pose = np.array([ x*np.cos(yaw) + y*np.sin(yaw),
                           -x*np.sin(yaw) + y*np.cos(yaw)
    ])
    return trans_pose

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
        obs_size = int(env.observation_space[0].size / args.k) + 2 + args.can_obs_people*5 #each human has five states
    else:
        obs_size = int(env.observation_space[0].size / args.k) + 2 # 2 is delta goal pose
    action_size = env.action_space.shape[0]

    policy = nn.Sequential(
        nn.Linear(obs_size, args.hidden_size),
        nn.ELU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ELU(),
        nn.Linear(args.hidden_size, action_size),
        nn.Tanh(),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            # var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_func=nn.Sigmoid(),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = nn.Sequential(
        nn.Linear(obs_size, args.hidden_size),
        nn.ELU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.ELU(),
        nn.Linear(args.hidden_size, 1),
    )

    model = pfrl.nn.Branched(policy, vf)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=None,
        gpu=0,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=args.eps_clip,
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

            if args.random_noise:
                observation = obs_random_noise(observation)

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
                    delta_goal_pose = convert_world2robot_frame(delta_goal_pose)
                else:
                    obs = np.concatenate([observation, delta_goal_pose], 0)
                    x = torch.from_numpy(obs.astype(np.float32)).clone()
                    action = agent.act(x)

                    if args.random_noise:
                        action = action_random_noise(action)

                    actions.append(action)
                    # action = np.array([0.0,0.0])
                    # actions.append(action)
                    observation, _, reward, done, delta_goal_pose =  env.step(actions)
                    delta_goal_pose = convert_world2robot_frame(delta_goal_pose)

                    if args.midas_env:
                        observation = normalize(observation[0])
                        observation = max_pooling(observation, args.k)

                    else:
                        observation = min_pooling(observation[0], args.k)

                    if args.random_noise:
                        observation = obs_random_noise(observation)

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

                    if (1 + i_episode) % args.render_interval == 0:
                        env.render()

            if i_episode % 5 == 0:
                writer.add_scalar("avg reward", total_reward/t, i_episode)
            env.close()
            pbar.set_postfix(OrderedDict(avg_reward=total_reward/t))

            if max_reward < (total_reward/t):
                torch.save(vf.state_dict(), os.path.join(save_dir, "weights", "vf.pth"))
                torch.save(policy.state_dict(), os.path.join(save_dir, "weights", "policy.pth"))

        torch.save(vf.state_dict(), os.path.join(save_dir, "weights", "final_vf.pth"))
        torch.save(policy.state_dict(), os.path.join(save_dir, "weights", "final_policy.pth"))

        writer.close()
