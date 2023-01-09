import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, device):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(32, action_dim)
        self.lstm = nn.LSTM(64, 32)
        self.tanh = nn.Tanh()
        self.device = device

        self.h = torch.zeros(1, 32).to(self.device)
        self.c = torch.zeros(1, 32).to(self.device)

    def forward(self, state, get_action_flag):
        x = self.tanh(self.linear1(state))
        x = self.tanh(self.linear2(x))
        if get_action_flag:
            x = torch.unsqueeze(x, dim=0)
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        x = self.tanh(output)
        x = self.linear3(x)
        return x

class Critic(nn.Module):

    def __init__(self, state_dim, device):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3= nn.Linear(32, 1)
        self.lstm = nn.LSTM(64, 32)
        self.tanh = nn.Tanh()
        self.device = device

        self.h = torch.zeros(1, 32).to(self.device)
        self.c = torch.zeros(1, 32).to(self.device)

    def forward(self, state):
        x = self.tanh(self.linear1(state))
        x = self.tanh(self.linear2(x))
        output, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        x = self.tanh(output)
        x = self.linear3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()

        self.device = device

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        self.h = torch.zeros(1, 32).to(self.device)
        self.c = torch.zeros(1, 32).to(self.device)
        # actor
        # self.linear1 = nn.Linear(state_dim, 64)
        # self.linear2 = nn.Linear(64, 64)
        # self.linear3 = nn.Linear(32, action_dim)
        # self.linear4 = nn.Linear(state_dim, 64)
        # self.linear5 = nn.Linear(64, 64)
        # self.linear6 = nn.Linear(32, 1)
        # self.tanh = nn.Tanh()
        #
        # self.lstm1 = nn.LSTM(64, 32)
        # self.lstm2 = nn.LSTM(64, 32)
        self.actor = Actor(state_dim, action_dim, device)
        self.critic = Critic(state_dim, device)
        # self.actor = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.LSTM(64, 32),
        #                 nn.Tanh(e()),
        #                 nn.Linear(32, action_dim),
        #             )
        # # critic
        # self.critic = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.LSTM(64, 32),
        #                 nn.Tanh(e()),
        #                 nn.Linear(32, 1)
        #             )


    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state, True)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        action_mean = self.actor(state, False)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device, action_std_init=0.6):


        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init, device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def get_action(self, state):

        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        s
