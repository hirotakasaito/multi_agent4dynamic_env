import torch
import torch.nn as nn
import pfrl

def min_pooling(obs, k):
    n_obs = np.empty(int(obs.size/k))

    for idx, v in enumerate(obs):
        d_obs = []
        d_obs.append(v)
        if idx % (k+1) == 0:
            min_v = min(d_obs)
            n_obs.append(min_v)
    return n_obs

class PPOLstm(nn.Module):

    def __init__(self, obs_size, delta_goal_size, embedding_size, action_size, k, hidden_size, lstm_hidden_size, num_layers):
        super(PPOLstm, self).__init__()

        self.embedding_size
        self.action_size = action_size
        self.delta_goal_size = delta_goal_size
        self.k = k
        self.obs_size = int(obs_size/k) + delta_goal_size

        self.fc1 = nn.Linear(slef.obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(lstm_hidden_size, 1)
        self.fc5 = nn.Linear(lstm_hidden_size, action_size)
        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = lstm_hidden_size, num_layers = num_layers)

        self.relu = nn.ReLU()


    def forward(self, obs, delta_goal):
        obs = min_pooling(obs, self.k)

        obs = torch.cat([obs, delta_goal], dim=-1)
        h = self.relu(self.fc1(obs))
        h = self.relu(self.fc2(obs))
        h = self.relu(self.fc3(obs))


        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),

