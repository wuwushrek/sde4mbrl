from skrl.models.torch import Model, DeterministicMixin, GaussianMixin

from modified_cartpole_continuous import CartPoleEnv
from skrl.envs.torch import wrap_env
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

import numpy as np

import os

env = wrap_env(CartPoleEnv(render_mode="rgb_array"))

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions),
                                 nn.Tanh())

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
    
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
    
models_ppo = {}

models_ppo["policy"] = Policy(env.observation_space, env.action_space, env.device)
models_ppo["value"] = Value(env.observation_space, env.action_space, env.device)
    
# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ppo.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo['rollouts'] = 2048 # number of steps per environment per update

# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=cfg_ppo['rollouts'], num_envs=env.num_envs, device=env.device, replacement=False)

agent_ppo = PPO(models=models_ppo,
                  memory=memory,
                  cfg=cfg_ppo,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=env.device)
    
# run_name = '23-04-07_19-43-28-811965_PPO'
# run_name = '23-04-07_19-52-40-164902_PPO'
run_name = '23-04-07_19-59-41-474383_PPO'
load_str = os.path.abspath(os.path.join('runs', run_name, 'checkpoints', 'best_agent.pt'))
agent_ppo.load(load_str)
agent_ppo._rnn = None

obs, _ = env.reset()

obs_list = []

for i in range(200):
    with torch.no_grad():
        act = agent_ppo.act(obs, 0, 0)[0]
    obs, _, _, _, _ = env.step(act)
    # env.render()
    obs_list.append(obs)

obs_array = np.vstack([obs.cpu().detach().numpy() for obs in obs_list])

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(obs_array[:, 3], label='cos(theta)')
ax = fig.add_subplot(212)
ax.plot(obs_array[:, 0], label='x')
plt.legend(fontsize=15)

plt.show()