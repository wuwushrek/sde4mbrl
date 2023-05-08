from skrl.models.torch import Model, DeterministicMixin, GaussianMixin

from modified_cartpole_continuous import CartPoleEnv
from skrl.envs.torch import wrap_env
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

import sys
sys.path.append('../..')
from mbrlLibUtils.rl_networks import Value, Policy

import numpy as np

from time import sleep
import os

env = wrap_env(CartPoleEnv(render_mode="rgb_array"))
    
models_ppo = {
    'policy': Policy(env.observation_space, env.action_space, env.device),
    'value' : Value(env.observation_space, env.action_space, env.device)
}
    
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
# run_name = '23-04-07_19-59-41-474383_PPO'
# run_name = '23-05-03_18-24-19-763565_PPO'
# run_name = '23-05-03_19-25-32-459262_PPO' # Good one.
# run_name = '23-05-04_14-28-04-729217_PPO'
# run_name = '23-05-04_15-17-28-709494_PPO'
run_name = '23-05-05_19-09-33-464430_PPO_gaussian_mlp'
load_str = os.path.abspath(os.path.join('runs', run_name, 'checkpoints', 'best_agent.pt'))
agent_ppo.load(load_str)
agent_ppo._rnn = None

with open(load_str, 'rb') as f:
    state_dict = torch.load(f)

env = CartPoleEnv(render_mode="human")

policy_net = Policy(env.observation_space, env.action_space, 'cpu')
policy_net.load_state_dict(state_dict['policy'])

obs, _ = env.reset()

obs_list = []

for i in range(500):
    with torch.no_grad():
        obs_th = torch.tensor(obs, device=policy_net.device, dtype=torch.float32)
        act, _, _ = policy_net.act({"states" : obs_th}, role='policy')
        sleep(0.02)
        # act = agent_ppo.act(obs, 0, 0)[0]
    obs, _, _, _, _ = env.step(act.detach().numpy())
    env.render()
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