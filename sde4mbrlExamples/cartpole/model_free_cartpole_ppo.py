# Import the skrl components to build the RL system

import sys
sys.path.append('../..')

from modified_cartpole_continuous import CartPoleEnv

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env

from mbrlLibUtils.rl_networks import Value, Policy

# class Policy(GaussianMixin, Model):
#     def __init__(self, observation_space, action_space, device,
#                  clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

#         self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
#                                  nn.ReLU(),
#                                  nn.Linear(64, 32),
#                                  nn.ReLU(),
#                                  nn.Linear(32, self.num_actions),
#                                  nn.Tanh())

#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#     def compute(self, inputs, role):
#         return self.net(inputs["states"]), self.log_std_parameter, {}

# class Value(DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False):
#         Model.__init__(self, observation_space, action_space, device)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
#                                  nn.ReLU(),
#                                  nn.Linear(64, 32),
#                                  nn.ReLU(),
#                                  nn.Linear(32, 1))

#     def compute(self, inputs, role):
#         return self.net(inputs["states"]), {}

env = wrap_env(CartPoleEnv())

device = env.device

models_ppo = {}

models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
models_ppo["value"] = Value(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ppo.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo['rollouts'] = 2048 # number of steps per environment per update

# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=cfg_ppo['rollouts'], num_envs=env.num_envs, device=device, replacement=False)

agent_ppo = PPO(models=models_ppo,
                  memory=memory,
                  cfg=cfg_ppo,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 150000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)

# start training
trainer.train()