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

from sde4mbrlExamples.cartpole.cartpole_sde import cartpole_sde_gym

env = cartpole_sde_gym(filename='~/Documents/sde4mbrl/sde4mbrlExamples/cartpole/my_models/cartpole_bb_rand_sde.pkl', num_particles=1, 
                       jax_seed=10, use_gpu=True, jax_gpu_mem_frac=0.2,)

# exit()
# env = wrap_env(CartPoleEnv())
env = wrap_env(env)

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
cfg_trainer = {"timesteps": int(5e5), "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ppo)

# start training
trainer.train()