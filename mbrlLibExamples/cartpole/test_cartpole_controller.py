import sys, os
from tqdm import tqdm

from mbrlLibUtils.save_and_load_models import load_model_and_config
from mbrlLibUtils.modified_model_env import ModifiedModelEnv

import mbrl.planning as planning
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
from mbrl.env.cartpole_continuous import CartPoleEnv

import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

seed = 42
generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Load the pre-trained dynamics model
experiment_name = 'gaussian_mlp_ensemble_cartpole'
save_folder = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
dynamics_model, cfg = load_model_and_config(save_folder, propagation_method="expectation")

# Wrap the dynamics model in our model environment
env = CartPoleEnv() # Only instantiating this to get the observation and action spaces
model_env = ModifiedModelEnv(
                env.observation_space, 
                env.action_space, 
                dynamics_model, 
                termination_fns.cartpole, 
                reward_fns.cartpole, 
                generator
            )

# Construct the CEM agent
from config.cem_agent_config import agent_cfg
agent = planning.create_trajectory_optim_agent_for_model(
    model_env,
    agent_cfg,
    num_particles=20
)

# Test the agent
def test_agent_performance(agent, env, num_trials, trial_length, render=False):
    rewards_list = []

    for _ in tqdm(range(num_trials)):
        obs = env.reset()
        total_reward = 0
        steps_trial = 0
        done = False

        # Plan once at the start of the episode.
        plan = [a for a in agent.plan(obs)]
        assert len(plan) >= trial_length, "The plan is too short!"

        while not done:
            # # Plan every step.
            # plan = [a for a in agent.plan(obs)]

            action = plan.pop(0)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps_trial += 1
            if render:
                env.render()

            if steps_trial == trial_length:
                break

        rewards_list.append(total_reward)

    return rewards_list

num_trials = 10
trial_length = 75

rewards_list = test_agent_performance(agent, env, num_trials, trial_length, render=True)

print(rewards_list)