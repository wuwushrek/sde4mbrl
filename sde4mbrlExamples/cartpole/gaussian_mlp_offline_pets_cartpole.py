import hydra

from tqdm import tqdm
import sys
sys.path.append('../..')

from sde4mbrlExamples.cartpole.modified_cartpole_continuous import CartPoleEnv
from sde4mbrlExamples.cartpole.trajectory_eval_functions import trajectory_eval_constructor

# Setup the CEM experiment.
trajectory_evaluation_type = 'true_env' # 'gaussian_mlp'
experiment_name = 'gaussian_mlp_ensemble_cartpole_learned'
num_particles = 1
seed = 42

# Construct the CEM agent
from config.cem_agent_config import agent_cfg
agent = hydra.utils.instantiate(agent_cfg)
trajectory_eval_fn = trajectory_eval_constructor(
    trajectory_evaluation_type,
    num_particles=num_particles,
    seed=seed,
    experiment_name=experiment_name,
)
agent.set_trajectory_eval_fn(trajectory_eval_fn)

# Test the agent
def test_agent_performance(agent, env, num_trials, trial_length, render=False):
    rewards_list = []

    if render:
        env.render_mode = 'human'
    else:
        env.render_mode = 'rgb_array'

    for _ in tqdm(range(num_trials)):
        obs, _ = env.reset()
        total_reward = 0
        steps_trial = 0
        done = False

        # # Plan once at the start of the episode.
        # plan = [a for a in agent.plan(obs)]
        # assert len(plan) >= trial_length, "The plan is too short!"

        while not done:
            # # Plan every step.
            # plan = [a for a in agent.plan(obs)]

            action = agent.act(obs)
            # action = plan.pop(0)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                break

        print(total_reward)

        rewards_list.append(total_reward)

    return rewards_list

env = CartPoleEnv()

num_trials = 10
trial_length = 75

rewards_list = test_agent_performance(agent, env, num_trials, trial_length, render=False)

print(rewards_list)