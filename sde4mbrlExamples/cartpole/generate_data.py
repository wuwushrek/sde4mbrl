from modified_cartpole_continuous import CartPoleEnv
from skrl.envs.torch import wrap_env

import numpy as np

import pickle, os
import torch
import gymnasium as gym
from typing import Optional

import sys
sys.path.append('../..')

from mbrlLibUtils.rl_networks import Policy

def rollout_gym_trajectory(env, policy, init_state=None, max_steps=200):
    """
    Rollout a trajectory in a gym environment using a policy.
    
    Parameters
    ----------
    env : 
        Gym environment.
    policy :
        Policy function that takes in an observation and returns an action.
    init_state :
        Initial state of the environment. If None, the initial state is
        set using the environment's reset method.
    max_steps :
        Maximum number of steps to rollout.
        
    Returns
    -------
    traj : tuple
        Tuple of (observations, actions) where observations and actions
        are numpy arrays of shape (T, obs_dim) and (T, action_dim)
    """
    obs, _ = env.reset()
    if init_state is not None:
        env.state = init_state
        obs = env.get_obs(init_state)
    obs_traj = [obs]
    action_traj = []
    for i in range(max_steps):
        action = policy(obs)
        obs, _, _, _, _ = env.step(action)
        obs_traj.append(obs)
        action_traj.append(action)
    return (np.vstack(obs_traj), np.vstack(action_traj))

def gen_dataset(env, policy, num_trajs=100, max_steps=200, init_state_low=None, init_state_high=None):
    dataset = []
    assert init_state_low is None and init_state_high is None or init_state_low is not None and init_state_high is not None
    assert init_state_low is None or init_state_low.shape == init_state_high.shape
    for i in range(num_trajs):
        if init_state_low is not None and init_state_high is not None:
            init_state = np.random.uniform(init_state_low, init_state_high)
            traj = rollout_gym_trajectory(env, policy, init_state, max_steps)
        else:
            traj = rollout_gym_trajectory(env, policy, max_steps)
        dataset.append(traj)
    return dataset

def define_policy(policy_type : str, env : gym.Env, policy_save_path : Optional[str] = None):
    """
    Define a policy function.
    
    Arguments
    ----------
    policy_type : str
        Type of policy to define. Options are 'random', 'zero', and 'learned'.
    env : gym.Env
        Gym environment.
    policy_save_path : str, optional
        Path to a saved policy network. Only used if policy_type is 'learned'.
    """
    if policy_type == 'random':
        def policy(obs):
            return env.action_space.sample()
    
    elif policy_type == 'no_actions':
        def policy(obs):
            return np.array([0.0])   
        
    elif policy_type == 'learned':
        # Load the pre-trained policy network
        with open(policy_save_path, 'rb') as f:
            state_dict = torch.load(f)
            
        policy_net = Policy(env.observation_space, env.action_space, device='cpu')
        policy_net.load_state_dict(state_dict['policy'])
        
        def policy(obs):
            obs_th = torch.tensor(obs, device=policy_net.device, dtype=torch.float32)
            with torch.no_grad():
                act, _, _ = policy_net.act(
                    {'states' : obs_th}, 
                    role='policy'
                )
            return act.detach().numpy()
         
    return policy

def main():
    env= CartPoleEnv(render_mode='rgb_array')

    policy_type = 'learned'

    # If loading a pre-trained policy, set the path to the policy here
    run_name = '23-04-07_19-59-41-474383_PPO'
    load_str = os.path.abspath(os.path.join('runs', run_name, 'checkpoints', 'best_agent.pt'))
    
    policy = define_policy(policy_type, env, load_str)
        
    dataset = gen_dataset(env, 
                        policy, 
                        num_trajs=100, 
                        max_steps=200, 
                        init_state_low=np.array([0.0, 0.0, -np.pi + 0.1, -0.1]), 
                        init_state_high=np.array([0.0, 0.0, np.pi - 0.1, 0.1])
                    )
    
    dataset_name = policy_type + '.pkl'
    save_path = os.path.abspath(os.path.join(os.path.curdir, 'my_data', dataset_name))
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dataset[0][0][:, 3], label='cos(theta)')
    plt.show()    

if __name__ == "__main__":
    main()