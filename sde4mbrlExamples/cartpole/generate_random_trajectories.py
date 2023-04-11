from modified_cartpole_continuous import CartPoleEnv

import numpy as np

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
        obs = env.get_obs(env.state)
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

def main():
    env= CartPoleEnv(render_mode='rgb_array')

    def policy(obs):
        return env.action_space.sample()
    
    def policy(obs):
        return np.array([0.0])
    
    obs_traj, action_traj = rollout_gym_trajectory(env, policy)
    
    dataset = gen_dataset(env, policy, num_trajs=100, max_steps=200, init_state_low=np.array([0.0, 0.0, -np.pi, -0.1]), init_state_high=np.array([0.0, 0.0, -np.pi, 0.1]))
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dataset[0][0][:, 3], label='cos(theta)')
    plt.show()    

if __name__ == "__main__":
    main()