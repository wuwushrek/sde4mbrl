from mbrl.util.replay_buffer import ReplayBuffer
import numpy as np
import torch

def populate_replay_buffers(dataset : tuple, buffer_size : int, save_actions : bool = False):
    """
    Populate replay buffer with the dataset of trajectories.

    Parameters
    ----------
    dataset : tuple
        Dataset is a tuple of trajectories. Each trajectory is a tuple of (observations, actions).
        observations is a time x obs_dim ndarray. actions is a time x action_dim ndarray.
    buffer_size : int
        Size of the replay buffer.
    save_actions : bool
        Flag indicating whether or not to save actions in the replay buffer.

    Returns
    -------
    replay_buffer : mbrl.util.replay_buffer.ReplayBuffer
        Replay buffer populated with the dataset of trajectories.
    """
    obs_dim = dataset[0][0].shape[1]
    if save_actions:
        action_dim = dataset[0][1].shape[1]
    else:
        action_dim = 0
    replay_buffer = ReplayBuffer(buffer_size, (obs_dim,), (action_dim,))

    for traj_ind in range(len(dataset)):
        traj = dataset[traj_ind]
        traj_len = len(traj[0])
        for t in range(traj_len - 1):
            current_obs = traj[0][t]
            current_action = traj[1][t]
            next_obs = traj[0][t + 1]
            replay_buffer.add(current_obs, current_action, next_obs, 0, False)

    return replay_buffer

def generate_sample_trajectories(init_state, num_particles, dynamics_model, generator, time_horizon, device=None):
    """
    Generate sample trajectories from the dynamics model
    
    Parameters
    ----------
    init_state : np.array
        Initial state of the system
    num_particles : int
        Number of particles to propagate
    dynamics_model : mbrl.models.OneDTransitionRewardModel
        Dynamics model to use for generating the sample trajectories.
        Should be a OneDTransitionRewardModel wrapper around a GaussianMLPEnsembleModel.
    generator : torch.Generator
        Random number generator
    time_horizon : int
        Time horizon for the sample trajectories
        
    Returns
    -------
    sample_trajectories : np.array
        Sample trajectories generated by the dynamics model
    """
    if device is None:
        device = torch.device('cpu')

    initial_obs_batch = np.tile(
        init_state, (num_particles, 1)
    ).astype(np.float32)

    model_state = dynamics_model.reset(
        initial_obs_batch.astype(np.float32), rng=generator
    )

    act = torch.zeros((num_particles,0), device=device)

    for t in range(time_horizon):
        with torch.no_grad():
            next_obs, _, _, model_state = dynamics_model.sample(act, model_state, rng=generator)
        if t == 0:
            sample_trajectories = next_obs.reshape(num_particles, 1, -1)
        else:
            sample_trajectories = torch.concatenate((sample_trajectories, next_obs.reshape(num_particles, 1, -1)), axis=1)

    return sample_trajectories