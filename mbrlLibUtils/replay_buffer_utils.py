from mbrl.util.replay_buffer import ReplayBuffer
import numpy as np
import torch

from mbrl.models import OneDTransitionRewardModel

from typing import Optional, Tuple, Callable

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
            replay_buffer.add(current_obs, current_action, next_obs, 0, False, False)

    return replay_buffer

def generate_sample_trajectories(
        init_state:np.ndarray, 
        num_particles:int, 
        dynamics_model:OneDTransitionRewardModel, 
        generator:torch.Generator, 
        time_horizon:int, 
        control_inputs:Optional[np.ndarray]=None,
        ufun:Optional[Callable]=None, 
        device:Optional[str]=None
    ):
    """
    Generate sample trajectories from the dynamics model. For control inputs,
    either pass in a batch of control inputs or a control function.
    If neither is passed, then control inputs of 0 are used.
    
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
    control_inputs : np.array
        Control inputs to use to generate the sample trajectories.
        Shape should be [time_horizon, control_dim]
    ufun : function
        Control function.
    device : torch.device
        Device to use for the computations.
        
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

    sample_trajectories = model_state['obs'].reshape(num_particles, 1, -1)

    act_size = dynamics_model.model.in_size - init_state.shape[-1]

    if control_inputs is not None:
        for t in range(time_horizon - 1):
            with torch.no_grad():
                actions = torch.tensor(np.tile(
                    control_inputs[t], (num_particles, act_size)
                ), device=device)
                next_obs, _, _, model_state = dynamics_model.sample(actions, model_state, rng=generator)
            sample_trajectories = torch.concatenate((sample_trajectories, next_obs.reshape(num_particles, 1, -1)), axis=1)
        return sample_trajectories
    else:
        if ufun is None:
            act_zero = torch.zeros((num_particles, act_size), device=device)
            ufun = lambda state : act_zero

        for t in range(time_horizon - 1):
            with torch.no_grad():
                next_obs, _, _, model_state = dynamics_model.sample(ufun(model_state), model_state, rng=generator)
            sample_trajectories = torch.concatenate((sample_trajectories, next_obs.reshape(num_particles, 1, -1)), axis=1)

        return sample_trajectories