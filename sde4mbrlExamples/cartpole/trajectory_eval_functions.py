import os
from mbrlLibUtils.save_and_load_models import load_model_and_config
from mbrlLibUtils import termination_functions
from mbrlLibUtils import reward_functions
from mbrl.models import ModelEnv
from modified_cartpole_continuous import CartPoleEnv

import torch

from torch import functional
import numpy as np

from typing import Optional, Callable, Tuple

def trajectory_eval_constructor(
    eval_fn_type : str,                             
    num_particles : int, 
    seed : Optional[int],
    experiment_name : Optional[str] = None
):
    """
    A factory function to construct a trajectory evaluation function.
    
    Parameters
    ----------
    eval_fn_type : str
        The type of trajectory evaluation function to construct. Must be one of
        "gaussian_mlp" or "true_env".
    num_particles : int
        The number of particles to use when evaluating the action sequences.
    seed : int
        The seed to use for the random number generator.
    experiment_name : str
        The name of the experiment to load the model from. Only required if
        eval_fn_type is "gaussian_mlp".
    """
    
    if eval_fn_type == "gaussian_mlp":
        assert experiment_name is not None, "Must provide experiment name for gaussian MLP trajectory evaluation function"
        return generate_gaussian_mlp_trajectory_eval_fn(
            experiment_name=experiment_name,
            num_particles=num_particles,
            generator_seed=seed
        )
    elif eval_fn_type == "true_env":
        return generate_true_env_trajectory_eval_fn(
            num_particles=num_particles,
            seed=seed
        )
    else:
        raise ValueError(f"Unrecognized trajectory evaluation function type: {eval_fn_type}")

def generate_gaussian_mlp_trajectory_eval_fn(
        experiment_name, 
        num_particles=1,
        generator_seed=42
    ):
    """
    Construct a function to evaluate the quality of planned actions from an initial state
    using a pre-trained ensemble of gaussian MLP models.
    
    Parameters
    ----------
    experiment_name : str
        The name of the experiment to load the model from.
    num_particles : int
        The number of particles to use when evaluating the action sequences.
    generator_seed : int
        The seed to use for the random number generator.
        
    Returns
    -------
    trajectory_eval_fn : function
        A function that takes an initial state and a batch of action sequences and returns
        the expected total reward for each action sequence.
    """

    save_folder = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
    dynamics_model, cfg = load_model_and_config(save_folder, propagation_method="expectation")

    generator = torch.Generator(device=dynamics_model.device)
    generator.manual_seed(generator_seed)

    # Wrap the dynamics model in our model environment
    env = CartPoleEnv() # Only instantiating this to get the observation and action spaces
    model_env = ModelEnv(
                    env,
                    dynamics_model, 
                    termination_functions.cartpole_swingup, 
                    reward_functions.cartpole_swingup, 
                    generator
                )
    
    def trajectory_eval_fn(initial_state, action_sequences):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles
        )
        
    return trajectory_eval_fn

def generate_true_env_trajectory_eval_fn(num_particles=1, seed=None):
    """
    Construct a function to evaluate the quality of planned actions from an initial state
    using the true cartpole environment.
    
    Parameters
    ----------
    num_particles : int
        number of times each action sequence is replicated. The final
        value of the sequence will be the average over its particles values.
    seed : int
        The seed to use for the random number generator.
        
    Returns
    -------
    trajectory_eval_fn : function   
        A function that takes an initial state and a batch of action sequences and returns
        the expected total reward for each action sequence.
    """
    env = CartPoleEnv(measurement_noise_diag=np.zeros(4))
    
    if seed is not None:
        env.reset(seed)
        
    def step_env(action, obs):
        env.reset()
        set_env_state(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # next_state = env.state
        return next_obs, reward, terminated, truncated 
        
    vectorized_step_env = np.vectorize(
        step_env, 
        signature='(a),(d)->(d),(),(),()'
    )
    
    def set_env_state(obs):
        (x, x_dot, sin_theta, cos_theta, theta_dot) = obs
        env.state = np.array([x, x_dot, np.arctan2(sin_theta, cos_theta), theta_dot])
    
    def evaluate_action_sequences(
        initial_obs: np.ndarray,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates a batch of action sequences on the model.

        Args:
            initial_obs (np.ndarray): the initial observation for the trajectories.
            action_sequences (torch.Tensor): a batch of action sequences to evaluate.  Shape must
                be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size, horizon,
                and action dimension, respectively.

        Returns:
            (torch.Tensor): the accumulated reward for each action sequence, averaged over its
            particles.
        """
        
        device = action_sequences.device
        
        action_sequences = action_sequences.cpu().numpy()
        assert len(action_sequences.shape) == 3
        population_size, horizon, action_dim = action_sequences.shape
        # either 1-D state or 3-D pixel observation
        assert initial_obs.ndim in (1, 3)
        tiling_shape = (num_particles * population_size,) + tuple(
            [1] * initial_obs.ndim
        )
        last_obs = np.tile(initial_obs, tiling_shape).astype(np.float32)
        batch_size = last_obs.shape[0]
        
        env.reset()
        set_env_state(initial_obs)
        
        total_rewards = np.zeros((batch_size,))
        dones = np.zeros((batch_size,), dtype=bool)
        
        for time_step in range(horizon):
            action_for_step = action_sequences[:, time_step, :]
            
            action_batch = np.repeat(
                action_for_step, num_particles, axis=0
            )
            
            last_obs, rewards, terminated, truncated = vectorized_step_env(
                action_batch, last_obs
            )
            
            rewards[dones] = 0
            dones = dones | truncated | terminated
            total_rewards += rewards

        total_rewards = total_rewards.reshape(-1, num_particles)
        return torch.tensor(total_rewards.mean(axis=1), device=device, dtype=torch.float32)
    
    return evaluate_action_sequences