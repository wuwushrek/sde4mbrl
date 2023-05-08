import os

import numpy as np

from sde4mbrlExamples.cartpole.modified_cartpole_continuous import CartPoleEnv
from typing import Optional

from mbrlLibUtils.save_and_load_models import load_model_and_config

import torch

# We create the gym environment
class CartPoleGaussianMLPEnv(CartPoleEnv):
    def __init__(
        self, 
        load_file_name : str, 
        num_particles : int = 1, 
        torch_seed : Optional[int] = None,
        use_gpu=True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.device = torch.device("cuda" if use_gpu else "cpu")
        
        self.load_file = load_file_name
        self.num_particles = num_particles
        self.use_gpu = use_gpu
        self.torch_seed = torch_seed
        
        dynamics_model, cfg = load_model_and_config(
            self.load_file, 
            propagation_method="expectation"
        )
        
        self.dynamics_model = dynamics_model
        self.cfg = cfg
        
        self.num_models = self.cfg['dynamics_model']['ensemble_size']
        
        self.generator = torch.Generator(device=self.device)
        if torch_seed is not None:
            self.generator.manual_seed(torch_seed)
        
    def step_dynamics(self, action):
        # We use the SDE model to predict the next state
        assert self.state is not None, 'The state must be initialized'
        
        init_state = self.state
        initial_obs = self.get_obs(init_state)
        
        initial_obs_batch = np.tile(
            initial_obs, (self.num_particles * self.num_models, 1)
        ).astype(np.float32)

        model_state = self.dynamics_model.reset(
            initial_obs_batch.astype(np.float32), rng=self.generator
        )
        
        action_dim = 1 if action.ndim==0 else action.shape[0]
        act_tiled = np.tile(
            action, (self.num_particles * self.num_models, action_dim)
        ).astype(np.float32)
        act_batch = torch.tensor(
            act_tiled, 
            device=self.device
        )
        
        with torch.no_grad():
            next_obs, _, _, _ = self.dynamics_model.sample(act_batch, model_state, rng=self.generator)
            next_obs_avg = next_obs.mean(dim=0).cpu().numpy()
        
        # Update the state using the observations. 
        # state = (x, x_dot, theta, theta_dot)
        # observations = (x, x_dot, sin(theta), cos(theta), theta_dot)
        self.state = (next_obs_avg[0], next_obs_avg[1], np.arctan2(next_obs_avg[2], next_obs_avg[3]), next_obs_avg[4])
    
    def get_obs(self, state):
        x, x_dot, theta, theta_dot = state
        return np.array((x, x_dot, np.sin(theta), np.cos(theta), theta_dot))
    
    def reset_model_seed(self,):
        if self.torch_seed is not None:
            self.generator.manual_seed(self.torch_seed)
    
if __name__ == '__main__':
    
    load_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_models', 'gaussian_mlp_ensemble_cartpole_learned'))
    
    env = CartPoleGaussianMLPEnv(
        load_file_name = load_dir,
        num_particles = 1,
        torch_seed = 42,
        use_gpu = True,
        # render_mode = 'human'
    )
    
    print(env)
    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
        env.render()
    env.close()