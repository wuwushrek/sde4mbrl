import jax

import jax.numpy as jnp

import numpy as np

from sde4mbrl.nsde import ControlledSDE, create_sampling_fn, compute_timesteps, create_diffusion_fn
from sde4mbrl.utils import load_yaml, apply_fn_to_allleaf, update_params
from sde4mbrl.train_sde import train_model

import haiku as hk

import os
import pickle

from functools import partial


class CartPoleSDE(ControlledSDE):
    """ An SDE and ODE model of the mass spring damper system.
        The parameters of the model are usually defined in a yaml file.
    """
    def __init__(self, params, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name=name)

        # Parameters initialization values -> Values for HK PARAMETERS initialization
        self.init_params = params.get('init_params', {})

        # Initialization of the residual networks
        # This function setup the parameters of the unknown neural networks and the residual network
        self.init_residual_networks()

        self.state_scaling = jnp.array(self.params.get('state_scaling', [1.0] * self.n_x))
        # In case scaling factor is give, we also need to ensure scaling diffusion network inputs
        if 'state_scaling' in self.params:
            self.reduced_state = lambda x : jnp.array([x[1]/self.state_scaling[1], jnp.sin(x[2]), jnp.cos(x[2]), x[-1]/self.state_scaling[-1]])
    

    def prior_diffusion(self, x, u, extra_args=None):
        # Set the prior to a constant noise as defined in the yaml file
        return jnp.array(self.params['noise_prior_params'])

    def state_transform_for_loss(self, x):
        """ A function to transform the state for loss computation
        """
        return jnp.concatenate([x[...,0:1], x[...,1:2], jnp.sin(x[...,2:3]), jnp.cos(x[...,2:3]), x[...,3:4]], axis=-1)
    

    def compositional_drift(self, x, u, extra_args=None):
        """ The drift function of the SDE.
        """
        # Extract the state and control
        _, xdot, theta_val, theta_dot = x
        sin_theta_sc, cos_theta_sc = jnp.sin(theta_val), jnp.cos(theta_val)
        # Scaled state for NNs input
        x_sc, xdot_sc, theta_sc, theta_dot_sc = x / self.state_scaling
        uval = u[0]

        # Scaling factor
        nn_out_scale = jnp.array(self.params['nn_out_scale'])

        # If side information is not included
        if not self.params.get('side_info', False):
            # Scaled state
            xdotdot, theta_dotdot = nn_out_scale * self.residual_nn(jnp.array([sin_theta_sc, cos_theta_sc, theta_dot_sc, uval]))
            return jnp.array([xdot, xdotdot, theta_dot, theta_dotdot])
    
    def init_residual_networks(self):
        """Initialize the residual networks.
        """
        if not self.params.get('side_info', False):
            assert 'residual_forces' in self.params, 'The residual network must be defined in the yaml file'
            _act_fn = self.params['residual_forces']['activation_fn']
            init_value = self.params['residual_forces'].get('init_value', 1e-3) # Small non-zero value
            self.residual_nn = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 2],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                                name = 'res_forces')
            return

    
############# SET OF FUNCTIONS TO TRAIN THE MODEL #####################
def load_predictor_function(learned_params_dir, prior_dist=False, nonoise=False, modified_params ={}, 
                            return_control=False, return_time_steps=False):
    """ Create a function to sample from the prior distribution or
        to sample from the posterior distribution
        Args:
            learned_params_dir (str): Directory where the learned parameters are stored
            prior_dist (bool): If True, the function will sample from the prior distribution
            nonoise (bool): If True, the function will return a function without diffusion term
            modified_params (dict): Dictionary of parameters to modify
        Returns:
            function: Function that can be used to sample from the prior or posterior distribution
    """
    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)

    # vehicle parameters
    _model_params = learned_params['nominal']

    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)

    # If prior distribution, set the diffusion to zero
    if prior_dist:
        # TODO: Remove ground effect if present
        # Remove the learned density function
        params_model.pop('diffusion_density_nn', None)

    # If no_noise
    if nonoise:
        params_model['noise_prior_params'] = [0] * len(params_model['noise_prior_params'])
    
    # Compute the timestep of the model the extract the time evolution starting t0 = 0
    time_steps = compute_timesteps(params_model)
    time_evol = np.array([0] + jnp.cumsum(time_steps).tolist())

    # Create the model
    _prior_params, m_sampling = create_sampling_fn(params_model, sde_constr=CartPoleSDE)

    _sde_learned = _prior_params if prior_dist else _sde_learned
    if not return_time_steps:
        return lambda *x : m_sampling(_sde_learned, *x)[1] if not return_control else m_sampling(_sde_learned, *x)[1:]
    else:
        res_fn = lambda *x : m_sampling(_sde_learned, *x)[1] if not return_control else m_sampling(_sde_learned, *x)[1:]
        return (res_fn, time_evol)

def load_learned_diffusion(model_path, num_samples=1):
    """ Load the learned diffusion from the path
        Args:
            model_path (str): The path to the learned model
            num_samples (int, optional): The number of samples to generate
    """
    learned_params_dir = os.path.expanduser(model_path)
    # Load the pickle file
    with open(learned_params_dir, 'rb') as f:
        learned_params = pickle.load(f)
    # vehicle parameters
    _model_params = learned_params['nominal']
    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    # These parameters are the optimal learned parameters of the SDE
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])
    # Create the function to compute the diffusion
    _, m_diff_fn = create_diffusion_fn(_model_params, sde_constr=CartPoleSDE)

    @partial(jax.jit, static_argnums=(2,))
    def diffusion_fn(y, rng, net=False):
        m_rng = jax.random.split(rng, num_samples)
        # We use zero control input
        return jax.vmap(m_diff_fn, in_axes=(None, None, None, 0, None))(_sde_learned, y, jnp.array([0.0]), m_rng, net)
    
    return diffusion_fn

# Load a pkl file
def _load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def load_trajectory(filename):
    """Load a trajectory from a pkl file.

    Args:
        traj_dir (str): The directory where the trajectory is stored

    Returns:
        tuple: The trajectory data.
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    indata_path = os.path.expanduser((current_dir + '/my_data/' + filename))
    my_data = _load_pkl(indata_path)
    full_data = [{'y' : np.vstack([x[:,0], x[:,1], np.arctan2(x[:,2], x[:,3]), x[:,4]]).T, 'u' : u} for x, u in my_data]
    return full_data

def main_train_sde(yaml_cfg_file, output_file):
    """ Main function to train the SDE model.
    """
    # Load the yaml file
    cfg_train = load_yaml(yaml_cfg_file)

    # Obtain the path to the log data
    logs_dir = cfg_train['data_dir']
    if type(logs_dir) != list:
        logs_dir = [logs_dir]
    full_data = []
    for log_dir in logs_dir:
        full_data += load_trajectory(log_dir)
    
    # Split the data into train and test according to the ratio in the yaml file
    ratio_test = cfg_train['ratio_test']
    ratio_seed = cfg_train['ratio_seed']
    # Obtain the number of testing trajectories and make sure it is always greater than 1 and less than the total number of trajectories
    num_test_traj = max(1, min(int(ratio_test * len(full_data)), len(full_data)))
    np.random.seed(ratio_seed)
    # Pick indexes for the test data
    test_idx = np.random.choice(len(full_data), num_test_traj, replace=False)
    # Remove the test data from the full data if requested
    if cfg_train.get('remove_test_data', False):
        full_data = [full_data[i] for i in range(len(full_data)) if i not in test_idx]
    # Extract the test data
    test_data = [full_data[i] for i in test_idx]
    train_data = full_data
    print('Number of total trajectories: {}'.format(len(full_data)))
    print('Number of testing trajectories: {}'.format(len(test_data)))
    print('Number of training trajectories: {}'.format(len(train_data)))

    # Check if the control input match the model
    assert cfg_train['model']['n_u'] == train_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_u'] == test_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_y'] == train_data[0]['y'].shape[-1], 'The state dimension does not match the model'
    assert cfg_train['model']['n_y'] == test_data[0]['y'].shape[-1], 'The state dimension does not match the model'

    # Print the maximum in absolute value of the full data
    print('Maximum in absolute value of each state of the full data')
    print(np.max(np.abs(np.concatenate([x['y'] for x in train_data])), axis=0))

    if 'data_state_scaling' in cfg_train['model']:
        cfg_train['model']['state_scaling'] = list(np.max(np.abs(np.concatenate([x['y'] for x in full_data])), axis=0))
        print('State scaling is set to {}'.format(cfg_train['model']['state_scaling']))
    
    # Vector field scaling factor from data
    if 'nn_out_scaling' in cfg_train['model']:
        _grad_data = np.abs([ np.gradient(x['y'][:,[1,-1]], cfg_train['sde_loss']['data_stepsize'], axis=0) for x in full_data])
        _grad_data = np.max(np.max(_grad_data, axis=0), axis=0)
        print('Gradient on states: ', _grad_data)
        cfg_train['model']['nn_out_scale'] = list(_grad_data)
    cfg_train['model']['nn_out_scale'] = cfg_train['model'].get('nn_out_scale', [1.0, 1.0])
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.expanduser(current_dir + '/my_models/' + output_file)

    # Train the model
    train_model(cfg_train, train_data, test_data, output_file, CartPoleSDE)

    
if __name__ == '__main__':
    from sde4mbrl.train_sde import train_model

    import argparse

    # Argument parser
    parser = argparse.ArgumentParser(description='Train the SDE model')
    parser.add_argument('--fun', type=str, default='train_sde', help='Path to the yaml training configuration file')
    parser.add_argument('--cfg', type=str, default='cartpole_sde.yaml', help='Path to the yaml training configuration file')
    parser.add_argument('--out', type=str, default='cartpole', help='Name of the output file')
    # Parse the arguments
    args = parser.parse_args()
    if args.fun == 'train_sde':
        # Call the main function
        main_train_sde(args.cfg, args.out)