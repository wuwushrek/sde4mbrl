import jax

import jax.numpy as jnp

import numpy as np

from sde4mbrl.nsde import ControlledSDE, create_sampling_fn, create_diffusion_fn, compute_timesteps
from sde4mbrl.utils import load_yaml, apply_fn_to_allleaf, update_params
from sde4mbrl.train_sde import train_model

import haiku as hk

import os
import pickle

from tqdm.auto import tqdm
from functools import partial

import yaml

class MassSpringDamper(ControlledSDE):
    """ An SDE and ODE model of the mass spring damper system.
        The parameters of the model are usually defined in a yaml file.
    """
    def __init__(self, params, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name=name)

        # Parameters initialization values -> Values for HK PARAMETERS initialization
        self.init_params = params['init_params']

        # Initialization of the residual networks
        # This function setup the parameters of the unknown neural networks and the residual network
        self.init_residual_networks()
    
    def vector_field(self, x, u, Fres=0.0):
        """Compute the dynamics of the system

        Args:
            x (jax.numpy.ndarray): The state of the system.
            u (jax.numpy.ndarray): The input of the system.
            Fres (jax.numpy.ndarray, optional): The residual force.

        Returns:
            jax.numpy.ndarray: The vector field of the system.
        """
        # Ge
        # t the parameters of the system
        m = self.init_params['mass']
        k = self.init_params['kq_coeff']
        b = self.init_params['bqdot_coeff']

        # Check if control is enabled
        if not self.params['control']:
            u = jnp.array([0.0])

        # Compute the vector field
        if self.params.get('ground_truth',False):
            xdot = jnp.array([x[1], (u[0] - k*x[0] - b*x[1])/m])
        else: # Vector field with side knowledge and residual forces
            xdot = jnp.array([x[1], (u[0] - k*x[0] + Fres)/m])

        return xdot

    def prior_diffusion(self, x, u, extra_args=None):
        """ User-Specified Maximum diffusion outside of the training dataset
        """
        # Set the prior to a constant noise as defined in the yaml file
        return jnp.array(self.params['noise_prior_params'])


    def compositional_drift(self, x, u, extra_args=None):
        """ Compute the drift of the system. Check nsde.py for more details
        """
        # In the ground truth case, we do not need to compute the residual
        if self.params.get('ground_truth',False):
            return self.vector_field(x, u)
        
        # If side information is included, the residual structure is constrained
        if self.params.get('include_side_info', False):
            Fres = self.residual(x[1:], u) # The residual is only a function of the velocity
            return self.vector_field(x, u, Fres)
        
        # If side information is not included, the residual structure is unconstrained
        Fres = self.residual(x, u) # The residual is a function of the state and the control
        return jnp.array([x[1], Fres])

    
    def init_encoder(self):
        """ Initialize the encoder -> For this example we essentially use the identity encoder
        """
        _encoder_type = self.params.get('encoder_type', 'identity')
        if _encoder_type == 'ground_truth' or _encoder_type == 'identity':
            self.obs2state = self.identity_obs2state
            self.state2obs = self.identity_state2obs
        else:
            raise ValueError('Unknown encoder type')
    
    def init_residual_networks(self):
        """Initialize the residual neural networks
        """
        if 'residual_forces' not in self.params:
            return

        # What is the parameterization of the residual forces?
        residual_type = self.params['residual_forces']['type']
        init_value = self.params['residual_forces'].get('init_value', 0.001)

        if residual_type == 'dnn':

            # Get the residual network parameters
            _act_fn = self.params['residual_forces']['activation_fn']
            self.residual_forces = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 1],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                                name = 'Fres')
            
            # Make sure if prior is asked, the residual is zero
            # Here prior is used for ground truth model (no residual)
            self.residual = lambda x, _: self.residual_forces(x)[0] if not self.params.get('prior', False) else 0.0
        
        else:
            raise NotImplementedError


def load_predictor_function(learned_params_dir, prior_dist=False, modified_params ={}):
    """ Create a function to sample the learned SDE distribution
        
        Args:
            learned_params_dir (str): The path to the learned parameters or to the nominal yaml model file
            prior_dist (bool, optional): If True, the function will return the prior knowledge 
                                        of the system (residual is zero) + zero diffusion
            modified_params (dict, optional): A dictionary of parameters to modify the default parameters
        
        Returns:
            function: A jitted function to sample from the learned SDE distribution
            time_evol: The array of time points at which the SDE is sampled

    """
    # Expand user name
    learned_params_dir = os.path.expanduser(learned_params_dir)

    # Check if the file has yaml extension
    if learned_params_dir.endswith('.yaml'): # yaml file are usually used for the nominal parameters
        _model_params = load_yaml(learned_params_dir)['model']
        _sde_learned = {}

    elif learned_params_dir.endswith('.pkl'):
        # Load the pickle file
        with open(learned_params_dir, 'rb') as f:
            learned_params = pickle.load(f)
        # vehicle parameters
        _model_params = learned_params['nominal']
        # SDE learned parameters -> All information are saved using numpy array to facilicate portability
        # of jax accross different devices. These parameters are the optimal learned parameters of the SDE after training
        _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)
    if prior_dist:
        # Set the key prior to True
        params_model['prior'] = True
        # Remove the density NN parameters so that the noise is given by the prior diffusion only
        params_model.pop('diffusion_density_nn', None)

    # Create the model
    _prior_params, m_sampling = create_sampling_fn(params_model, sde_constr=MassSpringDamper)

    # Compute the timestep of the model the extract the time evolution starting t0 = 0
    time_steps = compute_timesteps(params_model)
    time_evol = np.array([0] + jnp.cumsum(time_steps).tolist())

    # Do not use the sde_learned params when sampling from the prior distribution
    _sde_learned = _prior_params if prior_dist else _sde_learned

    # Print the model parameters
    print('Model config parameters:\n', params_model)
    print('\nLearned model parameters:\n', _sde_learned)

    return lambda *x : m_sampling(_sde_learned, *x), time_evol

def load_learned_model(model_path, horizon=1, num_samples=1, ufun=None, prior_dist=False):
    """ Load the learned model from the path

        Args:
            model_path (str): The path to the learned model
            horizon (int, optional): The horizon of the model
            num_samples (int, optional): The number of samples to generate
            ufun (function, optional): The control function
            prior_dist (bool, optional): If True, the function will sample from the prior knowledge of the system + prior diffusion

        Returns:
            sampling_jit (function): A cpu jitted function to sample from the learned model
                sampling_jit(y, rng) -> yevol
            
            _time_evol (np.array): The time evolution of the model
        
    """
    modified_params = {'horizon': horizon, 'num_particles': num_samples}
    # Load the model
    pred_fn, _time_evol = load_predictor_function(model_path, modified_params=modified_params, prior_dist=prior_dist)

    # Define the control function
    if ufun is None:
        ufun = lambda *x : 0.0
    
    @partial(jax.jit, backend='cpu')
    def sampling_jit(y, rng):
        _, yevol, _ = pred_fn(y, ufun, rng)
        return yevol

    return sampling_jit, _time_evol

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
    _, m_diff_fn = create_diffusion_fn(_model_params, sde_constr=MassSpringDamper)

    @partial(jax.jit, static_argnums=(2,))
    def diffusion_fn(y, rng, net=False):
        m_rng = jax.random.split(rng, num_samples)
        # We use zero control input
        return jax.vmap(m_diff_fn, in_axes=(None, None, None, 0, None))(_sde_learned, y, jnp.array([0.0]), m_rng, net)
    
    return diffusion_fn

def load_data_generator(model_dir, noise_info={}, horizon=1, ufun=None):
    """ Create a function to generate data from the prior distribution or
        to generate data from the posterior distribution

        Args:
            model_dir (str): to the nominal yaml model file
            noise_info (dict, optional): A dictionary of parameters for the noise in the data generator
                A key 'process_noise' would be used to specify noise amplitude in the diffusion term, quite similar to the 'amp_noise' in the model
                A key 'measurement_noise' would be used to specify noise term to add when data at the end of the sampling when the data is measured
            horizon (int, optional): The horizon of the data generator
            ufun (function, optional): The control function
        
        Returns:
            sampling_fn (function): A jitted function to generate trajectories given initial y0 and rng
                sampling_fn(y0, rng) -> yevol, uevol
            data_generator (function): A function to randomly generate data trajectories from uniformly sample initial conditions
                data_generator(x_lb, x_ub, n_trans, seed) -> [(yevol, uevol),....]

    """
    # Construct the modified params dictionary
    modified_params = {}
    modified_params['num_particles'] = 1
    modified_params['horizon'] = horizon
    modified_params['ground_truth'] = True
    modified_params['control'] = False if ufun is None else True
    modified_params['noise_prior_params'] = [0.0, 0.0] if 'process_noise' not in noise_info else noise_info['process_noise']
    
    # Create the model
    sampling_fn, _time_evol = load_predictor_function(model_dir, prior_dist=True, modified_params=modified_params)
    # Jit the sampling function on the CPU -> We only extract the first sample

    # Define the control function
    if ufun is None:
        ufun = lambda *x : jnp.array([0.0])

    @partial(jax.jit, backend='cpu')
    def sampling_fn_jit(y, rng):
        _, yevol, uevol = sampling_fn(y, ufun, rng)
        return yevol[0], uevol[0] # We only extract the first sample

    def data_generator(x_lb, x_ub, n_trans, seed=0):
        """ Generate data from the ground truth model
            
            Args:
                x_lb (list): The lower bound of the initial state
                x_ub (list): The upper bound of the initial state
                n_trans (int): The number of transitions to generate
                seed (int, optional): The seed for the random number generator
            
            Returns:
                List of tuples of the form (x, u, x_next) where u and x_next has same first dimension and could be more than 1 for multi-step prediction
        """
        # Set the seed
        np.random.seed(seed)

        # Buuild a PRNG key
        key = jax.random.PRNGKey(seed)

        # Iterate over the number of transitions to generate
        data = []

        for i in tqdm(range(n_trans)):
            # Sample the initial state with numpy
            y0 = np.random.uniform(x_lb, x_ub)

            # Split the key for next iteration
            key, next_key = jax.random.split(key)

            # Compute next state
            y_evol, u_evol = sampling_fn_jit(y0, next_key)
            y_evol = np.array(y_evol) # Take only one particle
            u_evol = np.array(u_evol) # Take only one particle

            # Add noise to the measurement if enabled except for the first state
            if 'measurement_noise' in noise_info:
                y_evol += np.random.normal(size=y_evol.shape) * noise_info['measurement_noise']
            # Add the data to the list
            data.append((y_evol, u_evol))
        
        return data
    
    return sampling_fn_jit, data_generator, _time_evol


def train_sde(yaml_cfg_file, model_type, train_data, test_data, output_file=None, modified_params={}):
    """ Main function to train the SDE

        Args:
            yaml_cfg_file (str): The path to the yaml configuration file
            model_type (str): The type of model to use.
            train_data (list): The list of training data
            test_data (list): The list of test data
            output_file (str, optional): The name of the output file
    """
    # Load the yaml file
    cfg_train = load_yaml(yaml_cfg_file)
    
    # Specify the dimension of the problem
    cfg_train['model']['n_x']  = 2
    cfg_train['model']['n_y']  = 2
    cfg_train['model']['n_u']  = 1

    # Modify the model according to the type [neural ODE, neural SDE, neural ODE with side info, neural SDE with side info]
    if 'node' in model_type:
        # We are dealing with training a neural ode model -> No noise and no learning of density
        cfg_train['model']['include_side_info'] = 'phys' in model_type
        cfg_train['model']['noise_prior_params'] = [0.0, 0.0]
        cfg_train['model'].pop('diffusion_density_nn', None) # Remove the diffusion density network
        cfg_train['sde_loss']['num_particles'] = 1 # No need for multiple particles
        cfg_train['sde_loss']['num_particles_test'] = 1 
        cfg_train['sde_loss'].pop('density_loss', None) # Remove the density loss
        cfg_train['sde_loss'].pop('num_sample2consider', None)
    elif 'nesde' in model_type:
        # We are dealing with training a neural sde model
        cfg_train['model']['include_side_info'] = 'phys' in model_type
    else:
        raise ValueError("Unknown model type: " + model_type)

    # Add the modifed parameters
    if len(modified_params) > 0:
        cfg_train = update_params(cfg_train, modified_params)
        
    full_data = [{ 'y' : x, 'u' : u} for x, u in train_data]
    test_traj_data = [{ 'y' : x, 'u' : u} for x, u in test_data]

    # Create the output file
    if output_file is None:
        # Create the file based on the current date and time
        # output_file = 'mass_spring_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = model_type
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = current_dir + '/my_models/' + output_file

    # Train the model
    train_model(cfg_train, full_data, test_traj_data, output_file, MassSpringDamper)


################# Bunch of commanda line functions to train the models and generate data #################

def gen_traj(model_dir, outfile, num_trans, domain_lb, domain_ub, pnoise, mnoise, horizon, seed_trans, regenerate):
    """ Generate a set of transitions from the groundtruth model and a noise profile, then save them to a file"""

    # Get the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # print(current_dir)
    outfile_data_path = current_dir + '/my_data/'+outfile+'.pkl'
    # print(outfile_data_path)
    outfile_data_config_path = current_dir + '/my_data/'+outfile+'_config.yaml'

    # Check if the data already exists
    if not regenerate and os.path.exists(outfile_data_config_path):
        print("Data already exists. Skipping generation")
        return

    # Should we add noise after integrating or during integration?
    noise_info = {}

    # Noise is disabled by default
    if pnoise is not None:
        noise_info["process_noise"] = pnoise
    
    if mnoise is not None:
        noise_info["measurement_noise"] = mnoise

    # Function for the control input
    uFun = None # No control input

    _, trans_generator, _ = load_data_generator(model_dir, noise_info, horizon, uFun)

    # Generate the data
    trans_data = trans_generator(domain_lb, domain_ub, num_trans, seed_trans)

    # Create a dictionary of configuration
    config_dict = {
        'domain_lb' : domain_lb,
        'domain_ub' : domain_ub,
        'pnoise' : pnoise,
        'mnoise' : mnoise,
        'horizon' : horizon,
        'seed_trans' : seed_trans,
        'model_dir' : model_dir,
        'outfile' : outfile_data_path,
        'num_trans' : num_trans,
        'name' : outfile
    }

    # Save the yaml file with the configuration
    with open(outfile_data_config_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
    
    # Pickle the data
    with open(outfile_data_path, 'wb') as f:
        pickle.dump(trans_data, f)
    

def gen_traj_yaml(model_dir, data_gen_dir):
    """ Generate transitions data from the model as specified by data_gen_dir

        Args:
            model_dir (str): The path for the groundtruth model definition
            datagen_dir (str): The path for the data generation configuration
    """
    # Load the data generation configuration
    cfg_data_gen = load_yaml(data_gen_dir)
    noise_settings = cfg_data_gen['noise_settings']
    domain_settings = cfg_data_gen['sampling_domain_settings']
    seed_trans  = cfg_data_gen['seed_trans']
    num_samples = cfg_data_gen['num_samples'] if isinstance(cfg_data_gen['num_samples'], list) else [cfg_data_gen['num_samples']]

    # Now create a list of all the possible configurations for the data generation
    data_gen_cfgs = []
    for num_sample in num_samples:
        for knoise, _noise in noise_settings.items():
            for kdomain, _domain in domain_settings.items():
                # Create an output file name based on the configuration
                output_file = 'MSD_{}_{}_{}'.format(knoise, kdomain, num_sample)
                data_gen_cfgs.append(
                    {
                        'outfile' : output_file,
                        'num_trans' : num_sample,
                        'domain_lb' : _domain['lb'],
                        'domain_ub' : _domain['ub'],
                        'pnoise' : _noise.get('pnoise', None),
                        'mnoise' : _noise.get('mnoise', None),
                        'horizon' : cfg_data_gen.get('horizon', 1),
                        'seed_trans': seed_trans,
                        'regenerate': cfg_data_gen.get('regenerate', True)
                    }
                )

    # Now generate the data
    for _k, _cfg in enumerate(data_gen_cfgs):
        print ("Generating data: {}/{}".format(_k+1, len(data_gen_cfgs)))
        print(_cfg)
        gen_traj(model_dir, **_cfg)
    
    # Generate the test data
    print("Generating test data")
    _cfg_t = cfg_data_gen['test_data']
    print(_cfg_t)

    gen_traj(model_dir, outfile='MSD_TestData', num_trans = _cfg_t['num_samples'], domain_lb = _cfg_t['domain_lb'], 
                domain_ub = _cfg_t['domain_ub'], pnoise = _cfg_t['pnoise'], mnoise = _cfg_t['mnoise'], 
                horizon = cfg_data_gen.get('horizon', 1), seed_trans= _cfg_t['test_seed'], regenerate=True
    )

# Load a pkl file
def _load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_models_on_dataset(model_dir, model_type, specific_data=None, modified_params={}):
    """ Train the given model type: It could be one of 'node_bboxes', 'nesde_bboxes', 'node_phys', 'nesde_phys'.
        The training dataset and testing dataset is extracted from my_data folder

        Args:
            model_dir (str): The path for the groundtruth model definition
            model_type (str): The type of model to train
    """
    current_dir = 'my_data/'
    # Extract all the files in the current directory that ends with .yaml
    files = [f for f in os.listdir(current_dir) if f.endswith('.yaml')]

    # Save the test data in a separate variable
    testData = None

    # Save the list of training data
    trainDataList = []

    # Save the names of the output file
    outputNames = []

    for fname in files:
        dataCfg = load_yaml(current_dir + '/' + fname)

        # Get data file path
        data_file_path = dataCfg['outfile']
        data_name = dataCfg['name']
        if 'TestData' in data_name:
            testData = _load_pkl(data_file_path)
            continue
        
        if specific_data is not None and specific_data not in data_name:
            continue

        # Load the data
        m_data = _load_pkl(data_file_path)

        # Add the data to the list
        trainDataList.append(m_data)

        # Add the output name
        outputNames.append(model_type + '__' + data_name)
    
    # Now train the model on each of the data
    for _k, _data in enumerate(trainDataList):
        print("=====================================================================")
        print ("Training model: {}/{}".format(_k+1, len(trainDataList)))
        print(outputNames[_k])
        print("=====================================================================")
        train_sde(model_dir, model_type, _data, testData, outputNames[_k], modified_params=modified_params)

if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Mass Spring Damper Model, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='gen_traj', help='The function to run')
    parser.add_argument('--model_dir', type=str, default='mass_spring_damper.yaml', help='The model configuration and groundtruth file')
    parser.add_argument('--data_gen_cfg', type=str, default='data_generation.yaml', help='The data generation configuration file')
    parser.add_argument('--model_type', type=str, default='node_bboxes', help='The model train: node_bboxes, nesde_bboxes, node_phys, nesde_phys')
    parser.add_argument('--data', type=str, default='', help='Specific data file to train on')

    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'gen_traj':
        gen_traj_yaml(args.model_dir, args.data_gen_cfg)
    
    if args.fun == 'train':
        train_models_on_dataset(args.model_dir, args.model_type, None if args.data == '' else args.data)
    
