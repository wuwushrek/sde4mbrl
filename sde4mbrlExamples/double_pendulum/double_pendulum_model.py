import jax
import jax.numpy as jnp

import numpy as np

from sde4mbrl.nsde import ControlledSDE, create_sampling_fn
from sde4mbrl.utils import load_yaml, apply_fn_to_allleaf, update_params
from sde4mbrl.train_sde import train_model

import haiku as hk

import os
import pickle

from tqdm.auto import tqdm
from functools import partial

import yaml


class DoublePendulum(ControlledSDE):
    """ An SDE and ODE model of the double pendulum.
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
    
    def vector_field(self, x, u, f1=0.0, f2=0.0):
        """Compute the dynamics of the double pendulum

            Args:
                x (jax.numpy.ndarray): The state of the system.
                u (jax.numpy.ndarray): The input of the system.
                g1 (float, optional): An unknown term on the pendulum dynamics
                g2 (float, optional): An unknown term on the pendulum dynamics
            Returns:
                jax.numpy.ndarray: The vector field of the system.
        """
        # Get the parameters of the system
        m1, m2, l1, l2, gravity = self.init_params['m1'], self.init_params['m2'], self.init_params['l1'], self.init_params['l2'], self.init_params['gravity']
        # Extract the state
        t1, t2, w1, w2 = x
        # Check if groundtruth
        if self.params['ground_truth']:
            # Compute f1 and f2 from the groundtruth
            f1 = -(l2 / l1) * (m2 / (m1 + m2)) * (w2**2) * jnp.sin(t1 - t2) - (gravity / l1) * jnp.sin(t1)
            f2 = (l1 / l2) * (w1**2) * jnp.sin(t1 - t2) - (gravity / l2) * jnp.sin(t2)

        # Compute the vector field
        a1 = (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(t1 - t2)
        a2 = (l1 / l2) * jnp.cos(t1 - t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)

        return jnp.array([w1, w2, g1, g2])

    
    def init_residual_networks(self):
        """Initialize the residual networks.
        """

        if 'residual_forces' not in self.params:
            pass

        residual_type = self.params['residual_forces']['type']
        if residual_type == 'dnn':
            # Get the residual network parameters
            _act_fn = self.params['residual_forces']['activation_fn']
            self.residual_forces = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 2],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
                                                name = 'Fres')
            self.residual = lambda x, u: self.residual_forces(x)
            return
        # elif residual_type == 'dnn_sym':

        #     # [TODO] This might not converge so to check
        #     # Create the side information with symmetry information
        #     _act_fn = self.params['residual_forces']['activation_fn']
        #     self.residual_forces = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 2],
        #                                         activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
        #                                         w_init=hk.initializers.RandomUniform(-1e-3, 1e-3),
        #                                         name = 'Fres')
        #     def residual(x, _u):
        #         # Absoulte value of the state
        #         xabs = jnp.abs(x)
        #         res_val = self.residual_forces(xabs)
        #         # We need to add the sign of the first two states
        #         return 

    def prior_drift(self, x, u, extra_args=None):
        """Drift of the prior dynamics
            This is the approximate and undamped dynamics of the system
        """

        if self.params['ground_truth']:
            return self.vector_field(x, u)

        # If there is no side information about the evolution, the prior is set to OU process
        if not self.params['include_side_info']:
            return jnp.array([x[2], x[3], -x[2], -x[3]]) # OU process for the second order derivative

        # We set the residual to zero by default
        return self.vector_field(x, u, f1=x[1]-x[0], f2=x[0]-x[1])
    
    def posterior_drift(self, x, u, extra_args=None):
        """Drift of the posterior dynamics
        """
        if self.params['ground_truth']:
            return self.vector_field(x, u)

        # Check if there is side information about the evolution
        if not self.params['include_side_info']:
            # Compute the residual
            Fres = self.residual(x, u)
            return jnp.array([x[2], x[3], Fres[0], Fres[1]])
        
        # Compute the residual
        xnew = jnp.array([x[0], x[1], jnp.abs(x[2]), jnp.abs(x[3])])
        Fres = self.residual(xnew, u)

        return self.vector_field(x, u, f1=Fres[0], f2=Fres[1])
    
    def prior_diffusion(self, x, extra_args=None):
        """Diffusion term of the prior dynamics
        """
        # We set the diffusion to zero, diagonal matrix by default
        zero_diffusion = jnp.zeros((self.params['n_x'],))

        if self.params['diffusion_type'] == 'zero' or self.params['diffusion_type'] == 'nonoise':
            return zero_diffusion
        
        if self.params['diffusion_type'] == 'constant':
            amp_noise = self.params['amp_noise'] if self.params['include_side_info'] else self.params['nosi_amp_noise']
            return jnp.array(amp_noise)

        raise ValueError('Unknown diffusion type')


def load_predictor_function(learned_params_dir, prior_dist=False, modified_params ={}):
    """ Create a function to sample from the prior distribution or
        to sample from the posterior distribution
        
        Args:
            learned_params_dir (str): The path to the learned parameters or to the nominal yaml model file
            prior_dist (bool, optional): If True, the function will sample from the prior distribution
            nonoise (bool, optional): If True, the function will sample from the prior distribution without noise
            modified_params (dict, optional): A dictionary of parameters to modify the default parameters
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
        # of jax accross different devices
        # These parameters are the optimal learned parameters of the SDE
        _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)

    # Create the model
    _, m_sampling = create_sampling_fn(params_model, sde_constr=DoublePendulum, prior_sampling=prior_dist)

    print(params_model)

    return lambda *x : m_sampling(_sde_learned, *x)

def load_learned_model(model_path, horizon=1, num_samples=1, ufun=None, prior_dist=False):
    """ Load the learned model from the path
        Args:
            model_path (str): The path to the learned model
    """
    modified_params = {'horizon': horizon, 'num_particles': num_samples}
    # Load the model
    pred_fn = load_predictor_function(model_path, modified_params=modified_params, prior_dist=prior_dist)

    # Define the control function
    if ufun is None:
        ufun = lambda *x : jnp.array([0.0])
    
    @partial(jax.jit, backend='cpu')
    def sampling_jit(x, rng):
        xevol, _ = pred_fn(x, ufun, rng)
        return xevol

    return sampling_jit

def load_data_generator(model_dir, noise_info={}, horizon=1, ufun=None):
    """ Create a function to generate data from the prior distribution or
        to generate data from the posterior distribution

        Args:
            model_dir (str): to the nominal yaml model file
            noise_info (dict, optional): A dictionary of parameters for the noise in the data generator
                A key 'process_noise' would be used to specify noise amplitude in the diffusion term, quite similar to the 'amp_noise' in the model
                A key 'measurement_noise' would be used to specify noise term to add when data at the end of the sampling when the data is measured
            horizon (int, optional): The horizon of the data generator
    """
    # Construct the modified params dictionary
    modified_params = {}
    modified_params['include_side_info'] = False
    modified_params['num_particles'] = 1
    modified_params['horizon'] = horizon
    modified_params['ground_truth'] = True
    modified_params['control'] = False if ufun is None else True
    modified_params['diffusion_type'] = 'zero' if 'process_noise' not in noise_info else 'constant'
    modified_params['nosi_amp_noise'] = noise_info['process_noise'] if 'process_noise' in noise_info else [0.0, 0.0, 0.0, 0.0]

    # Define the control function
    if ufun is None:
        ufun = lambda *x : jnp.array([0.0])
    
    # Create the model
    sampling_fn = load_predictor_function(model_dir, prior_dist=False, modified_params=modified_params)
    # Jit the sampling function on the CPU -> We only extract the first sample

    @partial(jax.jit, backend='cpu')
    def sampling_fn_jit(x, rng):
        xevol, uevol = sampling_fn(x, ufun, rng)
        return xevol[0], uevol[0] # We only extract the first sample

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
            x0 = np.random.uniform(x_lb, x_ub)
            # Split the key for next iteration
            key, next_key = jax.random.split(key)
            # Compute next state
            x_evol, u_evol = sampling_fn_jit(x0, next_key)
            x_evol = np.array(x_evol) # Take only one particle
            u_evol = np.array(u_evol) # Take only one particle
            # Add noise to the measurement if enabled except for the first state
            if 'measurement_noise' in noise_info:
                x_evol += np.random.normal(size=x_evol.shape) * noise_info['measurement_noise']
                # x_evol[1:] += np.random.normal(size=x_evol[1:].shape) * noise_info['measurement_noise']
            # Add the data to the list
            data.append((x_evol, u_evol))
        
        return data
    
    return sampling_fn_jit, data_generator

def train_sde(yaml_cfg_file, model_type, train_data, test_data, output_file=None):
    """ Main function to train the SDE

        Args:
            yaml_cfg_file (str): The path to the yaml configuration file
            model_type (str): The type of model to use. Choice between 'node_bboxes', 'nesde_bboxes', 'node_phys', 'nesde_phys'
            train_data (list): The list of training data
            test_data (list): The list of test data
            output_file (str, optional): The name of the output file
    """
    # Load the yaml file
    cfg_train = load_yaml(yaml_cfg_file)

    # Get the horizon from the dataset transition
    # not_one_step_split = 'nesde' in model_type
    # not_one_step_split = True
    # cfg_train['sde_loss']['horizon'] = train_data[0][1].shape[0] if not_one_step_split else 1
    
    # Specify the dimension of the problem
    cfg_train['model']['n_x']  = 4
    cfg_train['model']['n_y']  = 4
    cfg_train['model']['n_u']  = 1

    # Modify the model according to the type
    if model_type == 'node_bboxes':
        # Black box model neural ode model
        cfg_train['model']['diffusion_type'] = 'zero' # nonoise
        cfg_train['model']['include_side_info'] = False
        cfg_train['model']['ground_truth'] = False
        cfg_train['sde_loss']['num_particles'] = 1
    elif model_type == 'nesde_bboxes':
        # Black box model neural ode model
        assert cfg_train['model']['diffusion_type'] != 'zero' and cfg_train['model']['diffusion_type'] != 'nonoise', "Diffusion type cannot be zero or nonoise for SDE"
        cfg_train['model']['include_side_info'] = False
        cfg_train['model']['ground_truth'] = False
    elif model_type == 'node_phys':
        # Physical model neural ode model
        cfg_train['model']['diffusion_type'] = 'zero'
        cfg_train['model']['include_side_info'] = True
        cfg_train['model']['ground_truth'] = False
        cfg_train['sde_loss']['num_particles'] = 1
    elif model_type == 'nesde_phys':
        # Physical model neural ode model
        assert cfg_train['model']['diffusion_type'] != 'zero' and cfg_train['model']['diffusion_type'] != 'nonoise', "Diffusion type cannot be zero or nonoise for SDE"
        cfg_train['model']['include_side_info'] = True
        cfg_train['model']['ground_truth'] = False
    else:
        raise ValueError("Unknown model type: " + model_type)
        
    # # Build the full data
    # if not_one_step_split:
    #     is_data_trajectory = False
    #     full_data = { 'y' : np.array([x for x, _ in train_data]), 'u' : np.array([u for _, u in train_data])}
    #     test_traj_data = { 'y' : np.array([x for x, _ in test_data]), 'u' : np.array([u for _, u in test_data])}
    # else:
    #     is_data_trajectory = True
    #     full_data = [{ 'y' : x, 'u' : u} for x, u in train_data]
    #     test_traj_data = [{ 'y' : x, 'u' : u} for x, u in test_data]
    
    full_data = [{ 'y' : x, 'u' : u} for x, u in train_data]
    test_traj_data = [{ 'y' : x, 'u' : u} for x, u in test_data]

    # print(full_data['y'].shape)
    # print(full_data['u'].shape)

    # Build the test data
    # test_traj_data = { 'y' : np.array([x for x, _ in test_data]), 'u' : np.array([u for _, u in test_data])}

    # Create the output file
    if output_file is None:
        # Create the file based on the current date and time
        # output_file = 'mass_spring_' + datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = model_type
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = current_dir + '/my_models/' + output_file

    # TODO: Improve this stopping criteria
    def _improv_cond(opt_var, test_res, train_res, itr_count):
        """Improvement condition
        """
        optTotalLoss = opt_var['totalLoss']
        train_total_loss = train_res['totalLoss'] * cfg_train['sde_training']['coeff_improv_training_data']
        test_total_loss = test_res['totalLoss']
        if optTotalLoss > train_total_loss + test_total_loss or itr_count <= 1:
            test_res['totalLoss'] = train_total_loss + test_total_loss
            return True
        else:
            return False
        # return opt_var['totalLoss'] > test_res['totalLoss']

    # Train the model
    train_model(cfg_train, full_data, test_traj_data, output_file, _improv_cond, DoublePendulum)


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

    _, trans_generator = load_data_generator(model_dir, noise_info, horizon, uFun)

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
                horizon = _cfg_t['horizon'], seed_trans= _cfg_t['test_seed'], regenerate=True
    )

# Load a pkl file
def _load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_models_on_dataset(model_dir, model_type, specific_data=None):
    """ Train the given model type: It could be one of 'node_bboxes', 'nesde_bboxes', 'node_phys', 'nesde_phys'.
        The training dataset and testing dataset is extracted from my_data folder

        Args:
            model_dir (str): The path for the groundtruth model definition
            model_type (str): The type of model to train
    """
    current_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data'
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
        train_sde(model_dir, model_type, _data, testData, outputNames[_k])

if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Double Pendulum, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='gen_traj', help='The function to run')
    parser.add_argument('--model_dir', type=str, default='double_pendulum.yaml', help='The model configuration and groundtruth file')
    parser.add_argument('--data_gen_cfg', type=str, default='data_generation.yaml', help='The data generation configuration file')
    parser.add_argument('--model_type', type=str, default='node_bboxes', help='The model train: node_bboxes, nesde_bboxes, node_phys, nesde_phys')
    parser.add_argument('--data', type=str, default='', help='Specific data file to train on')

    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'gen_traj':
        gen_traj_yaml(args.model_dir, args.data_gen_cfg)
    
    if args.fun == 'train':
        train_models_on_dataset(args.model_dir, args.model_type, None if args.data == '' else args.data)
    
