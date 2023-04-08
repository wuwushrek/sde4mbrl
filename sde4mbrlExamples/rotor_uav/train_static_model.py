import numpy as np

import jax
import jax.numpy as jnp

import haiku as hk
import optax

# from jaxopt import PolyakSGD
# from jaxopt import ArmijoSGD

from sde4mbrlExamples.rotor_uav.sde_rotor_model import SDERotorModel
from sde4mbrlExamples.rotor_uav.utils import  parse_ulog
from sde4mbrl.utils import apply_fn_to_allleaf, load_yaml

# This requires extra library to be installed
from sde4mbrlExamples.rotor_uav.smoothing_data import filter_data

import time, datetime
import collections

# Import yaml
import yaml
import argparse

import pickle

from jax.tree_util import tree_flatten
import os

import copy

from tqdm.auto import tqdm
import pandas as pd
    

def set_values_all_leaves(d, v):
    """Set the same value to all the leaves of a dictionary

    Args:
        d (TYPE): Description
        v (TYPE): Description

    Returns:
        TYPE: Description
    """
    d = copy.deepcopy(d)
    for k, _v in d.items():
        if isinstance(_v, collections.abc.Mapping):
            d[k] = set_values_all_leaves(_v, v)
        else:
            d[k] = v
    return d

def set_values_matching_keys(d, keys_d):
    """ Set the values of the keys of d that matches any key in keys
    """
    key_changed = set()
    d = copy.deepcopy(d)
    for k, val in keys_d.items():
        for kd in d.keys():
            if k in kd:
                d[kd] = val if not isinstance(d[kd], collections.abc.Mapping) else set_values_all_leaves(d[kd], val)
                key_changed.add(kd)
    return key_changed, d


def get_penalty_parameters(dict_params, dict_penalty, default_value):
    """Get the penalty parameters for the loss penalization
        This function assumes that dict_penalty is a flat dictionary

    Args:
        dict_params (TYPE): The dictionary of the parameters
        dict_penlaty (TYPE): A flat dictionary of the penalty parameters

    Returns:
        TYPE: Description
    """
    # penalty_params = {}
    # If some of the key matches --> directly set them

    key_changed, _dict_params = set_values_matching_keys(dict_params, dict_penalty)

    for k, v in _dict_params.items():
        if k in key_changed:
            continue
        if isinstance(v, collections.abc.Mapping):
            _dict_params[k] = get_penalty_parameters(v, dict_penalty, default_value)
        else:
            if default_value is not None:
                _dict_params[k] = default_value
    return _dict_params

def get_non_negative_params(dict_params, enforced_nonneg):
    """Get the parameters that are non negative

    Args:
        dict_params (TYPE): The dictionary of the parameters
        enforced_nonneg (TYPE): The list of the parameters that are non negative

    Returns:
        TYPE: Description
    """
    key_changed, non_neg_params = set_values_matching_keys(dict_params, enforced_nonneg)
    for k, v in dict_params.items():
        if k in key_changed:
            continue
        if isinstance(v, collections.abc.Mapping):
            non_neg_params[k] = get_non_negative_params(v, enforced_nonneg)
        else:
            non_neg_params[k] = False
    return non_neg_params

def get_all_leaf_dict(d):
    """Get all the leaf of a dictionary"""
    res_dict = {}
    for k, v in d.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, collections.abc.Mapping):
            res_dict = {**res_dict, **get_all_leaf_dict(v)}
        else:
            res_dict[k] = v 
    return res_dict


def evaluate_loss_fn(loss_fn, m_params, data_eval, test_batch_size):
    """Compute the metrics for evaluation accross the data set

    Args:
        loss_fn (TYPE): A loss function lambda m_params, data : scalar
        m_params (dict): The parameters of the neural network model
        data_eval (iterator): The dataset considered for the loss computation
        num_iter (int): The number of iteration over the data set

    Returns:
        TYPE: Returns loss metrics
    """
    result_dict ={}

    num_test_batches = data_eval['x'].shape[0] // test_batch_size
    # Iterate over the test batches
    for n_i in tqdm(range(num_test_batches), leave=False):
        # Get the batch
        batch = {k : v[n_i*test_batch_size:(n_i+1)*test_batch_size] for k, v in data_eval.items()}
        # Infer the next state values of the system
        curr_time = time.time()
        # Compute the loss
        lossval, extra_dict = loss_fn(m_params, batch)
        lossval.block_until_ready()

        diff_time  = time.time() - curr_time
        extra_dict = {**extra_dict, 'Pred. Time' : diff_time}

        if len(result_dict) == 0:
            result_dict = {_key : np.zeros(num_test_batches) for _key in extra_dict}

        # Save the data for logging
        for _key, v in extra_dict.items():
            result_dict[_key][n_i] = v

    return {_k : np.mean(v) for _k, v in result_dict.items()}


def init_data(log_dir, cutoff_freqs, force_filtering=False, zmin=0.1, mavg_dict={}):
    """Load the data from the ulog file and return the data as a dictionary.

    Args:
        dataset_dir (str): The path to the ulog file

    Returns:
        dict: The data dictionary
    """
    log_dir = os.path.expanduser(log_dir)
    # Extract the current directory without the filename
    log_dir_dir = log_dir[:log_dir.rfind('/')]
    # Extract the file name without the .ulog extension
    log_name = log_dir[log_dir.rfind('/')+1:].replace('.ulg','')

    # Check if a filtered version of the data already exists in the log directory
    if not force_filtering:
        # Check if the filtered data already exists
        try:
            with open(log_dir_dir + '/' + log_name + '_filtered.pkl', 'rb') as f:
                data = pickle.load(f)
                # Print that the data was loaded from the filtered file
                tqdm.write('Data loaded from filtered file')
                return data
        except:
            pass

    # Load the data from the ulog file
    tqdm.write('Loading data from the ulog file..')

    # In static condition we want to avoid ground effect
    outlier_cond = lambda d: d['z'] > zmin
    log_datas = parse_ulog(log_dir, outlier_cond=outlier_cond, mavg_dict=mavg_dict)

    if len(log_datas) == 0:
        return None

    # Ordered state names
    name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    name_controls = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
    _reduced_cutoff_freqs = {k : cutoff_freqs[k] for k in name_states}

    # Filter the data as described in filter_data_analysis.ipynb
    tqdm.write('Filtering the data...')
    xdot_list, xlist, ulist = [], [], []
    for log_data in log_datas:
        log_data_dot = filter_data(log_data, _reduced_cutoff_freqs, 
                        suffix_der='_dot', save_old_state=False, include_finite_diff = False, 
                        state_names=list(_reduced_cutoff_freqs.keys()))

        # Check if any nan values are present in the data
        for k, v in log_data_dot.items():
            if np.any(np.isnan(v)):
                raise ValueError('Nan values present in the data: {}'.format(k))

        # Check if any inf values are present in the data
        for k, v in log_data_dot.items():
            if np.any(np.isinf(v)):
                raise ValueError('Inf values present in the data: {}'.format(k))

        # Build the ndarray of derivative of states [smooth derivatives]
        x_dot = np.stack([log_data_dot['{}_dot'.format(_name)] for _name in name_states], axis=1)

        # Build the ndarray of states [smoothed states]
        x = np.stack([log_data_dot[_name] for _name in name_states], axis=1)

        # Build the control action ndarray
        u = np.stack([log_data[_name] for _name in name_controls if _name in log_data], axis=1)

        # Print the size of the data
        tqdm.write('Size of the data : {}'.format(x.shape))

        # Append the data to the list
        xdot_list.append(x_dot)
        xlist.append(x)
        ulist.append(u)
    
    # Merge the list on the first axis
    x_dot = np.concatenate(xdot_list, axis=0)
    x = np.concatenate(xlist, axis=0)
    u = np.concatenate(ulist, axis=0)

    # Save the data in a dictionary
    m_res = {'x_dot' : x_dot, 'x' : x, 'u' : u}

    # Save the filtered data
    with open(log_dir_dir + '/' + log_name + '_filtered.pkl', 'wb') as f:
        pickle.dump(m_res, f)

    return m_res


def create_loss_function(prior_model_params, loss_weights, seed=0):
    """ Create the learning loss function and the parameters of the 
        approximation
    """
    # Predictor for the vector field
    vector_field_pred = lambda x, u : SDERotorModel(prior_model_params).vector_field(x, u)

    # Transform these functions into Haiku modules
    vector_field_pred_fn = hk.without_apply_rng(hk.transform(vector_field_pred))

    # Initialize the parameters of the vector field predictor
    nx, nu = prior_model_params['n_x'], prior_model_params['n_u']
    vector_field_pred_params = vector_field_pred_fn.init(seed, np.zeros((nx,)), np.zeros((nu,)))

    # The initial parameters of the model are stored in prior_model_params['init_params'
    init_params = prior_model_params.get('init_params', {})

    # Now we create the constraint to add to these parameters
    nominal_params = get_penalty_parameters(vector_field_pred_params, {}, 0.) # Set the default desired values to be zero
    nominal_params = get_penalty_parameters(nominal_params, init_params, None) # The desired values are the initial values

    # Print the resulting penalty coefficients
    print('Nominal parameters values: \n {}'.format(nominal_params))

    # Let's get the penalty coefficients for regularization
    special_parameters = loss_weights.get('special_parameters_pen', {}) # Penalty coefficients for special parameters
    default_weights = loss_weights.get('default_weights', 0.) # Penalty parameters for all other parameters
    penalty_coeffs = get_penalty_parameters(vector_field_pred_params, special_parameters, default_weights)

    # Print the resulting penalty coefficients
    print('Penalty coefficients: \n {}'.format(penalty_coeffs))

    # Create the loss function
    def _loss_fun(est_params, batch_xdot, batch_x, batch_u):
        """The loss function"""

        # Compute the vector field prediction
        xdot_pred = jax.vmap(lambda x, u : vector_field_pred_fn.apply(est_params, x, u), in_axes=(0,0))(batch_x, batch_u)

        # Compute the loss
        # only get the speed and angular velocity
        diff_x = xdot_pred - batch_xdot

        # loss_xdot = jnp.mean(jnp.square(diff_x))
        loss_xdot = jnp.mean(jnp.square(jnp.concatenate([diff_x[:,3:6], diff_x[:,10:13]], axis=0)))

        # W loss
        w_loss_arr = jnp.array( [jnp.sum(jnp.square(p - p_n)) * p_coeff \
                            for p, p_n, p_coeff in zip(jax.tree_util.tree_leaves(est_params), jax.tree_util.tree_leaves(nominal_params), jax.tree_util.tree_leaves(penalty_coeffs)) ]
                        )
        w_loss = jnp.sum(w_loss_arr)

        # Compute the deviation of est_params from prior_model_params
        total_loss = loss_xdot * loss_weights['pen_xdot']
        total_loss += w_loss * loss_weights.get('pen_params', 1.)

        return total_loss, {'total_loss' : total_loss,
                            'loss_xdot' : loss_xdot, 
                            'loss_par' : w_loss}
    
    return vector_field_pred_params, _loss_fun, vector_field_pred_fn

def convert_dict_jnp_to_dict_list(d):
    """Convert a dictionary with jnp arrays to a dictionary with lists"""
    res_dict = {}
    for k, v in d.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = convert_dict_jnp_to_dict_list(v)
        else:
            list_value = v.tolist()
            res_dict[k] = list_value 
    return res_dict


def train_static_model(yaml_cfg_file, output_file=None, fm_model=None):
    """Train the static model

    Args:
        yaml_cfg_file (str): The path to the yaml configuration file
    """
    # Open the yaml file containing the configuration to train the model
    cfg_train = load_yaml(yaml_cfg_file)

    # Obtain the cutoff frequencies for the data filtering
    cutoff_freqs = cfg_train['cutoff_freqs']
    # Pretty print the cutoff frequencies
    print('\nCutoff frequencies for the data filtering')
    for k, v in cutoff_freqs.items():
        print('\t - {} : {}'.format(k, v))

    # Obtain the path to the ulog files
    logs_dir = cfg_train['logs_dir']
    print('\nPath to the ulog files')

    # Number of states and inputs
    nx, nu, ny = cfg_train['n_x'], cfg_train['n_u'], cfg_train['n_y']

    # Load the data from the ulog file
    full_data = list()
    for log_dir in tqdm(logs_dir):
        # Pretty print the path to the ulog files
        tqdm.write('\t - {}'.format(log_dir))
        _data = init_data(log_dir, cutoff_freqs, force_filtering=cfg_train['force_filtering'], zmin=cfg_train.get('zmin', 0.1), mavg_dict=cfg_train.get('mavg_dict', {}))
        if _data is None:
            # warn the user that the data could not be loaded
            tqdm.write('WARNING: The following trajectory was empty: {}'.format(log_dir))
            continue
        # CHeck that the number of states and inputs is correct
        assert _data['x'].shape[1] == nx, 'The number of states is not correct'
        assert _data['u'].shape[1] == nu, 'The number of inputs is not correct'
        # Perform a moving average filter on the data, specically on control inputs and angular velocity if the flag is set
        full_data.append(_data)
    
    # Merge the data from full_data
    full_data = { k : np.concatenate([d[k] for d in full_data], axis=0) for k in full_data[0].keys()}
    
    # Load the test trajectory data
    test_traj_dir = cfg_train['test_trajectory']
    test_traj_data = init_data(test_traj_dir, cutoff_freqs, force_filtering=cfg_train['force_filtering'], zmin=cfg_train.get('zmin', 0.1), mavg_dict=cfg_train.get('mavg_dict', {}))
    assert test_traj_data is not None, 'The test trajectory data could not be loaded'
    # CHeck that the number of states and inputs is correct
    assert test_traj_data['x'].shape[1] == nx, 'The number of states in test trajectory is not correct'
    assert test_traj_data['u'].shape[1] == nu, 'The number of inputs in test trajectory is not correct'
    
    # Random number generator for numpy variables
    seed = cfg_train['seed']
    # Numpy random number generator
    m_numpy_rng = np.random.default_rng(seed)
    # Generate the JAX random key generator
    # train_rng = jax.random.PRNGKey(seed)

    # Get the path the directory of this file
    m_file_path = os.path.expanduser(cfg_train['vehicle_dir'])

    # Load the prior model parameters
    init_params = cfg_train['init_params']
    # if fm_model is not None:
    #     init_params['fm_model'] = fm_model
    # fm_model = init_params['fm_model']

    # Prettu print the prior model parameters
    print('\nPrior model parameters')
    for k, v in init_params.items():
        print('\t - {} : {}'.format(k, v))
    
    # Define the prior param models
    prior_model_params = {'init_params' : init_params, 'horizon' : 1, 'stepsize' : 0.01, 
                            'n_x' : nx, 'n_u' : nu, 'n_y' : ny} 

    # Create the hk parameters and the loss function
    pb_params, loss_fun, _ = \
        create_loss_function(prior_model_params, cfg_train['loss'], seed=seed)

    # Pretty print the loss weights
    print('\nLoss weights')
    for k, v in cfg_train['loss'].items():
        print('\t - {} : {}'.format(k, v))
        
    # Pretty print the initial parameters of the vector field predictor
    print('\nInitial parameters of the vector field predictor')
    for k, v in pb_params.items():
        # Check if the parameter is a dictionary
        if isinstance(v, dict):
            # Print the key first 
            print('\t - {}'.format(k))
            # Print the subkeys
            for k2, v2 in v.items():
                print('\t\t - {} : {}'.format(k2, v2))
        else:
            print('\t - {} : {}'.format(k, v))
    
    # Define the multi_trajectory loss
    @jax.jit
    def actual_loss(est_params, data):
        """The actual loss function"""
        # Get the batch of data
        batch_xdot = data['x_dot']
        batch_x = data['x']
        batch_u = data['u']
        # Compute the loss
        loss, loss_dict = loss_fun(est_params, batch_xdot, batch_x, batch_u)
        return loss, loss_dict
    
    # Define the evaluation function
    eval_test_fn = lambda est_params: evaluate_loss_fn(actual_loss, est_params, test_traj_data, cfg_train['training']['test_batch_size'])

    # Create the optimizer
    # Customize the gradient descent algorithm
    print('\nInitialize the optimizer')
    optim = cfg_train['optimizer']
    special_solver = False
    if type(optim) is list:
        chain_list = []
        for elem in optim:
            m_fn = getattr(optax, elem['name'])
            m_params = elem.get('params', {})
            print('Function : {} | params : {}'.format(elem['name'], m_params))
            if elem.get('scheduler', False):
                m_params = m_fn(**m_params)
                chain_list.append(optax.scale_by_schedule(m_params))
            else:
                chain_list.append(m_fn(**m_params))
        # Build the optimizer to be initialized later
        opt = optax.chain(*chain_list)
        opt_state = opt.init(pb_params)
    else:
        from jaxopt import PolyakSGD, ArmijoSGD
        # Create the optimizer
        if optim['name'] == 'PolyakSGD':
            opt_fun = PolyakSGD
        else:
            opt_fun = ArmijoSGD
        # Specify that the optimizer is a special solver (PolyakSGD or ArmijoSGD)
        special_solver = True
        # Initialize the optimizer
        opt = opt_fun(actual_loss, has_aux=True, jit=True, **optim['params'])
        opt_state = opt.init_state(pb_params, {k : v[:10,:] for k, v in full_data[0].items() })
    # Initialize the parameters of the neural network
    init_nn_params = pb_params

    # Get the non-negative parameters if any
    nonneg_params = get_non_negative_params(pb_params, { k : True for k in cfg_train.get('non_neg_params', [])} )
    print('Nonnegative parameters: \n {}'.format(nonneg_params))


    @jax.jit
    def projection(paramns, data):
        """Project the parameters onto non-negative values and compute the loss"""
        # Do the projection of only relevant parameters
        _paramns = jax.tree_map(lambda x, nonp : jnp.maximum(x, 1e-6) if nonp else x, paramns, nonneg_params)
        return _paramns, actual_loss(_paramns, data)[1]

    # Define the update function that will be used with no special solver
    @jax.jit
    def update(paramns, opt_state, data):
        """Update the parameters of the neural network"""
        # Compute the gradients
        grads, loss_dict = jax.grad(actual_loss, has_aux=True)(paramns, data)
        # Update the parameters
        updates, opt_state = opt.update(grads, opt_state, paramns)
        # Update the parameters
        paramns = optax.apply_updates(paramns, updates)
        # Do the projection
        paramns, loss_dict = projection(paramns, data)
        return paramns, opt_state, loss_dict
    
    # Utility function for printing / displaying loss evolution
    def fill_dict(m_dict, c_dict, inner_name, fstring):
        for k, v in c_dict.items():
            if k not in m_dict:
                m_dict[k] = {}
            m_dict[k][inner_name] = fstring.format(v)

    # Print the dictionary of values as a table in the console
    subset_key = cfg_train.get('key_to_show', None)
    pretty_dict = lambda d : pd.DataFrame({_k : d[_k] for _k in subset_key} \
                                            if subset_key is not None else d
                                          ).__str__()
    
    # Save the number of iteration
    itr_count = 0
    count_epochs_no_improv = 0

    # Save the loss evolution and other useful quantities
    opt_params_dict = init_nn_params
    opt_variables = {}
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    log_data_list = []
    parameter_evolution = []

    # Save all the parameters of this function
    m_parameters_dict = {'params' : cfg_train, 'seed' : seed}

    # Output directory
    output_dir = '{}/my_models/'.format(m_file_path)

    # Output file and if None, use the current data and time
    out_data_file = output_file if output_file is not None else \
        'static_model_{}'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    # get_all_leaf_dict
    def save_learned_params(_opt_params):
        # Save the initial parameters in a yaml file
        with open(output_dir+out_data_file+'.yaml', 'w') as params_outfile:
            converted_params = {**init_params, **convert_dict_jnp_to_dict_list(get_all_leaf_dict(_opt_params))}
            # Add the prior parameters
            save_dict = {'learned' : converted_params, 'prior' : init_params}
            yaml.dump(save_dict, params_outfile)

    # # Open the info file to save the parameters information
    # outfile = open(output_dir+out_data_file+'_info.txt', 'w')
    # outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    # outfile.write('\n////// Command line messages \n\n')
    # outfile.close()

    # Save the initial parameters in a yaml file
    save_learned_params(opt_params_dict)
    
    # Iterate through the epochs
    training_params = cfg_train['training']

    # Extract the batch size
    batch_size = training_params['train_batch_size']
    # Find the number of evals per epoch
    # num_evals_per_epoch = np.max([ _data['x'].shape[0] // batch_size for _data in full_data ])
    num_evals_per_epoch = full_data['x'].shape[0] // batch_size

    for epoch in tqdm(range(training_params['nepochs'])):
        # Counts the number of epochs until cost does not imrpove anymore
        count_epochs_no_improv += 1

        # Iterate through the number of total batches
        for i in tqdm(range(num_evals_per_epoch), leave=False):
            log_data = dict()

            # # Generate a bunch of random batch indexes for each trajectory in fulldata
            # batch_idx = [ m_numpy_rng.choice(_data['x'].shape[0], batch_size, replace=False) \
            #                 for _data in full_data ]

            # # Extract the data from the batch indexes
            # batch_data = [ {k : v[batch_idx_ind,:] for k, v in _data.items()} \
            #                 for batch_idx_ind, _data in zip(batch_idx, full_data) ]

            # # Concatenate the data
            # batch_data = {k : np.concatenate([_data[k] for _data in batch_data], axis=0) \
            #                 for k in batch_data[0].keys()}

            # Generate the batch data
            batch_idx = m_numpy_rng.choice(full_data['x'].shape[0], batch_size, replace=False)
            batch_data = {k : full_data[k][batch_idx] for k in full_data.keys()}

            
            if itr_count == 0:
                _train_dict_init = eval_test_fn(init_nn_params)
                _test_dict_init = copy.deepcopy(_train_dict_init)
            
            # Increment the iteration count
            itr_count += 1

            # Start the timer
            update_start = time.time()

            # Update the parameters
            if special_solver:
                # Update the parameters with the special solver
                pb_params, opt_state = opt.update(pb_params, opt_state, batch_data)
                tree_flatten(opt_state)[0][0].block_until_ready()

                # Projection onto non-negative values
                pb_params, _train_res = projection(pb_params, batch_data)
            else:
                # Update the parameters with the standard solver
                pb_params, opt_state, _train_res = update(pb_params, opt_state, batch_data)
                tree_flatten(opt_state)[0][0].block_until_ready()
                
            update_end = time.time() - update_start

            # Include time in _train_res for uniformity with test dataset
            _train_res['Pred. Time'] = update_end

            # Total elapsed compute time for update only
            if itr_count >= 5: # Remove the first few steps due to jit compilation
                update_time_average = (itr_count * update_time_average + update_end) / (itr_count + 1)
                compute_time_update.append(update_end)
                total_time += update_end
            else:
                update_time_average = update_end

            # Check if it is time to compute the metrics for evaluation
            if itr_count % training_params['test_freq'] == 0 or itr_count == 1:
                # Print the logging information
                print_str_test = '----------------------------- Eval on Test Data [Iteration count = {} | Epoch = {}] -----------------------------\n'.format(itr_count, epoch)
                tqdm.write(print_str_test)

                # Compute the metrics on the test dataset
                _test_res = eval_test_fn(pb_params)

                # First time we have a value for the loss function
                curr_improv_loss = _test_res['total_loss'] + _train_res['total_loss'] * training_params.get('TrainCoeff', 0.0)
                # improv_cond = opt_variables['total_loss'] >= 
                if itr_count == 1 or (opt_variables['total_loss'] >= curr_improv_loss):
                    opt_params_dict = pb_params
                    opt_variables = _test_res
                    opt_variables['total_loss'] = curr_improv_loss
                    count_epochs_no_improv = 0

                fill_dict(log_data, _train_res, 'Train', '{:.3e}')
                fill_dict(log_data, _test_res, 'Test', '{:.3e}')
                log_data_copy = copy.deepcopy(log_data)
                fill_dict(log_data_copy, opt_variables, 'Opt. Test', '{:.3e}')
                fill_dict(log_data_copy, _train_dict_init, 'Init Train', '{:.3e}')
                fill_dict(log_data_copy, _test_dict_init, 'Init Test', '{:.3e}')
                parameter_evolution.append(opt_params_dict)

                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += pretty_dict(log_data_copy)
                print_str += '\n Number epochs without improvement  = {}'.format(count_epochs_no_improv)
                print_str += '\n'
                # tqdm.write(print_str)

                # Pretty print the parameters of the model
                print_str += '----------------------------- Model Parameters -----------------------------\n'
                # Print keys adn values
                for key, value in opt_params_dict.items():
                    if isinstance(value, dict):
                        print_str += '{}: \n'.format(key)
                        for key2, value2 in value.items():
                            print_str += '    {}: \t OPT= {} \t CURR={} \n'.format(key2, value2, pb_params[key][key2])
                    else:
                        print_str += '{}: {} \n'.format(key, value)
                print_str += '\n'
                tqdm.write(print_str)

                # Save all the obtained data
                log_data_list.append(log_data)

                # # Save these info of the console in a text file
                # outfile = open(output_dir+out_data_file+'_info.txt', 'a')
                # outfile.write(print_str_test)
                # outfile.write(print_str)
                # outfile.close()

            last_iteration = (epoch == training_params['nepochs']-1 and i == num_evals_per_epoch-1)
            last_iteration |= (count_epochs_no_improv > training_params['patience'])

            if itr_count % training_params['save_freq'] == 0 or last_iteration:
                m_dict_res = {'last_params' : pb_params,
                                'best_params' : opt_params_dict,
                                'total_time' : total_time,
                                'opt_values' : opt_variables,
                                'compute_time_update' : compute_time_update,
                                'log_data' : log_data_list,
                                'init_losses' : (_train_dict_init, _test_dict_init),
                                'training_parameters' : m_parameters_dict,
                                'parameter_evolution' : parameter_evolution}
                outfile = open(output_dir+out_data_file+'.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()

                save_learned_params(opt_params_dict)

            if last_iteration:
                break
        if last_iteration:
            break


if __name__ == '__main__':

    # Parse the arguments
    # train_static_model.py --cfg cfg_sitl_iris.yaml --output_file test --fm_model cubic
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the yaml configuration file')
    parser.add_argument('--out', type=str, default=None, help='Path to the output file')
    # Parse the motor model
    parser.add_argument('--fm_model', type=str, default='', help='Force and Moment model to use: linear, quadratic, cubic, sigmoid_linear, sigmoid_quad')
    args = parser.parse_args()

    # Train the static model
    train_static_model(args.cfg, output_file=args.out, fm_model=args.fm_model if len(args.fm_model) > 0 else None)