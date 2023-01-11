# Import JAX and utilities
import jax

import jax.numpy as jnp
from jax.tree_util import tree_flatten

# Optax for the optimization scheme
import optax

from tqdm.auto import tqdm

import numpy as np

import copy

from sde4mbrl.nsde import create_model_loss_fn, create_valueNpolicy_loss_fn

from sde4mbrl.utils import get_value_from_dict

import pickle
import pandas as pd
import time


import datetime

# [NOTE] Only current estimate are use to compute the test loss
# No merge of copy for that, this shows the atual result instead of the biased result

# The test loss might increase at the beginning. That is perfectly fine
# The main idea is to wait until it stabilize, this will provide the perfect loss

def apply_fn_to_allleaf(fn_to_apply, types_change, dict_val):
    """Apply a function to all the leaf of a dictionary
    """
    res_dict = {}
    for k, v in dict_val.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = apply_fn_to_allleaf(fn_to_apply, types_change, v)
        elif isinstance(v, types_change):
            res_dict[k] = fn_to_apply(v)
        else:
            res_dict[k] = v
    return res_dict

def pick_dump(mdict, f):
    """Pick a dictionary and dump it into a file
        In the process, convert all jnp.ndarray to np.ndarray
    """
    mdict = apply_fn_to_allleaf(np.array, jnp.ndarray, mdict)
    pickle.dump(mdict, f)


def evaluate_sde_loss(loss_fn, m_params, data_eval, rng, test_batch_size, horizon):
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

    # num_test_batches = (data_eval['y'].shape[0] - (horizon+1)) // test_batch_size
    num_test_batches = data_eval['y'].shape[0] // test_batch_size

    # Iterate over the test batches
    for n_i in tqdm(range(num_test_batches), leave=False):

        # Get the batch
        # [TODO] Old way that assumed trajectories instead of transitions of fixed horizon
        # batch_current = { k : np.array([ v[i:i+horizon+1] if k == 'y' else v[i:i+horizon] \
        #                                 for i in range(n_i*test_batch_size, (n_i+1)*test_batch_size)\
        #                                ]\
        #                             )\
        #                     for k, v in data_eval.items()
        #                 }

        batch_current = { k : v[n_i*test_batch_size:(n_i+1)*test_batch_size] for k, v in data_eval.items() }

        # Separate the batch in finite horizon subtrajectories
        rng, loss_rng = jax.random.split(rng)

        # Infer the next state values of the system
        curr_time = time.time()

        # Compute the loss
        lossval, extra_dict = loss_fn(m_params, rng=loss_rng, **batch_current)
        lossval.block_until_ready()

        diff_time  = time.time() - curr_time
        extra_dict = {**extra_dict, 'Pred. Time' : diff_time}

        if len(result_dict) == 0:
            result_dict = {_key : np.zeros(num_test_batches) for _key in extra_dict}

        # Save the data for logging
        for _key, v in extra_dict.items():
            result_dict[_key][n_i] = v

    return {_k : np.mean(v) for _k, v in result_dict.items()}


def split_trajectories_into_transitions(data, horizon):
    """Split the trajectories in transitions and modify the data dictionary with the new y and u

    Args:
        data (dict): The data dictionary
        horizon (int): The horizon of the trajectories
    """
    # Check if the data is a dictionary
    data = { k : [v] for k, v in data.items()} if isinstance(data, dict) else { k : [_data[k] for _data in data ] for k in data[0].keys()}

    # Check the dimension
    for _, v in data.items():
        for _v in v:
            assert _v.shape[0] >= horizon+1, "The horizon is too large for the data"
    
    res_data = { k : np.array([_v[i:i+horizon+1] if k=='y' else _v[i:i+horizon] \
                                    for _k, _v in enumerate(v) \
                                        for i in range(data['y'][_k].shape[0]-(horizon+1))
                                ]
                            ) \
                    for k, v in data.items()
                }
    
    return res_data


def train_model(params, train_data, test_data, outfile, improvement_cond, sde_constr, **extra_args_sde_constr):
    """TODO: Add extra args for the dynamics in particular

    Args:
        params (TYPE): Description
        train_data (ndarray): Description

    Raises:
        NotImplemented: Description
    """

    # Random number generator for numpy variables
    seed = params['sde_loss']['seed']

    # Numpy random number generator
    m_numpy_rng = np.random.default_rng(seed)

    # Generate the JAX random key generator
    train_rng = jax.random.PRNGKey(seed)

    # Extract the training and testing data set
    print('\n1)   Initialize the data set\n')
    trainer_params = params['sde_training']
    _param2show = trainer_params.get('show_param', [])

    # Load some batching parameters
    train_batch_size, test_batch_size = \
        [ trainer_params[k] for k in ['train_batch', 'test_batch']]
    
    # Initialize the model
    print('\n2) Initialize the model\n')
    nn_params, _loss_fn, nonneg_proj_fn = create_model_loss_fn(params['model'], params['sde_loss'],
                                                sde_constr=sde_constr, seed=seed, 
                                                **extra_args_sde_constr)
    print('Model NN parameters: \n', nn_params)
    print('\nModel init parameters:\n', params['model'])
    print('\nLoss init parameters:\n', params['sde_loss'])

    # Jit the loss function now
    loss_fn = jax.jit(_loss_fn)

    # Check if the data is a trajectory or a set of transitions
    if train_data[0]['u'].shape[0] != params['model']['horizon']:
        assert train_data[0]['u'].shape[0] > params['model']['horizon'], 'The horizon is too large to split the trajectory'
        print ('[WARNING] The train data has a different horizon than the model. It will be split into transitions')
        # Convert the data into transitions
        train_data = split_trajectories_into_transitions(train_data, params['model']['horizon'])
    else:
        # Convert the data into a dictionary
        train_data = { k : np.array([_data[k] for _data in train_data]) for k in train_data[0].keys()}
    
    if test_data[0]['u'].shape[0] != params['model']['horizon']:
        assert test_data[0]['u'].shape[0] > params['model']['horizon'], 'The horizon is too large to split the trajectory'
        print ('[WARNING] The test data has a different horizon than the model. It will be split into transitions')
        test_data = split_trajectories_into_transitions(test_data, params['model']['horizon'])
    else:
        # Convert the data into a dictionary
        test_data = { k : np.array([_data[k] for _data in test_data]) for k in test_data[0].keys()}

    # Check if the train data size with respect to the batch size
    if train_data['y'].shape[0] < train_batch_size:
        train_batch_size = train_data['y'].shape[0]
    if test_data['y'].shape[0] < test_batch_size:
        test_batch_size = test_data['y'].shape[0]

    
    evaluate_loss = lambda m_params, rng: \
                        evaluate_sde_loss(loss_fn, m_params, test_data, rng, 
                                            test_batch_size, params['model']['horizon']
                                        )


    # Build the optimizer for the model
    # Customize the gradient descent algorithm
    print('\n3) Initialize the optimizer\n')
    optim = params['sde_optimizer']
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
    opt_state = opt.init(nn_params)
    init_nn_params = nn_params

    print('\n4) Start training the model...\n')
    # Define the update function
    @jax.jit
    def update(params, _opt_state, in_data, rng_key):
        """ Define the update rule for the parameters of the model
            :param params         : A tuple containing parameters of model
            :param _opt_state     : The current state of the optimizer
            :param in_data        : A batch of the data set
        """
        # By default only differentiate with respect to params
        grads, featvals = jax.grad(_loss_fn, has_aux=True)(params, rng=rng_key, **in_data)
        updates, _opt_state = opt.update(grads, _opt_state, params)
        params = optax.apply_updates(params, updates)
        # Do the projection
        params = nonneg_proj_fn(params)
        return params, _opt_state, featvals

    # Utility function for printing / displaying loss evolution
    def fill_dict(m_dict, c_dict, inner_name, fstring):
        for k, v in c_dict.items():
            if k not in m_dict:
                m_dict[k] = {}
            m_dict[k][inner_name] = fstring.format(v)

    # Print the dictionary of values as a table in the console
    subset_key = trainer_params.get('key_to_show', None)
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

    # Save all the parameters of this function
    m_parameters_dict = {'params' : params, 'seed' : seed}
    out_data_file = outfile if outfile is not None else \
        'sde_model_{}'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # Open the info file to save the parameters information
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('\n////// Command line messages \n\n')
    outfile.close()

    # Now let save the parameter of the model
    outfile = open(out_data_file+'_full_sde_params.pkl', 'wb')
    pick_dump({'sde' : nn_params, 'nominal' : params['model']}, outfile)
    outfile.close()

    # Find the number of evals per epoch
    #  num_evals_per_epoch = np.max([(_data['y'].shape[0] - (horizon_plan +1)) // train_batch_size for _data in train_data ])
    num_evals_per_epoch = train_data['y'].shape[0] // train_batch_size

    # # Check if no improvement has been made
    # has_improved = False

    # Start the iteration loop
    for epoch in tqdm(range(trainer_params['nepochs'])):
        # Counts the number of epochs until cost does not improve anymore
        count_epochs_no_improv += 1

        # Iterate on the total number of batches
        for i in tqdm(range(num_evals_per_epoch), leave=False):
            # # Generate a bunch of random batch indexes for each trajectory in fulldata
            # batch_idx = [ m_numpy_rng.choice(_data['y'].shape[0]-(horizon_plan +1), train_batch_size, replace=False) \
            #                 for _data in train_data ]
            # # Extract the data from the batch indexes
            # batch_data = [ {k : np.array([ v[b_i:(b_i+1+horizon_plan)] if k == 'y' else v[b_i:(b_i+horizon_plan)] \
            #                                 for b_i in b_idx \
            #                             ]) \
            #                     for k, v in _data.items()
            #                 } \
            #                 for b_idx, _data in zip(batch_idx, train_data) ]
            # # Concatenate the data
            # batch_data = {k : np.concatenate([_data[k] for _data in batch_data], axis=0) \
            #                 for k in batch_data[0].keys()}

            # Generate the batch data
            batch_idx = m_numpy_rng.choice(train_data['y'].shape[0], train_batch_size, replace=False)
            batch_data = {k : train_data[k][batch_idx] for k in train_data.keys()}

            # Initialize Log just in case
            log_data = dict()

            train_rng, update_rng = jax.random.split(train_rng)

            if itr_count == 0:
                # Compute the loss on the entire training set
                train_rng, eval_rng_test = jax.random.split(train_rng)
                # Compute the loss on the entire testing set
                _test_dict_init = \
                        evaluate_loss(init_nn_params, eval_rng_test)
                opt_variables = _test_dict_init
                count_epochs_no_improv = 0
                opt_params_dict = init_nn_params

                # Add the additional paramter in the output dictionary to print
                _param_train =  { _k : get_value_from_dict(_k, init_nn_params) for _k in _param2show}
                for _kparam, _vparam in _param_train.items():
                    if _vparam is None:
                        continue
                    _test_dict_init[_kparam] = _vparam

            # Increment the iteration count
            itr_count += 1

            # Update the weight of the nmodel via SGD
            update_start = time.time()
            nn_params, opt_state, _train_res = update(nn_params, opt_state, batch_data, update_rng)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start
            # Include time in _train_res for uniformity with test dataset
            _train_res['Pred. Time'] = update_end

            # Add the additional paramter in the output dictionary to print
            _param_train =  { _k : get_value_from_dict(_k, nn_params) for _k in _param2show}
            for _kparam, _vparam in _param_train.items():
                if _vparam is None:
                    continue
                _train_res[_kparam] = _vparam

            # Total elapsed compute time for update only
            if itr_count >= 5: # Remove the first few steps due to jit compilation
                update_time_average = (itr_count * update_time_average + update_end) / (itr_count + 1)
                compute_time_update.append(update_end)
                total_time += update_end
            else:
                update_time_average = update_end


            # Check if it is time to compute the metrics for evaluation
            if itr_count % trainer_params['test_freq'] == 0 or itr_count == 1:
                # Print the logging information
                print_str_test = '----------------------------- Eval on Test Data [epoch={} | num_batch = {}] -----------------------------\n'.format(epoch, i)
                tqdm.write(print_str_test)

                train_rng, eval_rng_test = jax.random.split(train_rng)

                # Compute the loss on the entire testing set
                _test_res = evaluate_loss(nn_params, eval_rng_test)

                for _kparam, _vparam in _param_train.items():
                    if _vparam is None:
                        continue
                    _test_res[_kparam] = _vparam


                # First time we have a value for the loss function
                # if itr_count == 1 or (opt_variables['Loss Fy'] > _test_res['Loss Fy'] + 10000):
                if improvement_cond(opt_variables, _test_res, _train_res, itr_count): # or trainer_params.get('epochs_before_checking_improv', 0) > epoch:
                    opt_params_dict = nn_params
                    opt_variables = _test_res
                    count_epochs_no_improv = 0
                    # has_improved = True
                
                # Do some printing for result visualization
                fill_dict(log_data, _train_res, 'Train', '{:.3e}')
                fill_dict(log_data, _test_res, 'Test', '{:.3e}')
                log_data_copy = copy.deepcopy(log_data)
                fill_dict(log_data_copy, opt_variables, 'Opt. Test', '{:.3e}')
                fill_dict(log_data_copy, _test_dict_init, 'Init Test', '{:.3e}')

                print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                print_str += pretty_dict(log_data_copy)
                print_str += '\n Number epochs without improvement  = {}'.format(count_epochs_no_improv)
                print_str += '\n'
                tqdm.write(print_str)

                # Save all the obtained data
                log_data_list.append(log_data)

                # Save these info of the console in a text file
                outfile = open(out_data_file+'_info.txt', 'a')
                outfile.write(print_str_test)
                outfile.write(print_str)
                outfile.close()

            last_iteration = (epoch == trainer_params['nepochs']-1 and i == num_evals_per_epoch-1)
            last_iteration |= (count_epochs_no_improv > trainer_params['patience'])

            if itr_count % trainer_params['save_freq'] == 0 or last_iteration:
                m_dict_res = {'best_params' : opt_params_dict,
                              'last_params' : nn_params,
                              'total_time' : total_time,
                              'compute_time_update' : compute_time_update,
                              'opt_values' : opt_variables, 'log_data' : log_data_list,
                              'init_losses' : _test_dict_init,
                              'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'.pkl', "wb")
                pick_dump(m_dict_res, outfile)
                # pickle.dump(m_dict_res, outfile)
                outfile.close()

                # Now let save the parameter of the model
                outfile = open(out_data_file+'_full_sde_params.pkl', 'wb')
                pick_dump({'sde' : opt_params_dict, 'nominal' : params['model']}, outfile)
                outfile.close()

            if last_iteration:
                break

        if last_iteration:
            break
    
def train_value_policy(params, outfile, value_policy_constr, simulation_tools):
    """Summary

    Args:
        params (TYPE): Description
        train_data (TYPE): Description
        outfile (TYPE): Description
        sde_constr (TYPE): Description
        opt_params_sde (TYPE): Description
        cost_fn (TYPE): Description
        R_inv_fn (TYPE): Description
        reset_env (TYPE): Description
        cost_end_simulation (None, optional): Description

    Deleted Parameters:
        improvement_cond (TYPE): Description
    """
    # Random number generator for numpy variables
    seed = params['seed']

    # Numpy random number generator
    m_numpy_rng = np.random.default_rng(seed)

    # Generate the JAX random key generator
    train_rng = jax.random.PRNGKey(seed)

    # Extract the training parameters
    trainer_params = params['valuepol_training']

    # Verbose print messages
    verbose = trainer_params.get('vverbose', False)
    verbose_print = lambda args : tqdm.write(args) if verbose else None

    # Extract few tools from simulation_tools
    obs2state_fn = simulation_tools['opt2state_fn']
    # The horizon length of each trajectory
    ep_len = simulation_tools['episode_length']
    # Create an array/buffer containing past visited states
    num_obs = simulation_tools['n_y']
    num_u = simulation_tools['n_u']
    mpc_solver_sim = simulation_tools['mpc_sim']
    # mpc_solver_sim = jax.jit(simulation_tools['mpc_sim'], backend='cpu')
    # mpc_solver_sim = jax.jit(simulation_tools['mpc_sim'], backend='cpu')

    # Model initialization
    nn_params, __value_loss_fn, (_pol_pure,_val_pure) = \
        create_valueNpolicy_loss_fn(params['valuepol_model'], params['valuepol_loss'], value_policy_constr, seed=seed)
    
    def _value_loss_fn(params, y, u, target_value, rng, opt_params, weight_opt):
        # Split the random number depending on the size of y
        rng = jax.random.split(rng, y.shape[0])
        return __value_loss_fn(params, jax.vmap(obs2state_fn)(y, rng), u, target_value, opt_params, weight_opt)

    # Print the initial parameter values
    print('Value Function Parameters: \n', nn_params)
    print('\nModel init parameters:\n', params['valuepol_model'])
    print('\nLoss init parameters:\n', params['valuepol_loss'])

    # Define the update function
    @jax.jit
    def update(params, _opt_state, batch_data, rng, opt_params, weight_opt):
        """ Define the update rule for the parameters of the model
            :param params         : A tuple containing parameters of model
            :param _opt_state     : The current state of the optimizer
            :param in_data        : A batch of the data set
        """
        # By default only differentiate with respect to params
        grads, featvals = jax.grad(_value_loss_fn, has_aux=True)(params, rng=rng, 
                            opt_params=opt_params, weight_opt=weight_opt, **batch_data)
        updates, _opt_state = opt.update(grads, _opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, _opt_state, featvals
    
    @jax.jit
    def value_target(xevol, cost_evol, opt_params):
        """ Compute the target value function for the value function training
        """
        # Get the control horizon
        H = params['horizon']
        # Now, compute the cumulative cost over the horizon H of the array cost_evol
        if cost_evol.shape[0] % H != 0:
            # If the cost_evol is not a multiple of H, then append zeros to the end of the array
            cost_evol = jnp.concatenate((cost_evol, jnp.zeros((H - cost_evol.shape[0] % H, ))))
        print(cost_evol.shape, xevol.shape)
        # cum_cost_H = jax.vmap(lambda _i : jnp.sum(cost_evol[_i:_i+H]))(jnp.arange(cost_evol.shape[0]))
        cum_cost_H = jax.vmap(lambda _i : jnp.sum(jax.lax.dynamic_slice(cost_evol, (_i,) , (H,) ) ))(jnp.arange(cost_evol.shape[0]))[:xevol.shape[0]]
        # Now extract the endpoints at each xevol
        x_endH = xevol[H:, :]
        # Repeat the last row of xevol and  append it to x_endH to match the size of cum_cost_H
        x_endH = jnp.concatenate((x_endH, jnp.tile(xevol[-1, :], (H, 1))), axis=0)
        # Now compute the value function at the end of the horizon
        value_endH = jax.vmap(lambda _x : _val_pure(opt_params, _x))(x_endH)
        return cum_cost_H + params['discount'] * value_endH

    # Build the optimizer for the model
    optim = params['optimizer']
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
    opt_state = opt.init(nn_params)

    # Utility function for printing / displaying loss evolution
    def fill_dict(m_dict, c_dict, inner_name, fstring):
        for k, v in c_dict.items():
            if k not in m_dict:
                m_dict[k] = {}
            m_dict[k][inner_name] = fstring.format(v)

    # Print the dictionary of values as a table in the console
    subset_key = trainer_params.get('key_to_show', None)
    pretty_dict = lambda d : pd.DataFrame({_k : d[_k] for _k in subset_key} \
                                            if subset_key is not None else d
                                          ).__str__()

    # Save the number of iteration
    itr_count = 0

    # Save the loss evolution and other useful quantities
    opt_params_dict = nn_params
    opt_variables = {}
    total_time, compute_time_update, update_time_average = 0, list(), 0.0
    log_data_list = []

    # Save all the parameters of this function
    m_parameters_dict = {'params' : params, 'seed' : seed}
    out_data_file = outfile

    # Open the info file to save the parameters information
    outfile = open(out_data_file+'_info.txt', 'w')
    outfile.write('Training parameters: \n{}'.format(m_parameters_dict))
    outfile.write('\n////// Command line messages \n\n')
    outfile.close()

    # Number of outer iterations / imaginary environment trajectories
    outer_num_iter = trainer_params['outer_num_iter']

    # Load the training batch size
    train_batch_size = trainer_params['train_batch']

    # Maximum number of trajectories in the buffer
    max_trajectory_in_buffer = trainer_params['max_traj_in_buffer']

    # Maximum number of trajectories used per epoch
    max_traj_per_epoch = trainer_params.get('num_trajs_per_epoch', max_trajectory_in_buffer)

    # Do we prioritize trajetcories with lower cost functions
    expl_eps = trainer_params.get('expl_eps', 0.) # Exploaration probability
    priority_traj = trainer_params.get('priority_on', True) # Prioritize trajectories with lower cost
    prob_temperature = trainer_params.get('prob_temp', 1.0) # Temperature for the probability distribution

    # The current estimate of the optimal cost
    current_opt_cost = np.inf

    # Buffer for previously seen trajectories
    obs_buf = np.zeros((max_trajectory_in_buffer, ep_len, num_obs))
    value_target_buf = np.zeros((max_trajectory_in_buffer, ep_len))
    cost_evol_buf = np.zeros((max_trajectory_in_buffer, ep_len))
    control_buf = np.zeros((max_trajectory_in_buffer, ep_len, num_u))
    cost_buf = np.array([ np.inf for _ in range(max_trajectory_in_buffer)] )

    merged_data_train = {'y' : obs_buf, 'target_value': value_target_buf, 'u' : control_buf}

    # Counter for the number of epochs without improvement
    count_epochs_no_improv = 0

    # Save the current estimate of the optimal parameters
    _outfile = open(out_data_file+'_opt_params.pkl', 'wb')
    pick_dump(opt_params_dict, _outfile)
    _outfile.close()

    # Iterate for learning
    _train_res = None

    for num_iter in tqdm(range(outer_num_iter)):

        # Split the random key generator
        train_rng, traj_rng = jax.random.split(train_rng)

        # We first generate trajectries from the current policy to gather data
        # this will be used to train the value function of the system and a policy
        verbose_print('Generating imaginary trajectory...')
        curr_time = time.time()
        # if num_iter == 0:
        #     faketraj, (u_evol, cost_value, extra_info) =  mpc_solver_sim(nn_params, traj_rng, init_sim=True)
        # else:
        faketraj, (u_evol, cost_value, extra_info) =  mpc_solver_sim(nn_params, traj_rng)
        faketraj.block_until_ready()
        end_time = time.time() - curr_time

        # Now, we obtain the target value from the cost evolution
        curr_target_value = np.array(value_target(faketraj, extra_info['costT'], opt_params_dict['value']))

        current_dataset = {'y' : faketraj, 'target_value': curr_target_value, 'u' : u_evol}
        # current_dataset = {'y' : faketraj, 'target_value': extra_info['costT'], 'u' : u_evol}

        if np.isnan(float(cost_value)):
            verbose_print('Nan value in cost, switching to opt and skipping... ')
            # Switch to the optimal policy
            nn_params = opt_params_dict
            continue

        # for k, v in current_dataset.items():
        #     print(k, v.shape)

        # Some printing
        verbose_print('Imaginary trajectory took {:.3f} seconds...\n'.format(end_time))
        if verbose:
            tqdm.write('Mean: ' + ' | '.join([ '{} : {:.3f}'.format(k,jnp.mean(v)) for k, v in extra_info.items()]))
            tqdm.write('Min : ' + ' | '.join([ '{} : {:.3f}'.format(k,jnp.min(v)) for k, v in extra_info.items()]))
            tqdm.write('Max : ' + ' | '.join([ '{} : {:.3f}'.format(k,jnp.max(v)) for k, v in extra_info.items()]))
            # Print initial state from fake_traj
            tqdm.write('Initial state : ' + faketraj[0,:].tolist().__str__())
        
        # for k in range(extra_info['costT'].shape[0]):
        #     curr_extra = jax.tree_map(lambda x : x[k], extra_info)
        #     tqdm.write('Iter {} : '.format(k) +' | '.join([ '{} : {:.3e}'.format(k,v) for k, v in curr_extra.items() if k != 'u']))
        #     tqdm.write('Current state : ' + faketraj[k,:].tolist().__str__())
        
        # exit()

        # We save the obtained trajectory --> faketraj does not contain the final state
        buffer_index = num_iter % max_trajectory_in_buffer # The buffer is circular

        # We remove the higher cost trajectories
        if priority_traj:
            buffer_index = np.argmax(cost_buf)
        total_num_data = min(max_trajectory_in_buffer, num_iter+1)

        # Incorporate the current trajectory into the data set
        for kname in merged_data_train:
            merged_data_train[kname][buffer_index] = np.array(current_dataset[kname])
        cost_evol_buf[buffer_index] = np.array(extra_info['costT'])

        # Update the cost array
        # Check if cost is nan 
        cost_buf[buffer_index] = float(cost_value) if not np.isnan(float(cost_value)) else np.inf
        # Delete the variables from GPU memory
        del faketraj, u_evol, extra_info

        # The actual cost value is the mean of the cost values
        if num_iter >= 3 * max_trajectory_in_buffer:
            cost_value = np.mean(cost_buf[:total_num_data])
        
        # if num_iter == 2 * max_trajectory_in_buffer:
        #     current_opt_cost = cost_value

        # # Update all the target values in the buffer
        # verbose_print('Updating target values...')
        # for k in range(total_num_data):
        #     curr_target_value = np.array(value_target(merged_data_train['y'][k], cost_evol_buf[k], nn_params['value']))
        #     merged_data_train['target_value'][k] = curr_target_value

        # Check if the cost value has been improved
        if current_opt_cost > cost_value or np.abs(current_opt_cost - cost_value) < trainer_params['cost_tol'] * np.abs(current_opt_cost):
            # count_epochs_no_improv += 1
            if current_opt_cost > cost_value:
                current_opt_cost = cost_value
                opt_params_dict = nn_params
                opt_variables = _train_res
                # Update all the target values in the buffer
                verbose_print('Updating target values...')
                for k in range(total_num_data):
                    curr_target_value = np.array(value_target(merged_data_train['y'][k], cost_evol_buf[k], opt_params_dict['value']))
                    merged_data_train['target_value'][k] = curr_target_value
            count_epochs_no_improv = 0
            pen_cost_opt = jnp.array(0.)
        else:
            count_epochs_no_improv += 1
            pen_cost_opt = jnp.square(current_opt_cost-cost_value)

        # # Hack
        # pen_div_opt = pen_div_opt / current_opt_cost

        explore = int(m_numpy_rng.choice([1,0], p=[expl_eps, 1.0-expl_eps])) == 1
        if priority_traj and not explore:
            prob_cost_arr = np.exp(- (cost_buf[:total_num_data] - current_opt_cost) * prob_temperature) # + 1e-5
            prob_cost_arr /= np.sum(prob_cost_arr)
            replace_sample = True
        else:
            prob_cost_arr = None
            replace_sample = False
        
        # Print the prob cost array with cost
        if verbose:
            tqdm.write('Prob cost array : ' + prob_cost_arr.__str__())
            tqdm.write('Cost array : ' + cost_buf[:total_num_data].__str__())

        # Number of iterations in order to loop over maximum number of trajectories per epoch
        max_size_data = min(max_traj_per_epoch, num_iter+1)
        max_num_data = ep_len * max_size_data

        assert train_batch_size < max_num_data, 'Not enough data for the given batch size'
        # _train_batch_size = min(train_batch_size, max_num_data)
        _train_batch_size = train_batch_size
        max_iterations = max_num_data // _train_batch_size

        # Start the iteration loop
        for epoch in tqdm(range(trainer_params['nepochs_per_iter']), leave=False):

            # Pick high reward data
            priority_trajs_indx = m_numpy_rng.choice(total_num_data,
                                            replace=replace_sample,
                                            size=max_size_data, p=prob_cost_arr)

            # if verbose:
            #     tqdm.write('{}'.format(priority_trajs_indx))
            #     tqdm.write('{}'.format(prob_cost_arr))

            # sUb training dataset
            _merged_data_train = {k : np.concatenate([v[ind] for ind in priority_trajs_indx])\
                                    for k,v in merged_data_train.items()
                                }

            # Iterate on the total number of batches
            for i in range(max_iterations):

                # Used in printing
                log_data = dict()

                # Split the random key generator
                train_rng, update_rng = jax.random.split(train_rng)

                # Get the batch indexes and the batch to work on
                batch_index = m_numpy_rng.choice(max_num_data, size=_train_batch_size,
                                    replace=False, shuffle=False)

                batch_data = { k : v[batch_index] for k, v in _merged_data_train.items()}

                # Increment the iteration count
                itr_count += 1

                # Update the weight of the nmodel via SGD
                update_start = time.time()
                # actual_pen_mult = (0.0 if 'pen_mult_opt' not in trainer_params else trainer_params['pen_mult_opt']) * pen_div_opt

                nn_params, opt_state, _train_res = update(nn_params, opt_state, batch_data, update_rng, opt_params_dict, pen_cost_opt)
                tree_flatten(opt_state)[0][0].block_until_ready()
                update_end = time.time() - update_start

                # Include time in _train_res for uniformity with test dataset
                _train_res['Pred. Time'] = update_end
                _train_res['Cost'] = cost_value

                # Total elapsed compute time for update only
                if itr_count >= 5: # Remove the first few steps due to jit compilation
                    update_time_average = (itr_count * update_time_average + update_end) / (itr_count + 1)
                    compute_time_update.append(update_end)
                    total_time += update_end
                else:
                    update_time_average = update_end

                # Check if it is time to compute the metrics for evaluation
                if itr_count % trainer_params['test_freq'] == 0 or itr_count == 1:
                    # Print the logging information
                    print_str_test = '----------------------------- Eval on Test Data [Iteration count = {} | Epoch = {}] -----------------------------\n'.format(itr_count, epoch)
                    tqdm.write(print_str_test)

                    # Do some printing for result visualization
                    if itr_count == 1:
                        _test_dict_init = copy.deepcopy(_train_res)

                    # First time we have a value for the loss function
                    # if itr_count == 1 or (opt_variables['Loss Fy'] > _test_res['Loss Fy'] + 10000):
                    # if itr_count == 1 or (opt_variables['Cost'] + opt_variables['TD Loss'] >= _train_res['Cost']+ _train_res['TD Loss']):
                    if itr_count == 1 or (opt_variables['Cost'] > _train_res['Cost']):
                        opt_variables = _train_res
                        # opt_params_dict = nn_params
                        # verbose_print('Updating target values...')
                        # for k in range(total_num_data):
                        #     curr_target_value = np.array(value_target(merged_data_train['y'][k], cost_evol_buf[k], opt_params_dict['value']))
                        #     merged_data_train['target_value'][k] = curr_target_value

                    fill_dict(log_data, _train_res, 'Train', '{:.3e}')
                    log_data_copy = copy.deepcopy(log_data)
                    fill_dict(log_data_copy, _test_dict_init, 'Init Test', '{:.3e}')
                    fill_dict(log_data_copy, opt_variables, 'Opt. Test', '{:.3e}')

                    print_str = 'Iter {:05d} | Total Update Time {:.2e} | Update time {:.2e}\n'.format(itr_count, total_time, update_end)
                    print_str += pretty_dict(log_data_copy)
                    print_str += '\n Number epochs without improvement  = {}'.format(count_epochs_no_improv)
                    print_str += '\n'
                    tqdm.write(print_str)

                    # Save all the obtained data
                    log_data_list.append(log_data)

                    # Save these info of the console in a text file
                    outfile = open(out_data_file+'_info.txt', 'a')
                    outfile.write(print_str_test)
                    outfile.write(print_str)
                    outfile.close()

                # last_iteration = (epoch == trainer_params['nepochs_per_iter']-1 and i == max_iterations-1)
                last_iteration = (num_iter == outer_num_iter-1)

                if itr_count % trainer_params['save_freq'] == 0 or last_iteration:
                    m_dict_res = {'last_params' : nn_params,
                                  'best_params' : opt_params_dict,
                                #   'model_params' : opt_params_sde,
                                  'total_time' : total_time,
                                  'opt_values' : opt_variables,
                                  'compute_time_update' : compute_time_update,
                                  'log_data' : log_data_list,
                                  'init_losses' : _test_dict_init,
                                  'training_parameters' : m_parameters_dict}
                    outfile = open(out_data_file+'.pkl', "wb")
                    pickle.dump(m_dict_res, outfile)
                    outfile.close()
                    _outfile = open(out_data_file+'_opt_params.pkl', 'wb')
                    pick_dump(opt_params_dict, _outfile)
                    _outfile.close()

                if last_iteration:
                    break

            if last_iteration:
                break

        if last_iteration:
                break
