import pickle
import time

# Import JAX and utilities
import jax

from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

# Optax for the optimization scheme
import optax

from tqdm.auto import tqdm

import numpy as np

import pandas as pd

import copy

from sde_wrapper import create_loss_fn

def evaluate_loss_fn(loss_fn, m_params, data_eval, rng, num_iter):
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

    for n_i in tqdm(range(num_iter),leave=False):
        rng, loss_rng = jax.random.split(rng)
        # Extract the current data
        batch_current = next(data_eval)

        # Infer the next state values of the system
        curr_time = time.time()
        lossval, extra_dict = loss_fn(m_params, rng=loss_rng, **batch_current)
        lossval.block_until_ready()
        diff_time  = time.time() - curr_time
        extra_dict = {**extra_dict, 'Total Loss' : lossval, 'Pred. Time' : diff_time}

        if len(result_dict) == 0:
            result_dict = {_key : np.zeros(num_iter) for _key in extra_dict}

        # Save the data for logging
        for _key, v in extra_dict.items():
            result_dict[_key][n_i] = v

    return {_k : np.mean(v) for _k, v in result_dict.items()}


def split_dataset_into_finitehorizon_traj(dataset, np_rng, horizon, max_num_traj=None):
    """Split a dataset into an array of size (M, H).
       The input dataset is a dictionary containing t, y, and u
       Each value of t, y, u contains a list of array. The arrays may be of
       different lengths but they are assumed to have the dimension on the first
       axis greater than horizon.
       This function returns a 2D array by merging splitted trajectory of lentgh horizon

    Args:
        dataset (dict): A dictionary with the trajectory of the system
        np_rng (TYPE): numpy radom number genrator
        horizon (int): Horizon of the trajectory in terms of number of points
        max_num_traj (None, optional): The maximum number of trajectory to consider

    Returns:
        TYPE: Description
    """
    # Get the length of the trajectories
    total_num_traj = len(dataset['y'])
    num_traj = total_num_traj if max_num_traj is None else min(max_num_traj, total_num_traj)
    len_traj = np.array([l.shape[0] for l in dataset['y']])

    # Do some dimension checking
    for key, v in dataset.items():
        assert len(v) == total_num_traj, 'Number of trajectories in dataset do not match'
        c_len_traj = np.array([l.shape[0] for l in v])
        assert np.sum(c_len_traj - len_traj) == 0, 'Dimension in dataset do not match'

    # Resulting array dictionary
    splitted_dict = { k : [] for k in dataset}

    # Now proceed with the separation
    for traj_id in range(num_traj):
        # Get the number of chunk and the remainder chunks
        nb_chunk = len_traj[traj_id] // horizon
        rem_chunk = len_traj[traj_id] % horizon

        # Pick a random initialzation between 0 and len - horizon
        indx = np_rng.integers(0, rem_chunk+1)
        n_end = indx + nb_chunk * horizon
        for k, v in dataset.items():
            splitted_dict[k].extend(np.split(v[traj_id][indx:n_end], nb_chunk))
    return splitted_dict
    # return {k : np.array(v) for k, v in splitted_dict.items()}


def shuffle_and_split(np_rng, splitted_dict, batch_size, shuffle=False):
    """Given a dataset of trajetories, this function split the trajectries into
       fixed horizon trajectories and shuffle them. Then it returns a batched version
       of the trajectories to ease the stochastic gradient descent algorithm

    Args:
        np_rng (TYPE): A numpy random generator
        dataset (TYPE): A dictionary of observation, time, and control values
        horizon (TYPE): The horizon of interest
        batch_size (TYPE): The size of the bactc when doing gradient descent
        shuffle (bool, optional): Specify the the dataarray should be split before batching or not

    Yields:
        TYPE: Description
    """
    # Indexes over which the split is going to happen
    indx = np.arange(len(splitted_dict['y']))

    # Shuffle the indexes if requested
    if shuffle:
        np_rng.shuffle(indx)

    # It might be the case that the number of split doesn't exactly matches with the
    # number of horizon trajectories. In this case, we discard the rest of data after shuffled
    assert batch_size <= indx.shape[0], 'The batch size should be lower than the size of dataset'
    num_split = indx.shape[0] // batch_size
    nb_used_data = batch_size * num_split
    splitted_indx = np.split(indx[:nb_used_data], num_split)

    # Save the total number of bactches
    # This is useful for iterating over the dataset without throwing errors
    yield num_split

    # Now iterate through the data set
    for idx in splitted_indx:
        batch_dict = {k : jnp.array([v[i] for i in idx]) for k, v in splitted_dict.items()}
        yield batch_dict


def separate_data(dataset, np_rng, ratio_train, ratio_test):
    """Extract from the dataset the trajectories used when evaluating the model
       for printing the loss function evolution

    Args:
        dataset (ndarray): The dataset. It is a dictionary of obs, control, and
                            time if present
        np_rng (TYPE): Numpy random generator for shuffling the data set
        ratio_test (float): The ratio of data to use for evaluating the model
                            in terms of number of trajectory

    Returns:
        TYPE: Test dataset
    """
    # [TODO Franck] Completely remove the test dataset from the training dataset
    # instead of just extracting a subset of training and still use it for training
    print('Data set dimension: \n', {k : (len(v), np.mean([l.shape[0] for l in v])) for k, v in  dataset.items()})
    # Extract the number of training trajectories
    num_trajectories = len(dataset['y'])
    # Find the number of testing trajectories based on the given ratio
    num_test_trajectories = int(ratio_test * num_trajectories)
    # Find the number of testing trajectories based on the given ratio
    num_train_trajectories = int(ratio_train * num_trajectories)
    # Randomly pick indexes for constructing the testing dataset
    indx_test = np_rng.choice(np.arange(num_trajectories), size=num_test_trajectories, replace=False)
    # Randomly pick indexes for constructing the training dataset
    indx_train = np_rng.choice(np.arange(num_trajectories), size=num_train_trajectories, replace=False)
    # Build the test dataset
    dataset_test = {k : [v[i] for i in indx_test] for k, v in dataset.items()}
    dataset_train = {k : [v[i] for i in indx_train] for k, v in dataset.items()}
    return dataset_train, dataset_test


def train_model(params, train_data, outfile, improvement_cond, sde_constr):
    """Fit a model on the training data set for either the front lateral force
       or the rear lateral and longitidunal force

    Args:
        params (TYPE): Description
        train_data (ndarray): Description

    Raises:
        NotImplemented: Description
    """

    # Random number generator for numpy variables
    seed = params['seed']
    # Numpy random number generator
    m_numpy_rng = np.random.default_rng(seed)
    # Generate the JAX random key generator
    train_rng = jax.random.PRNGKey(seed)

    # Extract the training and testing data set
    print('\n1)   Initialize the data set\n')

    trainer_params = params['training']

    # Load some batching parameters
    ratio_test, ratio_train, train_batch_size, test_batch_size = \
        [ trainer_params[k] for k in ['ratio_test', 'ratio_train', 'train_batch', 'test_batch']]

    # SPlit the training data into train and test dataset
    logging_train, logging_test = separate_data(train_data, m_numpy_rng, ratio_train, ratio_test)

    # During training evaluation for printing, only
    # When printing the loss, specify the number of training data set that will be used
    train_num_batch_eval = trainer_params['train_num_batch_eval']

    # Initialize the model
    print('\n2) Initialize the model\n')
    nn_params,  _loss_fn = create_loss_fn(params['model'], params['loss'],
                                                sde_constr=sde_constr, seed=seed)
    print('Model NN parameters: \n', nn_params)
    print('\nModel init parameters:\n', params['model'])
    print('\nLoss init parameters:\n', params['loss'])

    # Jit the loss function now
    loss_fn = jax.jit(_loss_fn)

    evaluate_loss = lambda m_params, data_eval, rng, num_iter: \
                        evaluate_loss_fn(loss_fn, m_params, data_eval, rng, num_iter)

    # Build the optimizer for the model
    # Customize the gradient descent algorithm
    print('\n3) Initialize the optimizer\n')
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
        grads, _ = jax.grad(_loss_fn, has_aux=True)(params, rng=rng_key, **in_data)
        updates, _opt_state = opt.update(grads, _opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, _opt_state

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
    itr_count_corr = 0
    count_epochs_no_improv = 0

    # Save the loss evolution and other useful quantities
    opt_params_dict = init_nn_params
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

    # The data set that will be used for evaluation
    horizon_plan = params['loss']['horizon']

    # Start the iteration loop
    for epoch in tqdm(range(trainer_params['nepochs'])):
        # Split the dataset into trajectory of fixed horizon
        if epoch % trainer_params['data_split_rate'] == 0:
            tqdm.write('Splitting dataset into finite length trajectories...\n')
            splitted_train_data = split_dataset_into_finitehorizon_traj(train_data, m_numpy_rng, horizon_plan)
            splitted_train_data_eval = split_dataset_into_finitehorizon_traj(logging_train, m_numpy_rng, horizon_plan)
            splitted_test_data_eval = split_dataset_into_finitehorizon_traj(logging_test, m_numpy_rng, horizon_plan)
            tqdm.write('End of splitting dataset into finite length trajectories...\n')

        # Shuffle the entire data set at each epoch and return iterables
        ds_train_c = shuffle_and_split(m_numpy_rng, splitted_train_data, train_batch_size, shuffle=True)
        num_train_batches = next(ds_train_c)

        # Counts the number of epochs until cost does not imrpove anymore
        count_epochs_no_improv += 1

        # Iterate on the total number of batches
        for i in tqdm(range(num_train_batches), leave=False):

            # Initialize Log just in case
            log_data = dict()

            # Get the next batch of images
            batch_current = next(ds_train_c)
            train_rng, update_rng = jax.random.split(train_rng)

            if itr_count == 0:
                batch_train_init = shuffle_and_split(m_numpy_rng, splitted_train_data_eval,
                                                        train_batch_size, shuffle=False)
                batch_test_init = shuffle_and_split(m_numpy_rng, splitted_test_data_eval,
                                                        test_batch_size, shuffle=False)
                num_train_batches_eval = next(batch_train_init)
                num_test_batches = next(batch_test_init)
                # Compute the loss on the entire training set
                train_rng, eval_rng_train, eval_rng_test = jax.random.split(train_rng, 3)
                _train_dict_init = \
                        evaluate_loss(init_nn_params, batch_train_init, eval_rng_train, num_train_batches_eval)

                # Compute the loss on the entire testing set
                _test_dict_init = \
                        evaluate_loss(init_nn_params, batch_test_init, eval_rng_test, num_test_batches)

            # Increment the iteration count
            itr_count += 1

            # Update the weight of the nmodel via SGD
            update_start = time.time()
            nn_params, opt_state = update(nn_params, opt_state, batch_current, update_rng)
            tree_flatten(opt_state)[0][0].block_until_ready()
            update_end = time.time() - update_start

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

                # Split / shuffle the data if needed
                batch_train_i = shuffle_and_split(m_numpy_rng, splitted_train_data_eval,
                                                        train_batch_size, shuffle=False)
                batch_test_i = shuffle_and_split(m_numpy_rng, splitted_test_data_eval,
                                                        test_batch_size, shuffle=False)
                num_train_batches_eval = next(batch_train_i)
                num_test_batches = next(batch_test_i)
                train_rng, eval_rng_train, eval_rng_test = jax.random.split(train_rng, 3)

                # Compute the loss on the entire training set
                _train_res = evaluate_loss(nn_params, batch_train_i, eval_rng_train, num_train_batches_eval)

                # Compute the loss on the entire testing set
                _test_res = evaluate_loss(nn_params, batch_test_i, eval_rng_test, num_test_batches)

                # First time we have a value for the loss function
                # if itr_count == 1 or (opt_variables['Loss Fy'] > _test_res['Loss Fy'] + 10000):
                if itr_count == 1 or improvement_cond(opt_variables, _test_res, _train_res):
                    opt_params_dict = nn_params
                    opt_variables = _test_res
                    count_epochs_no_improv = 0


                # Do some printing for result visualization
                fill_dict(log_data, _train_res, 'Train', '{:.3e}')
                fill_dict(log_data, _test_res, 'Test', '{:.3e}')
                log_data_copy = copy.deepcopy(log_data)
                fill_dict(log_data_copy, opt_variables, 'Opt. Test', '{:.3e}')
                fill_dict(log_data_copy, _train_dict_init, 'Init Train', '{:.3e}')
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

            last_iteration = (epoch == trainer_params['nepochs']-1 and i == meta['num_train_batches']-1)
            last_iteration |= (count_epochs_no_improv > trainer_params['patience'])

            if itr_count % trainer_params['save_freq'] == 0 or last_iteration:
                m_dict_res = {'best_params' : opt_params_dict,
                              'total_time' : total_time,
                              'compute_time_update' : compute_time_update,
                              'opt_values' : opt_variables, 'log_data' : log_data_list,
                              'init_losses' : (_train_dict_init, _test_dict_init),
                              'training_parameters' : m_parameters_dict}
                outfile = open(out_data_file+'.pkl', "wb")
                pickle.dump(m_dict_res, outfile)
                outfile.close()

            if last_iteration:
                break

        if last_iteration:
            break

# m_numpy_rng = np.random.default_rng(0)
# ydata = []
# for i in range(10):
#     traj_size = m_numpy_rng.integers(6, 9)
#     ydata.append(m_numpy_rng.standard_normal((traj_size,3)))

# ydict = {'y' : ydata}

# sp_dict = split_dataset_into_finitehorizon_traj(ydict, m_numpy_rng, 5)

# it_split = shuffle_and_split(m_numpy_rng, ydict, 5, 5, shuffle=True)

# nval = next(it_split)
# print(sp_dict)
# nval['y'][1][0,:] = 0
# print(nval)
# print(ydict)
# print(sp_dict)


# print('NEXTTTTTT')
# sp_dict['y'][0][0,:] = 0.
# print(sp_dict['y'])
# print(ydict)
