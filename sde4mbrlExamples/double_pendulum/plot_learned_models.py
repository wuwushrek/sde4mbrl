import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp

from double_pendulum_model import load_data_generator, load_learned_model, _load_pkl

import numpy as np

import pickle
import yaml

import matplotlib.pyplot as plt


######### Parameters for Rolling out models #########
# The groundtruth model
model_groundtruth_dir = "double_pendulum.yaml"
# Number of integration steps
HORIZON = 8000
# Number of samples when rolling out the model
NUM_SAMPLES = 100
# Number of transitions when evaluating the model accuracy
NUM_TRANSITIONS = 1000
seed_trans = 10
# Initial state
xinit = np.array([-0.02, 0.02, 0.0, 0.0])
# Sampling accuracy test
# This is used to create data to evaluate the models accuracy
xlb_range = np.array([-0.5, -0.5, -0.3, -0.3])
xub_range = np.array([0.5, 0.5, 0.3, 0.3])
# Random PRNG number
seed = 0
seed_rng = jax.random.PRNGKey(seed)
#####################################################


# Get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
my_models_dir = current_dir + "/my_models/"
my_data_dir = current_dir + "/my_data/"

def generate_data_for_plots(outfile='my_data'):
    global seed_rng

    ########## Groundtruth trajectory ###################
    # Generate a trajectory starting from xinit and length HORIZON
    groundtruth_sampler, _ = load_data_generator(model_groundtruth_dir, noise_info={}, horizon=HORIZON, ufun=None)
    gtruth_data, _ = groundtruth_sampler(xinit, seed_rng) # Second output is the control input
    gtruth_data = np.array(gtruth_data)

    # A function to generate transitions
    _, trans_generator = load_data_generator(model_groundtruth_dir, noise_info={}, horizon=1, ufun=None) # Maybe the noisy version instead of the groundtruth?
    # Generate transitions
    _true_trans = trans_generator(xlb_range, xub_range, NUM_TRANSITIONS, seed_trans)
    # true_trans = np.array([ _x for (xev, uev) in _true_trans for _x in xev])
    #####################################################

    # Extract all files in the directory containing __MSD_ and ending with _full_sde_params.pkl
    my_models_name = [f for f in os.listdir(my_models_dir) if '__MSD_' in f and f.endswith("_full_sde_params.pkl") ]
    
    # Go through all the models and load them
    my_data_set = {}
    for _k, model_name in enumerate(my_models_name):
        # Some printing
        print("Loading model [{}] --> {}/{}".format(model_name, _k+1, len(my_models_name)))
        # Split the model name to get the training dataset path
        _train_data_name = model_name.split("__")[-1].split("_full_sde_params.pkl")[0]
        # Check if this dataset is already in the dictionary
        if _train_data_name not in my_data_set:
            my_data_set[_train_data_name] = {}

        # Load the training dataset
        train_data = _load_pkl(my_data_dir + _train_data_name + ".pkl")
        # Create the samples from training dataset
        # This is a 2D array containing all the transitions
        train_data = np.array([ xev for (xev, _) in train_data ])

        # Save the samples in the dictionary
        if 'samples' not in my_data_set[_train_data_name]:
            my_data_set[_train_data_name]['samples'] = train_data
            my_data_set[_train_data_name]['groundtruth'] = gtruth_data
        
        # Load the model
        # We extract the model type from the model name
        model_type = model_name.split("__")[0]
        model_predict_fn = load_learned_model(my_models_dir+model_name, horizon=HORIZON, num_samples=NUM_SAMPLES if 'node' not in model_type else 1, ufun=None)
        prior_model_predict_fn = load_learned_model(my_models_dir+model_name, horizon=HORIZON, num_samples=NUM_SAMPLES if 'node' not in model_type else 1, ufun=None, prior_dist=True)

        # Split the original key into three
        _model_key, _prior_key, _sampling_key, seed_rng = jax.random.split(seed_rng, 4)

        # Rollout the model
        model_data = model_predict_fn(xinit, _model_key)
        if 'node' in model_type:
            model_data = model_data[0]
        # Save the model data
        my_data_set[_train_data_name][model_type] = np.array(model_data)

        # Rollout the prior model
        prior_model_data = prior_model_predict_fn(xinit, _prior_key)
        if 'node' in model_type:
            prior_model_data = prior_model_data[0]
        # Save the prior model data
        my_data_set[_train_data_name][model_type + "_prior"] = np.array(prior_model_data)

        # Generate transitions
        _model_sampler = load_learned_model(my_models_dir+model_name, horizon=1, num_samples=NUM_SAMPLES if 'node' not in model_type else 1, ufun=None)
        accuracy_data = []
        for _i in range(NUM_TRANSITIONS):
            _sampling_key, _curr_key = jax.random.split(_sampling_key)
            # Initial state
            m_init = _true_trans[_i][0][0]
            # Sample the model
            _model_data = _model_sampler(m_init, _curr_key)
            accuracy_data.append( np.sum(np.square(np.mean(_model_data, axis=0) -  _true_trans[_i][0][1]) ))
        # Save the accuracy data
        my_data_set[_train_data_name][model_type + "_accuracy"] = np.array(accuracy_data)

    # Save the data
    with open(my_data_dir+outfile+'.pkl', 'wb') as f:
        pickle.dump(my_data_set, f)

generate_data_for_plots()

# Plot configuration
plot_config = {
    'samples': {
        'label' : 'Sampled data',
        'color' : 'gray',
        'marker': 'o', 
        'markersize' : 1, 
        'alpha' : 0.2,
        'linestyle' : 'None',
        'zorder' : 1
    },
    'node_bboxes' : {
        'label' : 'Black Box Neural ODE',
        'color' : 'blue',
        'zorder': 10
    },
    'nesde_bboxes' : {
        'label' : 'Black Box Neural SDE',
        'color' : 'green',
        'zorder' : 20
    },
    'nesde_bboxes_prior' : {
        'label' : 'Prior Black Box Neural SDE',
        'color' : 'cyan',
        'zorder' : 10
    },
    'nesde_phys' : {
        'label' : 'Physics-Informed Neural SDE',
        'color' : 'magenta',
        'zorder' : 20
    },
    'nesde_phys_prior' : {
        'label' : 'Prior Physics-Informed Neural SDE',
        'color' : 'orange',
        'zorder' : 10
    },
    'groundtruth' : {
        'label' : 'True dynamics',
        'color' : 'black',
        'zorder' : 20, 
    }
}

# def plot_data(data_path='my_data'):
data_path='my_data'
data4comparison = _load_pkl(my_data_dir + data_path + '.pkl')

# Folder where all the figures will be saved
fig_folder = my_data_dir + '/figures/'

# # Plot the 2D evolution
# data2show = ['samples', 'node_bboxes','groundtruth', 'nesde_bboxes', 'nesde_phys'] #, 'nesde_bboxes', 'groundtruth']
# for dataname, _data in data4comparison.items():
#     print(_data.keys())
#     plt.figure()
#     for _model in data2show:
#         if _model not in _data:
#             continue
#         _xyvalues = _data[_model]
#         print(_xyvalues.shape)
#         # check the dimension of the data
#         if _xyvalues.ndim > 2:
#             assert _xyvalues.ndim == 3, "The data should be a 3D array"
#             _xyvalues = np.mean(_xyvalues, axis=0)
#         plt.plot(_xyvalues[:,0], _xyvalues[:,1], **plot_config[_model])
#     plt.xlabel('q')
#     plt.ylabel(r'$\dot{q}$')
#     plt.xlim([-0.15, 0.15])
#     plt.ylim([-0.15, 0.15])
#     plt.legend()
#     plt.grid()
#     plt.savefig(fig_folder + dataname + '_2d.png', dpi=300)

# Plot the 1D evolution. Plot x and y on subplots as functions of the number of time steps
data2show = ['samples', 'node_bboxes', 'groundtruth', 'nesde_bboxes', 'nesde_phys'] #, 'nesde_bboxes', 'groundtruth']
# data2show = ['groundtruth'] #, 'nesde_bboxes', 'groundtruth']
for dataname, _data in data4comparison.items():
    # Create a subplot with two rows
    fig, axs = plt.subplots(2, 2, sharex=True)

    for _model in data2show:
        if _model not in _data:
            continue
        _xyvalues = _data[_model]
        print(_xyvalues.shape)
        if _model == 'samples':
            for _i in range(_xyvalues.shape[0]):
                _conf = plot_config[_model] if _i == 0 else {**plot_config[_model], 'label' : None}
                axs[0,0].plot(_xyvalues[_i,:,0], **_conf)
                axs[1,0].plot(_xyvalues[_i,:,1], **_conf)
                axs[0,1].plot(_xyvalues[_i,:,2], **_conf)
                axs[1,1].plot(_xyvalues[_i,:,3], **_conf)
            continue

        # check the dimension of the data
        _xyvalues_min, _xyvalues_max = None, None
        if _xyvalues.ndim > 2:
            assert _xyvalues.ndim == 3, "The data should be a 3D array"
            _xyvalues_min = np.min(_xyvalues, axis=0)
            _xyvalues_max = np.max(_xyvalues, axis=0)
            _xyvalues = np.mean(_xyvalues, axis=0)
        
        xval = np.array(range(_xyvalues.shape[0]))
        # Plot the mean value first
        axs[0,0].plot(xval, _xyvalues[:,0], **plot_config[_model])
        axs[1,0].plot(xval, _xyvalues[:,1], **plot_config[_model])
        axs[0,1].plot(xval, _xyvalues[:,2], **plot_config[_model])
        axs[1,1].plot(xval, _xyvalues[:,3], **plot_config[_model])
        # Plot the min and max values
        if _xyvalues_min is not None:
            axs[0,0].fill_between(xval, _xyvalues_min[:,0], _xyvalues_max[:,0], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})
            axs[1,0].fill_between(xval, _xyvalues_min[:,1], _xyvalues_max[:,1], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})
            axs[0,1].fill_between(xval, _xyvalues_min[:,2], _xyvalues_max[:,2], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})
            axs[1,1].fill_between(xval, _xyvalues_min[:,3], _xyvalues_max[:,3], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})
        
    # # Set the axis limits
    # axs[0].set_ylim([-0.15, 0.15])
    # axs[1].set_ylim([-0.15, 0.15])

    # Set the axis labels
    axs[0,0].set_ylabel(r'$\theta_1$')
    axs[1,0].set_ylabel(r'$\theta_2$')
    axs[0,1].set_ylabel(r'$\dot{\theta}_1$')
    axs[1,1].set_ylabel(r'$\dot{\theta}_2$')

    axs[1,0].set_xlabel('Time step')
    axs[1,1].set_xlabel('Time step')

    # Set grid
    axs[0,0].grid()
    axs[1,0].grid()
    axs[0,1].grid()
    axs[1,1].grid()

    # Set a single legend for the figure
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    plt.savefig(fig_folder + dataname + '_1d.png', dpi=300)

# # Plot the prior
# data2show = ['groundtruth', 'nesde_bboxes', 'nesde_bboxes_prior', 'nesde_phys', 'nesde_phys_prior'] #, 'nesde_bboxes', 'groundtruth']
# for dataname, _data in data4comparison.items():
#     # Create a subplot with two rows
#     fig, axs = plt.subplots(2, 1, sharex=True)
#     for _model in data2show:
#         if _model not in _data:
#             continue
#         _xyvalues = _data[_model]

#         # check the dimension of the data
#         _xyvalues_min, _xyvalues_max = None, None
#         if _xyvalues.ndim > 2:
#             assert _xyvalues.ndim == 3, "The data should be a 3D array"
#             _xyvalues_min = np.min(_xyvalues, axis=0)
#             _xyvalues_max = np.max(_xyvalues, axis=0)
#             _xyvalues = np.mean(_xyvalues, axis=0)
        
#         xval = np.array(range(_xyvalues.shape[0]))
#         # Plot the mean value first
#         axs[0].plot(xval, _xyvalues[:,0], **plot_config[_model])
#         axs[1].plot(xval, _xyvalues[:,1], **plot_config[_model])
#         # Plot the min and max values
#         if _xyvalues_min is not None:
#             axs[0].fill_between(xval, _xyvalues_min[:,0], _xyvalues_max[:,0], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})
#             axs[1].fill_between(xval, _xyvalues_min[:,1], _xyvalues_max[:,1], **{**plot_config[_model], 'alpha':0.2, 'label':None, 'zorder':1})

#     # Set the axis labels
#     axs[0].set_ylabel('q')
#     axs[1].set_ylabel(r'$\dot{q}$')
#     axs[1].set_xlabel('Time step')

#     # Set grid
#     axs[0].grid()
#     axs[1].grid()

#     # Set a single legend for the figure
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', ncol=3)
#     # fig.tight_layout()
#     # fig.subplots_adjust(top=0.9)
#     plt.savefig(fig_folder + dataname + '_1d_prior.png', dpi=300)

# # Plot the accuracy as a function of the number of data and the noise level_accuracy
# # First get the different number of data from file name
# _names2plot = {'noise' : set(), 'domain' : set(), 'num_data' : set()}
# for datafile in data4comparison:
#     # Get noise level
#     dat_info = datafile.split('_')
#     noise_level = dat_info[1]
#     domain_info = dat_info[2]
#     num_data = int(dat_info[3])
#     _names2plot['noise'].add(noise_level)
#     _names2plot['domain'].add(domain_info)
#     _names2plot['num_data'].add(num_data)

# # Sort num data
# _names2plot['num_data'] = np.array(sorted(_names2plot['num_data']))

# data2show = ['nesde_bboxes_accuracy', 'nesde_phys_accuracy', 'node_bboxes_accuracy']
# for _domain in _names2plot['domain']:
#     plt.figure()
#     for _noise in _names2plot['noise']:
#         _data2plot = {}
#         for _num_data in _names2plot['num_data']:
#             _datafile = 'MSD_{}_{}_{}'.format(_noise, _domain, _num_data)
#             for dataname, _data in data4comparison.items():
#                 if dataname != _datafile:
#                     continue
#                 for _model in data2show:
#                     if _model not in _data:
#                         continue
#                     if _model not in _data2plot:
#                         _data2plot[_model] = []
#                     _data2plot[_model].append(_data[_model])
#         # Now we plot the information
#         for _model, _data in _data2plot.items():
#             _data = np.array(_data)
#             _data_mean = np.mean(_data, axis=1)
#             _data_std = np.std(_data, axis=1)

#             print(_data.shape)
#             print(_data_mean.shape)
#             print(_data_std.shape)
#             _model = '_'.join(_model.split('_')[:-1])
#             plt.plot(_names2plot['num_data'], _data_mean, **{**plot_config[_model], 'marker' : 'o', 'markersize' : 5, 'label' : plot_config[_model]['label']+ ' - ' + _noise})
#             # plt.fill_between(_names2plot['num_data'], _data_mean - _data_std, _data_mean + _data_std, **{**plot_config[_model], 'alpha' : 0.2, 'label' : None})
#     plt.xlabel('Number of data')
#     plt.ylabel('Accuracy')
#     plt.grid()
#     plt.legend()
#     plt.savefig(fig_folder + 'accuracy_{}.png'.format(_domain), dpi=300)
        