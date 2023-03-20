""" This script generate the dataset and plots used in the paper
    for prediction error and uncertainty analysis over the learned models
"""
import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from double_pendulum_model import load_data_generator, load_learned_model, load_learned_diffusion, _load_pkl
from sde4mbrl.utils import load_yaml

from tqdm.auto import tqdm

texConfig = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["times", "times new roman"],
    # "mathtext.fontset": "cm",
    # 'text.latex.preamble': [r'\usepackage{newtxmath}'],
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
}

type2label = {
  "t2q1": "$q_1$",
  "t2q2": "$q_2$",
  "t2q1dot": "$\dot{q}_1$",
  "t2q2dot": "$\dot{q}_2$",
  "t2Err" : "Prediction Error on Mean Trajectory",
  "traj2std": "Magnitude of Standard Deviation of Trajectory",
  "traj2error": "Prediction Error on Trajectory",
  # One for the error evolution
}

indx2label = {
    0: "$q_1$",
    1: "$q_2$",
    2: "$\dot{q}_1$",
    3: "$\dot{q}_2$"
}

# # files2plot_full = [ [files2plot.replace('XX', '{}').format(*_val_val) for _val_val in _val['value']] for _val in plot_configs]
# def create_density_mesh_plots(cfg_path, learned_dir, data_dir):
#     """ Create the mesh plots for the density model
#     Args:
#         cfg_path: The path to the config file
#         learned_dir: The directory where all the models are stored
#         data_dir: The directory where the dataset is stored
#     """

#     # Load the density configuration yaml
#     density_cfg = load_yaml(cfg_path)
#     # Check if learned_dir is empty, if so, use the default directory given by the current path
#     if len(learned_dir) == 0:
#         learned_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/'
#     # Check if data_dir is empty, if so, use the default directory given by the current path
#     if len(data_dir) == 0:
#         data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
#     # The directory where the plots will be stored
#     figure_out = data_dir + 'figures/'

#     # Extract the grid limits for the mesh plot
#     qrange = density_cfg['qrange']
#     qdotrange = density_cfg['qdotrange']
#     # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
#     qgrid, qdotgrid = np.meshgrid(np.linspace(qrange[0], qrange[1], density_cfg['num_grid_q']),
#                                     np.linspace(qdotrange[0], qdotrange[1], density_cfg['num_grid_qdot']))
    
#     # Create a random PRNG key
#     rng_key = jax.random.PRNGKey(density_cfg['seed'])

#     # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
#     files = [f for f in os.listdir(learned_dir) if '__MSD_' in f and f.endswith('_sde.pkl')]

#     # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
#     files2plot = density_cfg['files2plot']

#     # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
#     plot_configs = density_cfg['plot_configs']
#     files2plot_full = [ files2plot.replace('XX', '{}').format(*_val['value']) for _val in plot_configs]
    
#     # Now, we can check if the names make sense
#     files = [f for f in files if any([f2p+'_sde' in f for f2p in files2plot_full])]
#     # Make sure that the number of files is the same as the number of files2plot_full
#     assert len(files) == len(set(files2plot_full)), "The number of files to plot is not the same as the number of files2plot_full"

#     # Exract the figure and axis specifications
#     fig_specs = density_cfg['fig_args']
#     nrows = fig_specs['nrows']
#     ncols = fig_specs['ncols']

#     # Check if the number of subplots is the same as the number of files to plot
#     assert nrows*ncols >= len(files), "The number of subplots is not the same as the number of files to plot"

#     # Create the figure
#     fig, axs_2d = plt.subplots(**fig_specs)
#     # Flatten the axes
#     axs = axs_2d.flatten()

#     # Loop over the files
#     itr_count = 0
#     for fname, pconf, ax in zip(files2plot_full, plot_configs, axs):
#         # Print the current step
#         print('\n###########################################################')
#         print(f"Creating mesh plot for {fname}")
#         print(fname)
#         # Get the the file name in files corresponding to the file name fname in files2plot_full
#         fname = [f for f in files if (fname + '_sde') in f]
#         print(f"File name: {fname}")
#         assert len(fname) == 1, "The number of files corresponding to the file name in files2plot_full is not 1"
#         model_name = fname[0]
#         # Extract the data name and model name
#         _train_data_name = model_name.split("__")[-1].split("_sde.pkl")[0]
#         # Load the training dataset
#         train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
#         # Create the samples from training dataset
#         # This is a 2D array containing all the transitions
#         train_data = np.array([ xev for (xev, _) in train_data])
#         # Load the model
#         _diff_est_fn = load_learned_diffusion(learned_dir+model_name, num_samples=density_cfg['num_particles_diffusion'])
#         # Iterate over the meshgrid and compute the prediction error and std
#         _mesh_pred_density = np.zeros((len(qdotgrid), len(qgrid)))
#         # Check if the diffusion should becomputed using the actual density network or sigmoid
#         net_val = pconf.get('net', False)

#         for _i in tqdm(range(len(qgrid))):
#             for _j in range(len(qdotgrid)):
#                 rng_key, density_key = jax.random.split(rng_key)
#                 xcurr = np.array([qgrid[_j,_i], qdotgrid[_j, _i]])
#                 # Compute the density
#                 _mesh_pred_density[_j,_i] = float(_diff_est_fn(xcurr, density_key, net=net_val)[1]) # Second output is the density
        
#         # Scale the mesh between 0 and 1 if we are using the density network directly instead of a sigmoid of the density network
#         if net_val:
#             _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

#         # Plot the mesh
#         pcm = ax.pcolormesh(qgrid, qdotgrid, _mesh_pred_density, vmin=0, vmax=1,**density_cfg['mesh_args'])

#         # Set the title if 'title' is in pconf
#         if 'title' in pconf:
#             ax.set_title(pconf['title'])
        
#         # Set the x axis label
#         # Add xlabel only to the bottom row
#         if itr_count >= (nrows-1)*ncols:
#             ax.set_xlabel('$q$')
        
#         # Add ylabel only to the leftmost column
#         if itr_count % ncols == 0:
#             ax.set_ylabel('$\dot{q}$')

#         if 'title_right' in pconf:
#             # Add a twin axis on the right with no ticks and the label given by title_right
#             ax2 = ax.twinx()
#             ax2.set_ylabel(pconf['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
#             ax2.tick_params(axis='y', which='both', length=0)
#             ax2.set_yticks([])
        
#         itr_count += 1

#         # Now we plot the training data
#         for _i, _traj_sample in enumerate(train_data):
#             ax.plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
#                                                                 'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
#     # Set label for first ax only
#     axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

#     # Plot the colorbar
#     _ = fig.colorbar(pcm, ax=axs, **density_cfg['colorbar_args'])

#     # Save the figure
#     density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
#     fig.savefig(**density_cfg['save_config'])

#     # Plot the figure
#     plt.show()


# def create_uncertainty_plots(cfg_path, learned_dir, data_dir, gt_dir):
#     """ Create the mesh plots for displaying uncertainty in the learned models and prediction error
#     Args:
#         cfg_path: The path to the config file
#         learned_dir: The directory where all the models are stored
#         data_dir: The directory where the dataset is stored
#         gt_dir: The directory where the ground truth model
#     """

#     # Load the density configuration yaml
#     density_cfg = load_yaml(cfg_path)
#     # Check if learned_dir is empty, if so, use the default directory given by the current path
#     if len(learned_dir) == 0:
#         learned_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/'
#     # Check if data_dir is empty, if so, use the default directory given by the current path
#     if len(data_dir) == 0:
#         data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
#     # The directory where the plots will be stored
#     figure_out = data_dir + 'figures/'

#     # Extract the grid limits for the mesh plot
#     qrange = density_cfg['qrange']
#     qdotrange = density_cfg['qdotrange']
#     # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
#     qgrid, qdotgrid = np.meshgrid(np.linspace(qrange[0], qrange[1], density_cfg['num_grid_q']),
#                                     np.linspace(qdotrange[0], qdotrange[1], density_cfg['num_grid_qdot']))
    
#     # Extra parameters for prediction and groundtruth
#     num_particles = density_cfg['num_particles_std']
#     horizon_pred_accuracy = density_cfg['horizon_pred_accuracy']
#     horizon_uncertainty = density_cfg.get('horizon_uncertainty', horizon_pred_accuracy)

#     # Load the ground truth model
#     _mesh_gtsampler, _, _ = load_data_generator(gt_dir, noise_info={}, horizon=horizon_pred_accuracy, ufun=None)
    
#     # Create a random PRNG key
#     rng_key = jax.random.PRNGKey(density_cfg['seed'])

#     # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
#     files = [f for f in os.listdir(learned_dir) if '__MSD_' in f and f.endswith('_sde.pkl')]

#     # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
#     files2plot = density_cfg['files2plot']

#     # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
#     plot_configs = density_cfg['plot_configs']
#     files2plot_full = [ files2plot.replace('XX', '{}').format(*_val['value']) for _val in plot_configs]
    
#     # Now, we can check if the names make sense
#     files = [f for f in files if any([f2p+'_sde' in f for f2p in files2plot_full])]
#     # Make sure that the number of files is the same as the number of files2plot_full
#     assert len(files) == len(set(files2plot_full)), "The number of files to plot is not the same as the number of files2plot_full"

#     # Exract the figure and axis specifications
#     fig_specs = density_cfg['fig_args']
#     fig_specs['nrows'] = 2 # We will have two rows. The second row will show the prediction error while the first row will show the standard deviation
#     nrows = fig_specs['nrows']
#     ncols = fig_specs['ncols']

#     # Check if the number of subplots is the same as the number of files to plot
#     assert ncols >= len(files), "The number of subplots is not the same as the number of files to plot"

#     # Create the figure
#     fig, axs_2d = plt.subplots(**fig_specs)
#     # Flatten the axes
#     axs = axs_2d.flatten()

#     # Loop over the files
#     itr_count = 0
#     # Where the results are stored
#     _mesh_error_evol = np.zeros((len(files2plot_full), len(qdotgrid), len(qgrid)))
#     _mesh_std_evol = np.zeros((len(files2plot_full), len(qdotgrid), len(qgrid)))
#     _sampling_datas = []

#     for fname, pconf in zip(files2plot_full, plot_configs):
#         # Print the current step
#         print('\n###########################################################')
#         print(f"Creating mesh plot for {fname}")
#         print(fname)
#         # Get the the file name in files corresponding to the file name fname in files2plot_full
#         fname = [f for f in files if (fname + '_sde') in f]
#         print(f"File name: {fname}")
#         assert len(fname) == 1, "The number of files corresponding to the file name in files2plot_full is not 1"
#         model_name = fname[0]
#         # Extract the data name and model name
#         _train_data_name = model_name.split("__")[-1].split("_sde.pkl")[0]
#         # Load the training dataset
#         train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
#         # Create the samples from training dataset
#         # This is a 2D array containing all the transitions
#         train_data = np.array([ xev for (xev, _) in train_data])
#         _sampling_datas.append(train_data)

#         # Now let's create a model to compute prediction error and std on the meshgrid
#         # TODO: Make sure load learn model works for probabilistic models too
#         _mesh_learned_model, _ = load_learned_model(learned_dir+model_name, horizon=horizon_pred_accuracy, 
#                                                     num_samples=num_particles if 'node' not in model_name else 1, 
#                                                     ufun=None, prior_dist=pconf.get('prior_dist', False))

#         for _i in tqdm(range(len(qgrid))):
#             for _j in range(len(qdotgrid)):
#                 rng_key, gt_key, model_key = jax.random.split(rng_key, 3)
#                 xcurr = np.array([qgrid[_j,_i], qdotgrid[_j, _i]])
#                 _mesh_gtruth_data = np.array(_mesh_gtsampler(xcurr, gt_key)[0])
#                 _mesh_data = _mesh_learned_model(xcurr, model_key)
#                 _mesh_data = np.array(_mesh_data) # 3 dim array: (num_samples, horizon, dim)

#                 # Standard deviation first
#                 _mesh_std_evol[itr_count, _j, _i] = np.sum(np.std(_mesh_data[:,:horizon_uncertainty,:], axis=0))

#                 # COmpute error now
#                 _mean_result = np.mean(_mesh_data, axis=0)
#                 #rel_error_result = np.linalg.norm(_mean_result - _mesh_gtruth_data, axis=-1) / (np.linalg.norm(_mesh_gtruth_data, axis=-1) + 1e-10)
#                 rel_error_result = np.linalg.norm(_mean_result - _mesh_gtruth_data, axis=-1)
#                 _mesh_error_evol[itr_count, _j, _i] = np.sum(rel_error_result)
#                 # _mesh_error_evol[itr_count, _j, _i] = np.mean(rel_error_result)
        
#         itr_count += 1
    
#     # Let's compute the maximum error
#     _max_error = np.max(_mesh_error_evol)
        
#     # Now, let's plot the results
#     # Loop over the files
#     itr_count = 0
#     for pconf, ax in zip(plot_configs, axs):

#         # Plot the mesh
#         pcm = ax.pcolormesh(qgrid, qdotgrid, _mesh_error_evol[itr_count], vmin=0, vmax=_max_error,**density_cfg['mesh_args'])

#         # Set the title if 'title' is in pconf
#         if 'title' in pconf:
#             ax.set_title(pconf['title'])
        
#         # Set the x axis label
#         # Add xlabel only to the bottom row
#         if itr_count >= (nrows-1)*ncols:
#             ax.set_xlabel('$q$')
        
#         # Add ylabel only to the leftmost column
#         if itr_count % ncols == 0:
#             ax.set_ylabel('$\dot{q}$')
#         axs[itr_count+ncols].set_xlabel('$q$')
        
#         # Now let's add the color bar if this is the last subplot of the first row
#         if itr_count == ncols-1:
#             cbar = fig.colorbar(pcm, ax=axs[:ncols], **density_cfg['colorbar_args'])
#             cbar.ax.set_ylabel('Prediction Error', **density_cfg.get('extra_args',{}).get('label_colorbar_args', {}))

#         # Let's take care of the second row
#         _std_result = _mesh_std_evol[itr_count]
#         # Translate the standard deviation between 0 and 1
#         _std_result = (_std_result - np.min(_std_result)) / (np.max(_std_result) - np.min(_std_result))
#         pcm = axs[itr_count+ncols].pcolormesh(qgrid, qdotgrid, _std_result, vmin=0, vmax=1, **density_cfg['mesh_args'])

#         # Now let's add the color bar if this is the last subplot of the first row
#         if itr_count == ncols-1:
#             cbar = fig.colorbar(pcm, ax=axs[ncols:], **density_cfg['colorbar_args'])
#             cbar.ax.set_ylabel('Normalized Uncertainty', **density_cfg.get('extra_args',{}).get('label_colorbar_args', {}))

#         # Now we plot the training data
#         for _i, _traj_sample in enumerate(_sampling_datas[itr_count]):
#             ax.plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
#                                                                 'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
#         # Now we plot the training data
#         for _i, _traj_sample in enumerate(_sampling_datas[itr_count]):
#             axs[itr_count+ncols].plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
#                                                                 'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
#         itr_count += 1
    
#     # Set label for first ax only
#     axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

#     # Save the figure
#     density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
#     fig.savefig(**density_cfg['save_config'])

#     # Plot the figure
#     plt.show()


def create_state_prediction(cfg_path, learned_dir, data_dir, gt_dir):
    """ Creates the state prediction and error evolution plots
    """
    # Load the density configuration yaml
    density_cfg = load_yaml(cfg_path)

    # Check if learned_dir is empty, if so, use the default directory given by the current path
    if len(learned_dir) == 0:
        learned_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/'

    # Check if data_dir is empty, if so, use the default directory given by the current path
    if len(data_dir) == 0:
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'

    # The directory where the plots will be stored
    figure_out = data_dir + 'figures/'
    # Create the directory if it does not exist
    if not os.path.exists(figure_out):
        os.makedirs(figure_out)
    
    # Extra parameters for prediction and groundtruth
    num_particles = density_cfg['num_particles']
    horizon_accuracy = density_cfg['horizon_prediction']
    alpha_percentiles = density_cfg['alpha_percentiles']
    percentiles_array = density_cfg['percentiles_array']
    init_config = density_cfg['init_config']

    # For error evolution plots, we need the radius of the ball around xcurr and the number of points to sample
    radius = np.array(density_cfg['err_radius'])
    num_points = density_cfg['err_num_points']

    # Set numpy random seed
    np.random.seed(density_cfg['seed'])
    # Now sample uniformly the points around each xcurr in init_config
    xcurr_samples = [ np.random.uniform(-radius, radius, (num_points, 4)) + np.array(xcurr) for xcurr in init_config ]
    # Add the initial configuration to the list of samples
    xcurr_samples  = [ np.concatenate((np.array(xcurr)[None,:], xcurr_samples[i]), axis=0) for i, xcurr in enumerate(init_config) ]

    # Load the ground truth model
    _gtsampler, _, gt_time_evol = load_data_generator(gt_dir, noise_info={}, horizon=horizon_accuracy, ufun=None)
    _my_time = np.array(gt_time_evol)
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed'])

    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if '__DoPe_' in f and '_sde' in f]

    # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
    files2plot = density_cfg['files2plot']

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']
    files2plot_full = [ [files2plot.replace('XX', '{}').format(*_val_val) for _val_val in _val['value']] for _val in plot_configs]

    # Create a list of lists of files to plot
    files2plot_total = []
    for _f2p in files2plot_full:
        ftemp = []
        for _f2p_val in _f2p:
            fmatch = [f for f in files if _f2p_val+'_sde' in f]
            assert len(fmatch) == 1, "The number of files matching {} is not 1".format(_f2p_val)
            ftemp.append(fmatch[0])
        files2plot_total.append(ftemp)
    
    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    # Check if the axes has a shape attribute
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]

    # Plot style
    general_plot_style = density_cfg.get('general_style',{})
    curve_plot_style = density_cfg['curve_plot_style']

    # Loop over the files
    # itr_count = 0
    rng_key, gt_key = jax.random.split(rng_key)
    _dict_models_data = {}
    
    for row_models, pconf, ax in zip(files2plot_total, plot_configs, axs):
        # Set the x-axis labels
        if pconf.get('show_xlabel', True):
            ax.set_xlabel('Time (s)' if not isinstance(type2label[pconf['type']], list) else type2label[pconf['type']][0] )

        # Set the y-axis labels
        if pconf.get('show_ylabel', True):
            ax.set_ylabel(type2label[pconf['type']] if not isinstance(type2label[pconf['type']], list) else type2label[pconf['type']][1] )
        
        # Set the title
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the right-side title if it exists
        if 'title_right' in pconf:
            # Add a twin axis on the right with no ticks and the label given by title_right
            ax2 = ax.twinx()
            ax2.set_ylabel(pconf['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
            ax2.tick_params(axis='y', which='both', length=0)
            ax2.set_yticks([])

        first_iter = True
        for model_name, _style in zip(row_models, pconf['style']):
            # Now let's create a model to compute prediction error and std on the meshgrid
            if model_name not in _dict_models_data:
                if 'gaussian_mlp_ensemble' in model_name:
                    _mesh_learned_model, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_accuracy, num_samples=num_particles, ufun=None,  propagation_method='fixed_model')
                else:
                    # TODO: Make sure load learn model works for probabilistic models too
                    _mesh_learned_model, _ = load_learned_model(learned_dir+model_name, horizon=horizon_accuracy, num_samples=num_particles, ufun=None)
                _dict_models_data[model_name] = (_mesh_learned_model, {})
            _mesh_learned_model, _ipconf = _dict_models_data[model_name]

            # Check if pconf['init'] is in ipconf
            name2check = pconf['init'] if pconf['type'] != 't2Err' else '{}t2Err'.format(pconf['init'])
            if name2check not in _ipconf:
                # Get current samples
                curr_samples = xcurr_samples[pconf['init']] # This is a 2d array of shape (num_samples, dim)
                arr_save = ([], [])
                # Iterate through the samples
                for _i in range(curr_samples.shape[0]):
                    xinit = curr_samples[_i]
                    # Compute the groundtruth
                    _gtruth_data = np.array(_gtsampler(xinit, gt_key)[0])
                    # Model estimate
                    rng_key, model_key = jax.random.split(rng_key)
                    _data_evol = _mesh_learned_model(xinit, model_key)
                    _data_evol = np.array(_data_evol) # 3 dim array: (num_samples, horizon, dim)
                    # Append to the list
                    arr_save[0].append(_gtruth_data)
                    arr_save[1].append(_data_evol)
                    if pconf['type'] != 't2Err':
                        break
                _ipconf[name2check] = arr_save
            
            # Extract the data
            _gtruth_data, _data_evol = _ipconf[name2check]

            # _data_evol and _mean_evol have dim=4: q1, q2, q1dot, q2dot
            # We will plot each of these separately if provided in the type
            type2indx = {'t2q1': 0, 't2q2': 1, 't2q1dot': 2, 't2q2dot': 3}
            if pconf['type'] in type2indx:
                _gtruth_data, _data_evol = _gtruth_data[0], _data_evol[0]
                _indx = type2indx[pconf['type']]
                # Groundtruth plot first
                if first_iter:
                    ax.plot(_my_time, _gtruth_data[:,_indx], **{**general_plot_style, **curve_plot_style['groundtruth']})
                first_iter = False
                # Plot the mean model
                _mean_evol = np.mean(_data_evol, axis=0)
                ax.plot(_my_time, _mean_evol[:,_indx], **{**general_plot_style, **curve_plot_style[_style]})
                # Now let's sort the data along the number of samples
                q_data = _data_evol[:,:,_indx]
                q_data = np.sort(q_data, axis=0)
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * q_data.shape[0] )
                    q_bot = q_data[idx,:]
                    q_top = q_data[-idx,:]
                    ax.fill_between(_my_time, q_bot, q_top, alpha=_alph, color=curve_plot_style[_style]['color'])
                continue
            
            if pconf['type'] == 't2Err':
                # Compute the error for each sample
                # Error arr has dim=2: num_samples, horizon
                error_arr = np.array([ np.linalg.norm(c_gt - np.mean(c_model, axis=0), axis=-1) for c_gt, c_model in zip(_gtruth_data, _data_evol) ])
                # Maybe cumulate the error first
                error_arr = np.cumsum(error_arr, axis=-1) / np.arange(1, error_arr.shape[-1]+1)
                # Compute the mean and std
                mean_error = np.mean(error_arr, axis=0)
                std_error = np.std(error_arr, axis=0)
                # Plot the mean error
                ax.plot(_my_time, mean_error, **{**general_plot_style, **curve_plot_style[_style]})
                # Plot the std error
                ax.fill_between(_my_time, mean_error-std_error, mean_error+std_error, alpha=0.8, color=curve_plot_style[_style]['color'])
                continue

            # Raise error if type is not recognized
            raise ValueError('Type {} not recognized'.format(pconf['type']))

        # Set the grid
        ax.grid(True)

        # Axis constraints if given
        if 'axis_constraints' in pconf:
            for fname, fval in pconf['axis_constraints'].items():
                getattr(ax, fname)(**fval)
        

    # Set label for first ax only
    axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

    # Save the figure
    density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
    fig.savefig(**density_cfg['save_config'])

    # Plot the figure
    plt.show()



def create_error_vs_samples_plot(cfg_path, learned_dir, data_dir, gt_dir):
    """ Create the error vs samples plot """

    # Load the density configuration yaml
    density_cfg = load_yaml(cfg_path)

    # Check if learned_dir is empty, if so, use the default directory given by the current path
    if len(learned_dir) == 0:
        learned_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/'

    # Check if data_dir is empty, if so, use the default directory given by the current path
    if len(data_dir) == 0:
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'

    # The directory where the plots will be stored
    figure_out = data_dir + 'figures/'
    # Create the directory if it does not exist
    if not os.path.exists(figure_out):
        os.makedirs(figure_out)
    
    # Extra parameters for prediction and groundtruth
    num_particles = density_cfg['num_particles']
    horizon_accuracy = density_cfg['horizon_prediction']
    init_config = density_cfg['init_config']

    # For error evolution plots, we need the radius of the ball around xcurr and the number of points to sample
    radius = np.array(density_cfg['err_radius'])
    num_points = density_cfg['err_num_points']
    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if '__DoPe_' in f and '_sde' in f]

    # Set numpy random seed
    np.random.seed(density_cfg['seed'])
    # Now sample uniformly the points around each xcurr in init_config
    if num_points > 0:
        xcurr_samples = [ np.random.uniform(-radius, radius, (num_points, 4)) + np.array(xcurr) for xcurr in init_config ]
        # Add the initial configuration to the list of samples
        xcurr_samples  = [ np.concatenate((np.array(xcurr)[None,:], xcurr_samples[i]), axis=0) for i, xcurr in enumerate(init_config) ]
    else:
        xcurr_samples = [ np.array(xcurr)[None,:] for xcurr in init_config ]
        

    # Load the ground truth model
    _gtsampler, _, _ = load_data_generator(gt_dir, noise_info={}, horizon=horizon_accuracy, ufun=None)
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed'])
    
    # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
    files2plot = density_cfg['files2plot']
    data_ids = density_cfg['data_ids']
    arr_trajectories = np.sort([ int(_v) for _v in data_ids])

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']

    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    # Check if the axes has a shape attribute
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]

    # Plot style
    general_plot_style = density_cfg.get('general_style',{})
    curve_plot_style = density_cfg['curve_plot_style']

    # Loop over the files
    # itr_count = 0
    rng_key, gt_key = jax.random.split(rng_key)
    _dict_models_data = {}

    for pconf, ax in zip(plot_configs, axs):
        # Set the x-axis labels
        if pconf.get('show_xlabel', True):
            ax.set_xlabel('Number of training trajectories')

        # Set the y-axis labels
        if pconf.get('show_ylabel', True):
            ax.set_ylabel(type2label[pconf['type']])
        
        # Set the title
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the right-side title if it exists
        if 'title_right' in pconf:
            # Add a twin axis on the right with no ticks and the label given by title_right
            ax2 = ax.twinx()
            ax2.set_ylabel(pconf['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
            ax2.tick_params(axis='y', which='both', length=0)
            ax2.set_yticks([])
        
        for _model_name, _style in zip(pconf['models'], pconf['style']):
            _res_array = []

            for _data_num in arr_trajectories:
                if _model_name in _dict_models_data:
                    _res_array = _dict_models_data[_model_name]
                    break
                # Extract the model to load
                model_name = files2plot.replace('XX', '{}').format(_model_name, _data_num)
                # Get the unique file with _model_name in files
                model_file = [f for f in files if model_name+'_sde' in f]
                assert len(model_file) == 1, 'There should be only one file with {} in {}'.format(model_name, files)
                model_file = model_file[0]
                # print('Loading model {}, {}'.format(model_file, model_name))
                # Load the model
                if 'gaussian_mlp_ensemble' in model_name:
                    _mesh_learned_model, _ = load_learned_ensemble_model(learned_dir+model_file, horizon=horizon_accuracy, num_samples=num_particles, ufun=None,  propagation_method='fixed_model')
                else:
                    # TODO: Make sure load learn model works for probabilistic models too
                    _mesh_learned_model, _ = load_learned_model(learned_dir+model_file, horizon=horizon_accuracy, num_samples=num_particles, ufun=None)

                # Get current samples
                curr_samples = xcurr_samples[pconf['init']] # This is a 2d array of shape (num_samples, dim)
                arr_save = ([], [])
                # Iterate through the samples
                for _i in range(curr_samples.shape[0]):
                    xinit = curr_samples[_i]
                    # Compute the groundtruth
                    _gtruth_data = np.array(_gtsampler(xinit, gt_key)[0])
                    # Model estimate
                    rng_key, model_key = jax.random.split(rng_key)
                    _data_evol = _mesh_learned_model(xinit, model_key)
                    _data_evol = np.array(_data_evol) # 3 dim array: (num_samples, horizon, dim)
                    # Append to the list
                    arr_save[0].append(_gtruth_data)
                    arr_save[1].append(_data_evol)
                # Extract the data
                _gtruth_data, _data_evol = arr_save
                error_arr = np.array([ np.linalg.norm(c_gt - np.mean(c_model, axis=0), axis=-1) for c_gt, c_model in zip(_gtruth_data, _data_evol) ])
                std_pred_arr = np.array([ np.linalg.norm(np.std(c_model, axis=0), axis=-1) for c_model in _data_evol ]) # [:, :100]
                # Total mean error
                # np.cumsum(error_arr, axis=-1) / np.arange(1, error_arr.shape[-1]+1)
                # total_error = (np.cumsum(error_arr, axis=-1) / np.arange(1, error_arr.shape[-1]+1))[:, -1]
                total_error = np.sum(error_arr, axis=-1) / error_arr.shape[-1]
                mean_error = np.mean(total_error)
                std_error = np.std(total_error)
                # Total mean std
                total_std = np.sum(std_pred_arr, axis=-1) / std_pred_arr.shape[-1]
                mean_std = np.mean(total_std)
                std_std = np.std(total_std)
                # Append to the results array
                _res_array.append([mean_error, std_error, mean_std, std_std])

            if _model_name not in _dict_models_data:
                _dict_models_data[_model_name] = _res_array

            _res_array = np.array(_res_array)
            # print(_res_array)
            # Plot the results
            if pconf['type'] == 'traj2error':
                # Plot the mean error
                ax.plot(arr_trajectories, _res_array[:,0], **{**general_plot_style, **curve_plot_style[_style]})
                # Plot the std error
                ax.fill_between(arr_trajectories, _res_array[:,0]-_res_array[:,1], _res_array[:,0]+_res_array[:,1], alpha=0.4, color=curve_plot_style[_style]['color'])
            elif pconf['type'] == 'traj2std':
                # Plot the mean error
                ax.plot(arr_trajectories, _res_array[:,2], **{**general_plot_style, **curve_plot_style[_style]})
                # Plot the std error
                ax.fill_between(arr_trajectories, _res_array[:,2]-_res_array[:,3], _res_array[:,2]+_res_array[:,3], alpha=0.4, color=curve_plot_style[_style]['color'])
            else:
                raise ValueError('Unknown type {}'.format(pconf['type']))
        
        # Set the grid
        ax.grid(True)

        # Axis constraints if given
        if 'axis_constraints' in pconf:
            for fname, fval in pconf['axis_constraints'].items():
                getattr(ax, fname)(**fval)
        

    # Set label for first ax only
    axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

    # Save the figure
    density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
    fig.savefig(**density_cfg['save_config'])

    # Plot the figure
    plt.show()
        


def show_dataset(cfg_path, data_dir):
    """ Load, show and save the dataset """

    # Load the density configuration yaml
    density_cfg = load_yaml(cfg_path)
    # Check if data_dir is empty, if so, use the default directory given by the current path
    if len(data_dir) == 0:
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
    # The directory where the plots will be stored
    figure_out = data_dir + 'figures/'
    # Create the directory if it does not exist
    if not os.path.exists(figure_out):
        os.makedirs(figure_out)
    
    # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
    files2plot = density_cfg['files2plot']
    data_ids = density_cfg['data_ids']

    # Extract the figure arguments
    fig_args = density_cfg['fig_args']
    fig_args['nrows'] = 2
    fig_args['ncols'] = 2
    fig_args['sharex'] = True

    # Loop over the data_ids, then replace XX in the template with the data_id and load the dataset
    for data_id in data_ids:
        # Replace XX with the data_id
        files2plot_id = files2plot.replace('XX', data_id)
        # Load the dataset
        # Load the training dataset
        train_data = _load_pkl(data_dir + files2plot_id + ".pkl")
        # Create the samples from training dataset
        # This is a 2D array containing all the transitions
        train_data = np.array([ xev for (xev, _) in train_data])
        # Create the figure
        fig, axs = plt.subplots(**fig_args)
        axs_flatten = axs.flatten()
        assert fig_args['nrows'] == 2 and fig_args['ncols'] == 2, "The figure must have 4 subplots"
        # Loop over the subplots
        for _i, ax in enumerate(axs_flatten):
            # Axis information
            ax.set_xlabel('Time step')
            ax.set_ylabel(indx2label[_i])
            for _k, _data in enumerate(train_data):
                ax.plot(_data[:,_i], linestyle='none', marker = 'o', markersize = 8)
            ax.grid(True)
        # Save the figure
        save_name = figure_out + files2plot_id + '.png'
        fig.savefig(save_name, **density_cfg.get('save_config', {}))
    # Plot the figure
    plt.show()


if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Double Pendulum Model, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='density', help='The function to run')
    parser.add_argument('--model_dir', type=str, default='double_pendulum.yaml', help='The model configuration and groundtruth file')
    parser.add_argument('--cfg_path', type=str, default='config_pred_analysis.yaml', help='The data generation and training configuration file')
    parser.add_argument('--learned_dir', type=str, default='', help='The directory where all the models are stored')
    parser.add_argument('--data_dir', type=str, default='', help='The directory where trajectories data are stored')



    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'density':
        create_density_mesh_plots(args.cfg_path, args.learned_dir, args.data_dir)
    
    if args.fun == 'uncertainty':
        create_uncertainty_plots(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)
    
    if args.fun == 'state':
        create_state_prediction(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)
    
    if args.fun == 'show_dataset':
        show_dataset(args.cfg_path, args.data_dir)
    
    if args.fun == 'sample_efficiency':
        create_error_vs_samples_plot(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)