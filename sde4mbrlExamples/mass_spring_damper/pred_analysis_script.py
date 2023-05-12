""" This script generate the dataset and plots used in the paper
    for prediction error and uncertainty analysis over the learned models
"""
import os
import jax
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mass_spring_model import load_data_generator, load_learned_model, load_learned_diffusion, _load_pkl
from sde4mbrl.utils import load_yaml
from mbrlLibUtils.save_and_load_models import load_learned_ensemble_model

from tqdm.auto import tqdm

type2label = {
  "t2q": "$q$",
  "t2qdot": "$\dot{q}$",
  "q2qdot": ["$q$", "$\dot{q}$"],
  "t2Err": "Prediction Error"
}


def get_size_paper(width_pt, fraction=1, subplots=(1,1)):
    """ Get the size of the figure in inches
        width_pt: Width of the figure in latex points
        fraction: Fraction of the width which you wish the figure to occupy
        subplots: The number of rows and columns
    """
    # Width of the figure in inches
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

def create_density_mesh_plots(cfg_path, learned_dir, data_dir):
    """ Create the mesh plots for the density model
    Args:
        cfg_path: The path to the config file
        learned_dir: The directory where all the models are stored
        data_dir: The directory where the dataset is stored
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

    # Extract the grid limits for the mesh plot
    qrange = density_cfg['qrange']
    qdotrange = density_cfg['qdotrange']
    # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
    qgrid, qdotgrid = np.meshgrid(np.linspace(qrange[0], qrange[1], density_cfg['num_grid_q']),
                                    np.linspace(qdotrange[0], qdotrange[1], density_cfg['num_grid_qdot']))
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed'])

    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if '__MSD_' in f and '_sde' in f]

    # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
    files2plot = density_cfg['files2plot']

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']
    files2plot_full = [ files2plot.replace('XX', '{}').format(*_val['value']) for _val in plot_configs]
    
    # Now, we can check if the names make sense
    files = [f for f in files if any([f2p+'_sde' in f for f2p in files2plot_full])]
    # Make sure that the number of files is the same as the number of files2plot_full
    assert len(files) == len(set(files2plot_full)), "The number of files to plot is not the same as the number of files2plot_full"

    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    # Check if the number of subplots is the same as the number of files to plot
    assert nrows*ncols >= len(files), "The number of subplots is not the same as the number of files to plot"

    # Check if use_tex is enabled
    use_pgf = density_cfg.get('use_pgf', False)
    use_pdf = density_cfg.get('use_pdf', False)
    if use_pgf:
        mpl.use("pgf")
    
    # Modify the figure size if use_pgf is enabled or use_pdf is enabled
    if use_pgf or use_pdf:
        paper_width_pt = density_cfg['paper_width_pt']
        fraction = density_cfg['fraction']
        # Get the size of the figure in inches
        fig_size = get_size_paper(paper_width_pt, fraction=fraction, subplots=(nrows, ncols))
        # Set the figure size
        fig_specs['figsize'] = fig_size
        # Set the rcParams
        plt.rcParams.update(density_cfg['texConfig'])

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]
    
    # Check if diffusion is enabled or model std is enabled
    compute_density = density_cfg.get('compute_density', True)

    if not compute_density:
        # Extra parameters for prediction and groundtruth
        num_particles = density_cfg['num_particles_std']
        horizon_uncertainty = density_cfg['horizon_uncertainty']

    # Loop over the files
    itr_count = 0
    for fname, pconf, ax in zip(files2plot_full, plot_configs, axs):
        # Print the current step
        print('\n###########################################################')
        print(f"Creating mesh plot for {fname}")
        print(fname)
        # Get the the file name in files corresponding to the file name fname in files2plot_full
        fname = [f for f in files if (fname + '_sde') in f]
        print(f"File name: {fname}")
        assert len(fname) == 1, "The number of files corresponding to the file name in files2plot_full is not 1"
        model_name = fname[0]
        # Extract the data name and model name
        _train_data_name = model_name.split("__")[-1].split("_sde.pkl")[0]
        # Load the training dataset
        train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
        # Create the samples from training dataset
        # This is a 2D array containing all the transitions
        train_data = np.array([ xev for (xev, _) in train_data])
    
        # Load the model
        if compute_density:
            __diff_est_fn = load_learned_diffusion(learned_dir+model_name, num_samples=density_cfg['num_particles_diffusion'])
            _diff_est_fn = lambda _x, _key, net: __diff_est_fn(_x, _key, net=net)[1] # Second output is the density
        else:
            __diff_est_fn,_ = load_learned_model(learned_dir+model_name, horizon=horizon_uncertainty, 
                                                        num_samples=num_particles, 
                                                        ufun=None, prior_dist=False)
            _diff_est_fn = lambda _x, _key, net: np.sum(np.std(__diff_est_fn(_x, _key), axis=0))
        
        # Iterate over the meshgrid and compute the prediction error and std
        _mesh_pred_density = np.zeros((len(qdotgrid), len(qgrid)))
        # Check if the diffusion should becomputed using the actual density network or sigmoid
        net_val = density_cfg.get('net', pconf.get('net', False))

        for _i in tqdm(range(len(qgrid))):
            for _j in range(len(qdotgrid)):
                rng_key, density_key = jax.random.split(rng_key)
                xcurr = np.array([qgrid[_j,_i], qdotgrid[_j, _i]])
                # Compute the density
                _mesh_pred_density[_j,_i] = float(_diff_est_fn(xcurr, density_key, net_val)) # Second output is the density
        
        # Scale the mesh between 0 and 1 if we are using the density network directly instead of a sigmoid of the density network
        if net_val:
            _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

        # Plot the mesh
        vmin = density_cfg.get('vmin', pconf.get('vmin', None))
        vmax = density_cfg.get('vmax', pconf.get('vmax', None))
        pcm = ax.pcolormesh(qgrid, qdotgrid, _mesh_pred_density, vmin=vmin, vmax=vmax, **density_cfg['mesh_args'])

        # Set the title if 'title' is in pconf
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the x axis label
        # Add xlabel only to the bottom row
        if itr_count >= (nrows-1)*ncols:
            ax.set_xlabel(r'$q$')
        
        # Add ylabel only to the leftmost column
        if itr_count % ncols == 0:
            ax.set_ylabel(r'$\dot{q}$')

        if 'title_right' in pconf:
            # Add a twin axis on the right with no ticks and the label given by title_right
            ax2 = ax.twinx()
            ax2.set_ylabel(pconf['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
            ax2.tick_params(axis='y', which='both', length=0)
            ax2.set_yticks([])
        
        itr_count += 1

        # Now we plot the training data
        for _i, _traj_sample in enumerate(train_data):
            ax.plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
                                                                'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
    # Set label for first ax only
    axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

    # Plot the colorbar
    _ = fig.colorbar(pcm, ax=axs, **density_cfg['colorbar_args'])

    # Save the figure
    if 'save_config' in density_cfg:
        density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
        if use_pgf:
            # Replace the extension with pgf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pgf'
            fig.savefig(output_name, format='pgf')
        
        if use_pdf:
            # Replace the extension with pdf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pdf'
            fig.savefig(output_name, format='pdf', bbox_inches='tight')

        fig.savefig(**density_cfg['save_config'], bbox_inches='tight')

    # Plot the figure
    if use_pgf:
        return
    
    # Plot the figure
    plt.show()



def find_dictkey_contained_in_str(_dict, _str):
    """ Find the key in _dict that is contained in _str
    Args:
        _dict: The dictionary
        _str: The string
    Returns:
        The key in _dict that is contained in _str
    """
    list_key = []
    for _key in _dict.keys():
        if _key in _str:
            list_key.append(_key)
    assert len(list_key) == 1, "The number of keys in _dict contained in _str is not 1"
    return list_key[0]


def create_uncertainty_plots(cfg_path, learned_dir, data_dir, gt_dir):
    """ Create the mesh plots for displaying uncertainty in the learned models and prediction error
    Args:
        cfg_path: The path to the config file
        learned_dir: The directory where all the models are stored
        data_dir: The directory where the dataset is stored
        gt_dir: The directory where the ground truth model
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

    # Extract the grid limits for the mesh plot
    qrange = density_cfg['qrange']
    qdotrange = density_cfg['qdotrange']
    # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
    qgrid, qdotgrid = np.meshgrid(np.linspace(qrange[0], qrange[1], density_cfg['num_grid_q']),
                                    np.linspace(qdotrange[0], qdotrange[1], density_cfg['num_grid_qdot']))
    
    # Extra parameters for prediction and groundtruth
    num_particles = density_cfg['num_particles_std']
    horizon_pred_accuracy = density_cfg['horizon_pred_accuracy']
    horizon_uncertainty = density_cfg.get('horizon_uncertainty', horizon_pred_accuracy)

    # Load the ground truth model
    _mesh_gtsampler, _, _ = load_data_generator(gt_dir, noise_info={}, horizon=horizon_pred_accuracy, ufun=None)
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed'])

    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if '__MSD_' in f and '_sde' in f]
    print(files)

    # Extract files2plot which provide a template in form of XX__MSD_XX_XX_XX
    files2plot = density_cfg['files2plot']

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']
    files2plot_full = [ files2plot.replace('XX', '{}').format(*_val['value']) for _val in plot_configs]
    
    # Now, we can check if the names make sense
    files = [f for f in files if any([f2p+'_sde' in f for f2p in files2plot_full])]
    # Make sure that the number of files is the same as the number of files2plot_full
    assert len(files) == len(set(files2plot_full)), "The number of files to plot is not the same as the number of files2plot_full"

    # Exract the figure and axis specifications
    # We will have two rows. The second row will show the prediction error while the first row will show the standard deviation
    fig_specs = density_cfg['fig_args']
    fig_specs['nrows'] = 2
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    # Check if the number of subplots is the same as the number of files to plot
    assert ncols >= len(files), "The number of subplots is not the same as the number of files to plot"

    # Check if use_tex is enabled
    use_pgf = density_cfg.get('use_pgf', False)
    use_pdf = density_cfg.get('use_pdf', False)
    if use_pgf:
        mpl.use("pgf")
    
    # Modify the figure size if use_pgf is enabled or use_pdf is enabled
    if use_pgf or use_pdf:
        paper_width_pt = density_cfg['paper_width_pt']
        fraction = density_cfg['fraction']
        # Get the size of the figure in inches
        fig_size = get_size_paper(paper_width_pt, fraction=fraction, subplots=(nrows, ncols))
        # Set the figure size
        fig_specs['figsize'] = fig_size
        # Set the rcParams
        plt.rcParams.update(density_cfg['texConfig'])

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]

    # Loop over the files
    itr_count = 0
    # Where the results are stored
    _mesh_error_evol = np.zeros((len(files2plot_full), len(qdotgrid), len(qgrid)))
    _mesh_std_evol = np.zeros((len(files2plot_full), len(qdotgrid), len(qgrid)))
    _sampling_datas = []

    for fname, pconf in zip(files2plot_full, plot_configs):
        # Print the current step
        print('\n###########################################################')
        print(f"Creating mesh plot for {fname}")
        print(fname)
        # Get the the file name in files corresponding to the file name fname in files2plot_full
        fname = [f for f in files if (fname + '_sde') in f]
        print(f"File name: {fname}")
        assert len(fname) == 1, "The number of files corresponding to the file name in files2plot_full is not 1"
        model_name = fname[0]
        # Extract the data name and model name
        if 'gaussian_mlp_ensemble' in model_name:
            _train_data_name = model_name.split("__")[1].split("_sde")[0]
        else:
            _train_data_name = model_name.split("__")[-1].split("_sde.pkl")[0]
        # Load the training dataset
        train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
        # Create the samples from training dataset
        # This is a 2D array containing all the transitions
        train_data = np.array([ xev for (xev, _) in train_data])
        _sampling_datas.append(train_data)

        if 'gaussian_mlp_ensemble' in model_name:
            __mesh_learned_model, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_pred_accuracy+1, 
                                                        num_samples=num_particles, 
                                                        ufun=None, 
                                                        propagation_method=density_cfg.get('gaussian_propagation_method', 'fixed_model'),
                                                        rseed=density_cfg['seed'],
                                                        device=density_cfg.get('device', 'cpu'),
                                                        )
            _mesh_learned_model = lambda x, key: __mesh_learned_model(x, None, key)
        else:
            # Now let's create a model to compute prediction error and std on the meshgrid
            # TODO: Make sure load learn model works for probabilistic models too
            _mesh_learned_model, _ = load_learned_model(learned_dir+model_name, horizon=horizon_pred_accuracy, 
                                                        num_samples=num_particles if 'node' not in model_name else 1, 
                                                        ufun=None, prior_dist=pconf.get('prior_dist', False))

        # Strategy for mean error
        _mean_error_strat = pconf.get('mean_type', 'mean')
        for _i in tqdm(range(len(qgrid))):
            for _j in range(len(qdotgrid)):
                rng_key, gt_key, model_key = jax.random.split(rng_key, 3)
                xcurr = np.array([qgrid[_j,_i], qdotgrid[_j, _i]])
                _mesh_gtruth_data = np.array(_mesh_gtsampler(xcurr, gt_key)[0])
                _mesh_data = _mesh_learned_model(xcurr, model_key)
                _mesh_data = np.array(_mesh_data) # 3 dim array: (num_samples, horizon, dim)

                # Standard deviation first
                _mesh_std_evol[itr_count, _j, _i] = np.sum(np.std(_mesh_data[:,:horizon_uncertainty,:], axis=0))

                # COmpute error now
                if _mean_error_strat == 'median':
                    _mean_result = np.median(_mesh_data, axis=0)
                else:
                    _mean_result = np.mean(_mesh_data, axis=0)
                #rel_error_result = np.linalg.norm(_mean_result - _mesh_gtruth_data, axis=-1) / (np.linalg.norm(_mesh_gtruth_data, axis=-1) + 1e-10)
                rel_error_result = np.linalg.norm(_mean_result - _mesh_gtruth_data, axis=-1)
                _mesh_error_evol[itr_count, _j, _i] = np.sum(rel_error_result)
                # _mesh_error_evol[itr_count, _j, _i] = np.mean(rel_error_result)
        
        itr_count += 1
    
    # Let's compute the maximum error to display
    _max_error = np.max(_mesh_error_evol)
    # Check if the user has specified a maximum error
    _max_error = density_cfg.get('max_error', _max_error)
        
    # Now, let's plot the results
    # Loop over the files
    itr_count = 0
    for pconf, ax in zip(plot_configs, axs):

        # Plot the mesh
        pcm = ax.pcolormesh(qgrid, qdotgrid, _mesh_error_evol[itr_count], vmin=0, vmax=_max_error,**density_cfg['mesh_args'])

        # Set the title if 'title' is in pconf
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the x axis label
        # Add xlabel only to the bottom row
        if itr_count >= (nrows-1)*ncols:
            ax.set_xlabel(r'$q$')
        
        # Add ylabel only to the leftmost column
        if itr_count % ncols == 0:
            ax.set_ylabel(r'$\dot{q}$')
        axs[itr_count+ncols].set_xlabel('$q$')
        axs[ncols].set_ylabel(r'$\dot{q}$')
        
        # Now let's add the color bar if this is the last subplot of the first row
        if itr_count == ncols-1:
            cbar = fig.colorbar(pcm, ax=axs[:ncols], **density_cfg['colorbar_args'])
            cbar.ax.set_ylabel('Err. prediction', **density_cfg.get('extra_args',{}).get('label_colorbar_args', {}))

        # Let's take care of the second row
        _std_result = _mesh_std_evol[itr_count]
        print(np.min(_std_result), np.max(_std_result))
        # Translate the standard deviation between 0 and 1
        # Normalize the mesh
        if density_cfg.get('normalize_mesh', False):
            if 'node' in pconf['value'][0]:
                _std_result = np.zeros_like(_std_result)
            else:
                _std_result = (_std_result - np.min(_std_result)) / (np.max(_std_result) - np.min(_std_result))
            # _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

        # Plot the mesh
        vmin = density_cfg.get('vmin', density_cfg.get('vmin', None))
        vmax = density_cfg.get('vmax', density_cfg.get('vmax', None))

        # _std_result = (_std_result - np.min(_std_result)) / (np.max(_std_result) - np.min(_std_result))
        pcm = axs[itr_count+ncols].pcolormesh(qgrid, qdotgrid, _std_result, vmin=vmin, vmax=vmax, **density_cfg['mesh_args'])

        # Now let's add the color bar if this is the last subplot of the first row
        if itr_count == ncols-1:
            cbar = fig.colorbar(pcm, ax=axs[ncols:], **density_cfg['colorbar_args'])
            cbar.ax.set_ylabel('Uncertainty', **density_cfg.get('extra_args',{}).get('label_colorbar_args', {}))

        # Now we plot the training data
        for _i, _traj_sample in enumerate(_sampling_datas[itr_count]):
            ax.plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
                                                                'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
        # Now we plot the training data on the second row
        for _i, _traj_sample in enumerate(_sampling_datas[itr_count]):
            axs[itr_count+ncols].plot(_traj_sample[:,0], _traj_sample[:,1], **{**density_cfg['training_dataset_config'], 
                                                                'label' : density_cfg['training_dataset_config']['label'] if _i == 0 else None} )
        
        itr_count += 1
    
    # Set label for first ax only
    axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

    # Save the figure
    if 'save_config' in density_cfg:
        density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
        if use_pgf:
            # Replace the extension with pgf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pgf'
            fig.savefig(output_name, format='pgf')
        
        if use_pdf:
            # Replace the extension with pdf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pdf'
            fig.savefig(output_name, format='pdf', bbox_inches='tight')

        fig.savefig(**density_cfg['save_config'], bbox_inches='tight')

    # Plot the figure
    plt.show()


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
    
    # Extra parameters for prediction and groundtruth
    num_particles = density_cfg['num_particles']
    horizon_accuracy = density_cfg['horizon_prediction']
    alpha_percentiles = density_cfg['alpha_percentiles']
    percentiles_array = density_cfg['percentiles_array']
    init_config = density_cfg['init_config']

    # Load the ground truth model
    _gtsampler, _, gt_time_evol = load_data_generator(gt_dir, noise_info={}, horizon=horizon_accuracy, ufun=None)
    _my_time = np.array(gt_time_evol)
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed'])

    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if '__MSD_' in f and '_sde' in f]

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
    
    for row_models, pconf, ax in zip(files2plot_total, plot_configs, axs):
        # Set the axis labels
        if pconf.get('show_xlabel', True):
            ax.set_xlabel('Time (s)' if not isinstance(type2label[pconf['type']], list) else type2label[pconf['type']][0] )
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
            # Extract the data name and model name
            if 'gaussian_mlp_ensemble' in model_name:
                _train_data_name = model_name.split("__")[1].split("_sde")[0]
            else:
                _train_data_name = model_name.split("__")[-1].split("_sde.pkl")[0]
            # Load the training dataset
            train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
            # Create the samples from training dataset
            # This is a 2D array containing all the transitions
            train_data = np.array([ xev for (xev, _) in train_data])
            # Now let's create a model to compute prediction error and std on the meshgrid
            # TODO: Make sure load learn model works for probabilistic models too
            # _mesh_learned_model, _my_time = load_learned_model(learned_dir+model_name, horizon=horizon_accuracy, num_samples=num_particles, ufun=None)
            if 'gaussian_mlp_ensemble' in model_name:
                _mesh_learned_model, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_accuracy, num_samples=num_particles, ufun=None,  propagation_method='fixed_model')
            else:
                # Now let's create a model to compute prediction error and std on the meshgrid
                # TODO: Make sure load learn model works for probabilistic models too
                _mesh_learned_model, _ = load_learned_model(learned_dir+model_name, horizon=horizon_accuracy, num_samples=num_particles, ufun=None)

            # _my_time = np.array(_my_time)
            # Extract the initial state for integration
            xinit = np.array(init_config[pconf['init']])
            # Compute the groundtruth
            _gtruth_data = np.array(_gtsampler(xinit, gt_key)[0])
            # Model estimate
            rng_key, model_key = jax.random.split(rng_key)
            _data_evol = _mesh_learned_model(xinit, model_key)
            _data_evol = np.array(_data_evol) # 3 dim array: (num_samples, horizon, dim)
            _mean_evol = np.mean(_data_evol, axis=0)

            # Let's check the type and plot according
            if pconf['type'] == 'q2qdot':
                # Groundtruth plot first
                if first_iter:
                    ax.plot(_gtruth_data[:,0], _gtruth_data[:,1], **{**general_plot_style, **curve_plot_style['groundtruth']})
                # Plot the mean model
                ax.plot(_mean_evol[:,0], _mean_evol[:,1], **{**general_plot_style, **curve_plot_style[_style]})
                # If the training data is required, plot it
                if pconf.get('show_dataset', True) and first_iter:
                    for _i, _xev in enumerate(train_data):
                        ax.plot(_xev[:,0], _xev[:,1], **{**general_plot_style, **curve_plot_style['dataset'], 'label' : None if _i > 0 else curve_plot_style['dataset']['label']})
                first_iter = False
                continue
            
            if pconf['type'] == 't2q':
                # Groundtruth plot first
                if first_iter:
                    ax.plot(_my_time, _gtruth_data[:,0], **{**general_plot_style, **curve_plot_style['groundtruth']})
                first_iter = False
                # Plot the mean model
                ax.plot(_my_time, _mean_evol[:,0], **{**general_plot_style, **curve_plot_style[_style]})
                # Now let's sort the data along the number of samples
                q_data = _data_evol[:,:,0]
                q_data = np.sort(q_data, axis=0)
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * q_data.shape[0] )
                    q_bot = q_data[idx,:]
                    q_top = q_data[-idx,:]
                    ax.fill_between(_my_time, q_bot, q_top, alpha=_alph, color=curve_plot_style[_style]['color'])
                continue
            
            if pconf['type'] == 't2qdot':
                # Groundtruth plot first
                if first_iter:
                    ax.plot(_my_time, _gtruth_data[:,1], **{**general_plot_style, **curve_plot_style['groundtruth']})
                first_iter = False
                # Plot the mean model
                ax.plot(_my_time, _mean_evol[:,1], **{**general_plot_style, **curve_plot_style[_style]})
                # Now let's sort the data along the number of samples
                qdot_data = _data_evol[:,:,1]
                qdot_data = np.sort(qdot_data, axis=0)
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * qdot_data.shape[0] )
                    qdot_bot = qdot_data[idx,:]
                    qdot_top = qdot_data[-idx,:]
                    ax.fill_between(_my_time, qdot_bot, qdot_top, alpha=_alph, color=curve_plot_style[_style]['color'])
                continue
            
            if pconf['type'] == 't2Err':
                # Compute the error
                _err_mean = np.linalg.norm(_mean_evol - _gtruth_data, axis=1)
                _err_mean = np.cumsum(_err_mean) / np.arange(1, _err_mean.shape[0]+1)

                # _err_evol = np.linalg.norm(_data_evol - _gtruth_data, ord=2, axis=2)
                # _err_evol = np.cumsum(_err_evol, axis=1) # / np.arange(1, _err_evol.shape[1]+1)
                # _err_mean = np.mean(_err_evol, axis=0)

                ax.plot(_my_time,_err_mean,  **{**general_plot_style, **curve_plot_style[_style]})
                # _err_evol_std = np.std(_err_evol, axis=0)
                # _err_mean = np.mean(_err_evol, axis=0)

                # _err_mean = np.cumsum(_err_mean) / np.arange(1, _err_mean.shape[0]+1)
                # _err_evol_std = np.cumsum(_err_evol_std) / np.arange(1, _err_evol_std.shape[0]+1)

                # # Make it a cumulative sum along the horizon
                # _err_evol = np.cumsum(_err_evol, axis=1) / np.arange(1, _err_evol.shape[1]+1)
                # Plot the mean model
                # ax.plot(_my_time, _err_mean, **{**general_plot_style, **curve_plot_style[_style]})
                # Plot the std
                # ax.fill_between(_my_time, _err_mean - _err_evol_std, _err_mean + _err_evol_std, alpha=0.5, color=curve_plot_style[_style]['color'])
                # Now sort the data along the number of samples
                # _err_sorted = np.sort(_err_evol, axis=0)
                # for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                #     idx = int( (1 - _perc) / 2.0 * _err_sorted.shape[0] )
                #     err_bot = _err_sorted[idx,:]
                #     err_top = _err_sorted[-idx,:]
                #     ax.fill_between(_my_time, err_bot, err_top, alpha=_alph, color=curve_plot_style[_style]['color'])
                
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


def plot_prediction_accuracy(cfg_path, learned_dir, data_dir, gt_dir):
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

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    model2plot = density_cfg['model2plot']

    # Curve plot
    curve_plot_style = density_cfg['curve_plot_style']
    density_cfg['std_style'] = density_cfg.get('std_style', None)
    alpha_std = density_cfg.get('alpha_std', 0.3)

    # Percentiles
    alpha_percentiles = density_cfg['alpha_percentiles']
    percentiles_array = density_cfg['percentiles_array']
    general_style = density_cfg.get('general_style', {})
    
    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']
    # fig_specs['nrows'] = 1
    # fig_specs['ncols'] = 4
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']
    # fig_specs['sharex'] = True
    assert nrows * ncols == 4, "The number of subplots must be 4"

    # Check if use_tex is enabled
    use_pgf = density_cfg.get('use_pgf', False)
    use_pdf = density_cfg.get('use_pdf', False)
    if use_pgf:
        mpl.use("pgf")
    
    # Modify the figure size if use_pgf is enabled or use_pdf is enabled
    if use_pgf or use_pdf:
        paper_width_pt = density_cfg['paper_width_pt']
        fraction = density_cfg['fraction']
        # Get the size of the figure in inches
        fig_size = get_size_paper(paper_width_pt, fraction=fraction, subplots=(nrows, ncols))
        # Set the figure size
        fig_specs['figsize'] = fig_size
        # Set the rcParams
        plt.rcParams.update(density_cfg['texConfig'])

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]
    
    # Initial state for the prediction
    init_config = density_cfg['init_config']
    xinit = np.array(init_config)

    # Horizon for evaluation and number of particles
    horizon_eval = density_cfg['horizon_eval']
    num_particles_eval = density_cfg['num_particles_eval']

    # Load the ground truth model
    _gtsampler, _, gt_time_evol = load_data_generator(gt_dir, noise_info={}, horizon=horizon_eval, ufun=None)
    _my_time = np.array(gt_time_evol)
    # Groundtruth prediction
    gt_key = jax.random.PRNGKey(density_cfg['seed_eval'])
    _gtruth_data = np.array(_gtsampler(xinit, gt_key)[0])

    figure_label = [r'$q$', r'$\dot{q}$', r'$\dot{q}$', r'Cum. Avg. Pred. Error']
    xlabels = ['Time (s)', 'Time (s)', r'$q$', 'Time (s)']

    # Load the training dataset
    _train_data_name = density_cfg['train_data_name']
    train_data = _load_pkl(data_dir + _train_data_name + ".pkl")
    train_data = np.array([ xev for (xev, _) in train_data])

    if 'data_spacing' in density_cfg:
        _my_time = _my_time[::density_cfg['data_spacing']]
        _gtruth_data = _gtruth_data[::density_cfg['data_spacing']]
        train_data = train_data[:, ::density_cfg['data_spacing'], :]

    # Loop over the files
    first_model = True
    for model_name in model2plot:
        # Get the style for the current model
        model_style = find_dictkey_contained_in_str(curve_plot_style, model_name)
        model_style = curve_plot_style[model_style]

        # Model actual filename
        model_name = model_name + '__' + _train_data_name + '_sde.pkl'
        # Extract the predictor model
        if 'gaussian_mlp_ensemble' in model_name:
            __mesh_learned_model, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_eval+1, 
                                                        num_samples=num_particles_eval, 
                                                        ufun=None, 
                                                        propagation_method=density_cfg.get('gaussian_propagation_method', 'fixed_model'),
                                                        rseed=density_cfg['seed'],
                                                        device=density_cfg.get('device', 'cpu'),
                                                        )
            _pred_fn = lambda x, key: __mesh_learned_model(x, None, key)
        else:
            # Now let's create a model to compute prediction error and std on the meshgrid
            # TODO: Make sure load learn model works for probabilistic models too
            _pred_fn, _ = load_learned_model(learned_dir+model_name, horizon=horizon_eval, 
                                                        num_samples=num_particles_eval if 'node' not in model_name else 1, 
                                                        ufun=None, prior_dist=density_cfg.get('prior_dist', False))
        
        # Extract the initial state for integration
        rng_key = jax.random.PRNGKey(density_cfg['seed_eval'])
        _xpred  = _pred_fn(xinit, rng_key)
        # Space the data
        if 'data_spacing' in density_cfg:
            _xpred = _xpred[:, ::density_cfg['data_spacing'], :]

        x_pred_mean = np.mean(_xpred, axis=0)
        x_pred_std = np.std(_xpred, axis=0)

        # pred_error = np.linalg.norm(_xpred - _gtruth_data[None], axis=-1)
        pred_error = np.linalg.norm(np.mean(_xpred, axis=0) - _gtruth_data, axis=-1)[None]
        cum_pred_error = np.cumsum(pred_error, axis=1) / np.arange(1, pred_error.shape[1]+1)[None]
        # cum_pred_error = np.cumsum(pred_error) / np.arange(1, pred_error.shape[0]+1)

        for _i, ax in enumerate(axs):
            mean_style = model_style.copy()
            mean_style.pop('color_std', None)
            mean_style.pop('std_style', None)
            std_style = model_style.get('std_style', density_cfg['std_style'])

            # Only set the xlabel for the last row
            ax.set_xlabel(xlabels[_i])

            ax.set_ylabel(figure_label[_i])
            ax.grid(True)
            ax.autoscale(enable=True, axis='both', tight=None)

            if _i == len(axs)-1:
                # For the last plot, plot the cumulative prediction error
                pred_error_mean = np.mean(cum_pred_error, axis=0)
                pred_error_std = np.std(cum_pred_error, axis=0)
                ax.plot(_my_time, pred_error_mean, 
                            **{**general_style, **mean_style,
                                'label' : model_style['label'] if _i == 0 else None
                            }
                )
                ax.fill_between(_my_time, pred_error_mean - pred_error_std, pred_error_mean + pred_error_std,
                                    linewidth=0.0, alpha=alpha_std, color=model_style['color_std']
                                )
                continue
            
            if _i == len(axs)-2:
                # Print the training dataset
                if 'dataset' in curve_plot_style:
                    for _j, _xev in enumerate(train_data):
                        ax.plot(_xev[:,0], _xev[:,1], 
                                **{**general_style, **curve_plot_style['dataset'], 
                                'label' : None if _j > 0 or not first_model else curve_plot_style['dataset']['label']
                                }
                                )
                # It's the x-y plot
                # First plot the groundtruth
                ax.plot(_gtruth_data[:,0], _gtruth_data[:,1],
                            **{**general_style, **curve_plot_style['groundtruth'],
                                'label' : None
                            }
                )
                # Then plot the mean prediction
                ax.plot(x_pred_mean[:,0], x_pred_mean[:,1],
                            **{**general_style, **mean_style,
                                'label' : model_style['label'] if _i == 0 else None
                            }
                )
                # Do a fill_between in 2D by adding patches
                if 'nesde' not in model_name:
                    continue
                
                # for xind in range(x_pred_mean.shape[0]):
                #     for yind in range(x_pred_mean.shape[0]):
                #         print(xind, yind)
                #         # Create a patch
                #         rect = Rectangle((x_pred_mean[xind,0] - x_pred_std[xind,0], x_pred_mean[yind,1] - x_pred_std[yind,1]), 
                #                                     2*x_pred_std[xind,0], 2*x_pred_std[yind,1], 
                #                                     linewidth=0.0, alpha=alpha_std, color=model_style['color_std']
                #                                 )
                #         ax.add_patch(rect)
                continue
            
            if 'ylim' in density_cfg and density_cfg['ylim'][_i] is not None:
                ax.set_ylim(density_cfg['ylim'][_i])


            # Plot the groundtruth
            if first_model:
                ax.plot(_my_time, _gtruth_data[:, _i], 
                            **{**general_style, **curve_plot_style['groundtruth'], 
                                    'label' : curve_plot_style['groundtruth']['label'] if _i == 0 and first_model else None
                                } 
                        )

            # Plot the mean prediction accuracy
            if std_style is None:
                for _j in range(_xpred.shape[0]):
                    ax.plot(_my_time, _xpred[_j, :, _i], 
                                **{**general_style, **mean_style, 
                                        'label' : model_style['label'] if _i == 0 and _j == 0 else None
                                    } 
                            )
            else:
                ax.plot(_my_time, x_pred_mean[:, _i], 
                            **{**general_style, **mean_style, 
                                    'label' : model_style['label'] if _i == 0 else None
                                } 
                        )
            
            if 'node' in model_name:
                first_model = False
                continue

            # Plot the std depending on the std_style
            if std_style == 'std':
                ax.fill_between(_my_time, x_pred_mean[:, _i] - x_pred_std[:, _i], 
                                            x_pred_mean[:, _i] + x_pred_std[:, _i], 
                                            linewidth=0.0, 
                                            alpha=alpha_std, color=model_style['color_std']
                                        )
            elif std_style == 'perc':
                state_sorted = np.sort(_xpred[:, :, _i], axis=0)
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * state_sorted.shape[0] )
                    q_bot = state_sorted[idx,:]
                    q_top = state_sorted[-idx,:]
                    ax.fill_between(_my_time, q_bot, q_top, alpha=_alph, linewidth=0.0, color=model_style['color_std'])
            elif std_style == 'perc75':
                percetile25, percentile75 = np.percentile(_xpred[:, :, _i], [25, 75], axis=0)
                ax.fill_between(_my_time, percetile25, percentile75, alpha=alpha_std, linewidth=0.0, color=model_style['color_std'])
        first_model = False
    
    # Collect all the labels and show them in the legend
    if  density_cfg.get('global_legend', True) == True:
        fig.legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))

    # Save the figure
    if 'save_config' in density_cfg:
        density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
        if use_pgf:
            # Replace the extension with pgf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pgf'
            fig.savefig(output_name, format='pgf')
        
        if use_pdf:
            # Replace the extension with pdf
            output_name = density_cfg['save_config']['fname'].split('.')[:-1]
            output_name = '.'.join(output_name) + '.pdf'
            fig.savefig(output_name, format='pdf', bbox_inches='tight')

        fig.savefig(**density_cfg['save_config'], bbox_inches='tight')
    
    if 'save_config_tex' in density_cfg.keys():
        axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))
        tikzplotlib_fix_ncols(fig)
        import tikzplotlib
        density_cfg['save_config_tex']['fname'] = figure_out + density_cfg['save_config_tex']['fname']
        tikzplotlib.clean_figure(fig)
        tikzplotlib.save(density_cfg['save_config_tex']['fname'], figure=fig)

    plt.show()

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Mass Spring Damper Model, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='density', help='The function to run')
    parser.add_argument('--model_dir', type=str, default='mass_spring_damper.yaml', help='The model configuration and groundtruth file')
    parser.add_argument('--cfg_path', type=str, default='config_pred_analysis.yaml', help='The data generation and training configuration file')
    parser.add_argument('--learned_dir', type=str, default='', help='The directory where all the models are stored')
    parser.add_argument('--data_dir', type=str, default='', help='The directory where trajectories data are stored')



    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'density':
        create_density_mesh_plots(args.cfg_path, args.learned_dir, args.data_dir)
    
    if args.fun == 'unc':
        create_uncertainty_plots(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)
    
    if args.fun == 'state':
        create_state_prediction(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)
    
    if args.fun == 'pred':
        plot_prediction_accuracy(args.cfg_path, args.learned_dir, args.data_dir, args.model_dir)