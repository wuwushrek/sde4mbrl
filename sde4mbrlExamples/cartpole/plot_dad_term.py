
import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from cartpole_sde import load_predictor_function, load_learned_diffusion, _load_pkl, load_trajectory
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

# files2plot_full = [ [files2plot.replace('XX', '{}').format(*_val_val) for _val_val in _val['value']] for _val in plot_configs]
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
    # # Check if data_dir is empty, if so, use the default directory given by the current path
    # if len(data_dir) == 0:
    #     data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'

    # The directory where the plots will be stored
    figure_out = data_dir + 'figures/'

    # Extract the grid limits for the mesh plot
    theta = density_cfg['thetarange']
    thetadot = density_cfg['thetadotrange']
    # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
    theta_grid, thetadot_grid = np.meshgrid(np.linspace(theta[0], theta[1], density_cfg['num_grid_theta']),
                                    np.linspace(thetadot[0], thetadot[1], density_cfg['num_grid_thetadot']))
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed_mesh'])

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']
    files2plot_full = [ _val['value'] for _val in plot_configs]
    
    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    # Check if the number of subplots is the same as the number of files to plot
    assert nrows*ncols == len(files2plot_full), "The number of subplots is not the same as the number of files to plot"

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]

    # Loop over the files
    itr_count = 0
    for fname, pconf, ax in zip(files2plot_full, plot_configs, axs):
        # Print the current step
        print('\n###########################################################')
        print(f"Creating mesh plot for {fname}")
        print(fname)
        model_name = fname

        # Extract the data name and model name
        _train_data_name = density_cfg.get('train_data', pconf.get('train_data', None))
        assert _train_data_name is not None, "The training data name is not specified in the config file"

        # Load the training dataset
        train_data = load_trajectory(data_dir + _train_data_name)
        # Create the samples from training dataset
        # This is a 2D array containing all the transitions
        train_data = np.array([ _dict_x['y'][:,2:] for _dict_x in train_data])
        # train_data = train_data.reshape(-1, train_data.shape[-1])

        # Load the model
        _diff_est_fn = load_learned_diffusion(learned_dir+model_name, num_samples=density_cfg['num_particles_diffusion'])
        
        # Iterate over the meshgrid and compute the prediction error and std
        _mesh_pred_density = np.zeros((len(thetadot_grid), len(theta_grid)))
        # Check if the diffusion should becomputed using the actual density network or sigmoid
        net_val = density_cfg.get('net', pconf.get('net', False))

        for _i in tqdm(range(len(theta_grid))):
            for _j in range(len(thetadot_grid)):
                rng_key, density_key = jax.random.split(rng_key)
                xcurr = np.array([0.0, 0.0, theta_grid[_j,_i], thetadot_grid[_j, _i]])
                # Compute the density
                _mesh_pred_density[_j,_i] = float(_diff_est_fn(xcurr, density_key, net=net_val)[1]) # Second output is the density
        
        # Scale the mesh between 0 and 1 if we are using the density network directly instead of a sigmoid of the density network
        if net_val:
            _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

        # Plot the mesh
        pcm = ax.pcolormesh(theta_grid, thetadot_grid, _mesh_pred_density, vmin=0, vmax=1,**density_cfg['mesh_args'])

        # Set the title if 'title' is in pconf
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the x axis label
        # Add xlabel only to the bottom row
        if itr_count >= (nrows-1)*ncols:
            ax.set_xlabel(r'$\theta$')
        
        # Add ylabel only to the leftmost column
        if itr_count % ncols == 0:
            ax.set_ylabel(r'$\dot{\theta}$')

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

    # # Save the figure
    # density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
    # fig.savefig(**density_cfg['save_config'])

    # Plot the figure
    plt.show()

if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Mass Spring Damper Model, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='dad', help='The function to run')
    parser.add_argument('--density_cfg', type=str, default='config_plot_dad_term.yaml', help='The data generation adn training configuration file')
    parser.add_argument('--learned_dir', type=str, default='', help='The directory where all the models are stored')
    parser.add_argument('--data_dir', type=str, default='', help='The directory where trajectories data are stored')


    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'dad':
        create_density_mesh_plots(args.density_cfg, args.learned_dir, args.data_dir)