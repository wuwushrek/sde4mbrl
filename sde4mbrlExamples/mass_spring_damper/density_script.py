""" This script generate the dataset and plots used in the paper
"""
import os
import jax
import numpy as np
import matplotlib.pyplot as plt


from mass_spring_model import gen_traj_yaml, train_models_on_dataset, load_learned_diffusion, _load_pkl
from sde4mbrl.utils import load_yaml

from tqdm.auto import tqdm

MODEL_NAME = "nesde"

def train_density_model(model_dir, density_cfg):
    """ Train the density model for different values of the parameters and different trajectories
    """
    # Load the density configuration yaml
    density_cfg = load_yaml(density_cfg)
    # Extract the modified parameters
    modified_params = density_cfg['density_trainer']
    # Extract the range of values for the density model
    for gval in density_cfg['grad_pen_range_values']:
        for brad in density_cfg['ball_radius_range']:
            for mucoeff in density_cfg['mu_coeff_range']:
                modified_params['sde_loss']['pen_grad_density'] = gval
                modified_params['sde_loss']['density_loss']['ball_radius'] = brad
                modified_params['sde_loss']['density_loss']['mu_coeff'] = mucoeff
                model_name = f"nesde_Grad{gval}_Rad{brad}_Mu{mucoeff}"
                train_models_on_dataset(model_dir, model_name, 'DensityExp', modified_params=modified_params)


def create_density_mesh_plots(density_cfg, learned_dir, data_dir, net=False):
    """ Create the mesh plots for the density model
        learned_dir: The directory where all the models are stored
    """
    # Load the density configuration yaml
    density_cfg = load_yaml(density_cfg)
    # Check if learned_dir is empty, if so, use the default directory given by the current path
    if len(learned_dir) == 0:
        learned_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/density_models/'
    # Check if data_dir is empty, if so, use the default directory given by the current path
    if len(data_dir) == 0:
        data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/density_dataset/'
    # The directory where the plots will be stored
    figure_out = data_dir + 'density_plots/'

    # Extract the grid limits for the mesh plot
    qrange = density_cfg['qrange']
    qdotrange = density_cfg['qdotrange']
    # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
    qgrid, qdotgrid = np.meshgrid(np.linspace(qrange[0], qrange[1], density_cfg['num_grid_q']),
                                    np.linspace(qdotrange[0], qdotrange[1], density_cfg['num_grid_qdot']))
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed_mesh'])

    # Extract all files in the directory containing DensityExp in learned_dir and edning with _sde.pkl
    files = [f for f in os.listdir(learned_dir) if 'DensityExp' in f and f.endswith('_sde.pkl')]

    # Extract files2plot which provide a template in form of GradXX_RadXX_MuXX__DensityExp_XX
    files2plot = density_cfg['files2plot']
    assert 'XX' in files2plot, "The template for the files to plot must contain at least one XX"

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    plot_configs = density_cfg['plot_configs']
    files2plot_full = [ files2plot.replace('XX', '{}').format(*_val['value']) for _val in plot_configs]
    
    # Now, each name in files2plot_full can be separated using __ into GradXX_RadXX_MuXX and DensityExp_XX
    # The first part is the name of the model, the second part is the name of the trajectory
    # We need to remove files that do not contains DensityExp_XX in files2plot_full
    files = [f for f in files if any([f2p.split('__')[1] in f for f2p in files2plot_full])]
    # Now, we need to remove files that do not contain GradXX_RadXX_MuXX in files2plot_full
    files = [f for f in files if any([f2p.split('DensityExp')[0] in f for f2p in files2plot_full])]
    # Make sure that the number of files is the same as the number of files2plot_full
    assert len(files) == len(files2plot_full), "The number of files to plot is not the same as the number of files2plot_full"

    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    # Check if the number of subplots is the same as the number of files to plot
    assert nrows*ncols == len(files), "The number of subplots is not the same as the number of files to plot"

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    # Flatten the axes
    axs = axs_2d.flatten()

    # Loop over the files
    itr_count = 0
    for fname, pconf, ax in zip(files2plot_full, plot_configs, axs):
        # Print the current step
        print('\n###########################################################')
        print(f"Creating mesh plot for {fname}")
        print(fname)
        # Get the the file name in files corresponding to the file name fname in files2plot_full
        fname = [f for f in files if fname.split('DensityExp')[0] in f and fname.split('__')[1] in f]
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
        _diff_est_fn = load_learned_diffusion(learned_dir+model_name, num_samples=density_cfg['num_particles_diffusion'])
        # Iterate over the meshgrid and compute the prediction error and std
        _mesh_pred_density = np.zeros((len(qdotgrid), len(qgrid)))
        for _i in tqdm(range(len(qgrid))):
            for _j in range(len(qdotgrid)):
                rng_key, density_key = jax.random.split(rng_key)
                xcurr = np.array([qgrid[_j,_i], qdotgrid[_j, _i]])
                # Compute the density
                _mesh_pred_density[_j,_i] = float(_diff_est_fn(xcurr, density_key, net=net)[1]) # Second output is the density
        
        # Scale the mesh between 0 and 1 if we are using the density network directly instead of a sigmoid of the density network
        if net:
            _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

        # Plot the mesh
        pcm = ax.pcolormesh(qgrid, qdotgrid, _mesh_pred_density, vmin=0, vmax=1,**density_cfg['mesh_args'])

        # Set the title if 'title' is in pconf
        if 'title' in pconf:
            ax.set_title(pconf['title'])
        
        # Set the x axis label
        # Add xlabel only to the bottom row
        if itr_count >= (nrows-1)*ncols:
            ax.set_xlabel('$q$')
        
        # Add ylabel only to the leftmost column
        if itr_count % ncols == 0:
            ax.set_ylabel('$\dot{q}$')

        if 'title_right' in pconf:
            # Add a twin axis on the right with no ticks and the label given by title_right
            ax2 = ax.twinx()
            ax2.set_ylabel(pconf['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
            ax2.tick_params(axis='y', which='both', length=0)
            ax2.set_yticks([])

        # # Set the colorbar for each row
        # if (itr_count+1) % ncols == 0:
        #     cbar = fig.colorbar(pcm, ax=axs[itr_count+1-ncols:itr_count+1], **density_cfg['colorbar_args'])
        #     if 'title_right' in pconf:
        #         # Same font and size as the title
        #         cbar.ax.set_ylabel(pconf['title_right'], rotation=270, labelpad=20)
            
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
    density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
    fig.savefig(**density_cfg['save_config'])

    # Plot the figure
    plt.show()




if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Mass Spring Damper Model, Data Generator, and Trainer')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='gen_traj', help='The function to run')
    parser.add_argument('--model_dir', type=str, default='mass_spring_damper.yaml', help='The model configuration and groundtruth file')
    parser.add_argument('--density_cfg', type=str, default='config_density_dataset.yaml', help='The data generation adn training configuration file')
    parser.add_argument('--learned_dir', type=str, default='', help='The directory where all the models are stored')
    parser.add_argument('--data_dir', type=str, default='', help='The directory where trajectories data are stored')


    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'gen_traj':
        gen_traj_yaml(args.model_dir, args.density_cfg)
    
    if args.fun == 'train':
        train_density_model(args.model_dir, args.density_cfg)
    
    if args.fun == 'plot':
        create_density_mesh_plots(args.density_cfg, args.learned_dir, args.data_dir)