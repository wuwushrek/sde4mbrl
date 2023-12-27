
import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from cartpole_sde import load_predictor_function, load_learned_diffusion, _load_pkl, load_trajectory
from sde4mbrl.utils import load_yaml

from mbrlLibUtils.save_and_load_models import load_learned_ensemble_model

from tqdm.auto import tqdm

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
    figure_out = 'my_data/figures/'

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
        train_data = load_trajectory(_train_data_name)
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
        vmin = density_cfg.get('vmin', pconf.get('vmin', 0))
        vmax = density_cfg.get('vmax', pconf.get('vmax', 1))
        pcm = ax.pcolormesh(theta_grid, thetadot_grid, _mesh_pred_density, vmin=vmin, vmax=vmax, **density_cfg['mesh_args'])

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

    # Save the figure
    if 'save_config' in density_cfg.keys():
        density_cfg['save_config']['fname'] = figure_out + density_cfg['save_config']['fname']
        fig.savefig(**density_cfg['save_config'])

    # Plot the figure
    plt.show()


def plot_prediction_accuracy(cfg_path):
    """ Plot the prediction accuracy of the learned models
    """
    # Load the density configuration yaml
    density_cfg = load_yaml(cfg_path)
    # Director containing the models
    learned_dir = 'my_models/'
    # The directory where the plots will be stored
    figure_out = 'my_data/figures/'

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
    fig_specs['nrows'] = 2
    fig_specs['ncols'] = 3
    fig_specs['sharex'] = True
    nrows, ncols = fig_specs['nrows'], fig_specs['ncols']

    # Check if use_tex is enabled
    use_pgf = density_cfg.get('use_pgf', False)
    use_pdf = density_cfg.get('use_pdf', False)

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
    
    # Load the dataset used for evaluation of the models
    train_data = load_trajectory(density_cfg['eval_data'])

    # Pick the trajectories on which the models will be evaluated
    num_traj_eval = density_cfg['num_traj_eval']
    if 'seed_traj_eval' in density_cfg:
        seed_traj_eval = density_cfg['seed_traj_eval']
        np.random.seed(seed_traj_eval) # Set the seed for reproducibility
        # Pick the indices of the trajectories
        idx_traj_eval = np.random.choice(len(train_data), num_traj_eval, replace=False)
    else:
        idx_traj_eval = np.arange(num_traj_eval)
    xevol = np.array([ train_data[_indx]['y'] for _indx in idx_traj_eval])
    uevol = np.array([ train_data[_indx]['u'] for _indx in idx_traj_eval])
    
    # Horizon for evaluation and number of particles
    horizon_eval = density_cfg['horizon_eval']
    num_particles_eval = density_cfg['num_particles_eval']
    # Where the trajectory for evaluation will start
    start_eval_strategy = density_cfg.get('start_eval_strategy', 'first')
    if start_eval_strategy == 'first':
        start_eval = 0
    elif start_eval_strategy == 'random':
        start_eval = np.random.randint(0, xevol.shape[1]-(horizon_eval+1))
    else:
        raise NotImplementedError(f"Start evaluation strategy {start_eval_strategy} is not implemented")
    
    # Select the fragment of the trajectories to evaluate
    xevol = xevol[:, start_eval:start_eval+horizon_eval+1, :]
    uevol = uevol[:, start_eval:start_eval+horizon_eval, :]

    # The time step of integration
    time_steps = np.array([ i * 0.02 for i in range(horizon_eval+1)])

    # Function to convert from x, xdot, theta, thetadot to x, xdot, sin(theta), cos(theta), thetadot
    def _convert_to_cartpole_state(_x, array_lib=jax.numpy):
        return array_lib.array([_x[0], _x[1], array_lib.sin(_x[2]), array_lib.cos(_x[2]), _x[3]])
    
    figure_label = [r'$p_x$', r'$\dot{p}_x$', r'$\sin(\theta)$', r'$\cos(\theta)$', r'$\dot{\theta}$', r'Cum. Pred. Error']
    xlabel = r'Time (s)'

    # Loop over the files
    first_model = True
    for model_name in model2plot:
        # check if _sde is in the model name or gaussian_mlp
        if '_sde' in model_name:
            # Load an SDE model
            _pred_fn = load_predictor_function(learned_dir+model_name+'.pkl', 
                                            modified_params={
                                                'horizon' : horizon_eval, 
                                                'num_particles' : num_particles_eval,
                                                'stepsize' : 0.02
                                            }
                        )
            
            def _pred_fn_convert(_x, _u, _key):
                res = _pred_fn(_x, _u, _key)
                return jax.vmap(jax.vmap(lambda _v : _convert_to_cartpole_state(_v)))(res)
            
            # Convert the function to a jitted function
            pred_fn = jax.jit(_pred_fn_convert, backend=density_cfg.get('jax_backend', 'cpu'))

        elif 'gaussian_mlp' in model_name:

            # Load an ensemble model
            _pred_fn, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_eval+1,
                            num_samples=num_particles_eval,
                            propagation_method=density_cfg.get('gaussian_propagation_method', 'fixed_model'),
                            rseed=density_cfg['seed_eval']
                        )
            
            pred_fn = lambda _x, _u, _key : _pred_fn(_convert_to_cartpole_state(_x, np), _u, _key)

        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")
        
        # Compute the prediction error
        rng_key = jax.random.PRNGKey(density_cfg['seed_eval'])
        x_pred_list = []
        for _i in tqdm(range(num_traj_eval)):
            rng_key, eval_key = jax.random.split(rng_key)
            # Predicted trajectory
            _xpred = pred_fn(xevol[_i,0,:], uevol[_i,:,:], eval_key)
            x_pred_list.append(_xpred)
        
        # Convert the list to an array
        x_pred = np.array(x_pred_list)
        x_pred_mean = np.mean(x_pred, axis=1)
        x_pred_std = np.std(x_pred, axis=1)

        # convert the xevol to cartpole state
        xevol_converted = np.array([ [ _convert_to_cartpole_state(_x) for _x in _xevol] for _xevol in xevol])
        print(x_pred.shape, xevol_converted.shape)
        # Compute the prediction error wrt mean trajectory
        pred_error = np.linalg.norm(x_pred_mean - xevol_converted, axis=-1)

        # # Compute the prediction error
        # xevol_con_reshaped = xevol_converted[None].transpose(1,0,2,3)
        # # pred_error = np.linalg.norm(x_pred_mean - xevol_converted, axis=-1)
        # pred_error = np.linalg.norm(x_pred - xevol_con_reshaped, axis=-1)
        # pred_error = np.mean(pred_error, axis=0)

        # First let's plot the groundtruth and predicted quantities
        indx2plot = density_cfg.get('indx2plot', 0)
        for _i, ax in enumerate(axs):

            mean_style = curve_plot_style[model_name].copy()
            mean_style.pop('color_std', None)
            mean_style.pop('std_style', None)
            std_style = curve_plot_style[model_name].get('std_style', density_cfg['std_style'])

            # Only set the xlabel for the last row
            if _i >= (fig_specs['nrows']-1)*fig_specs['ncols']:
                ax.set_xlabel(xlabel)

            ax.set_ylabel(figure_label[_i])
            ax.grid(True)
            ax.autoscale(enable=True, axis='both', tight=None)

            if _i == len(axs)-1:
                # For the last plot, plot the cumulative prediction error
                pred_error = np.cumsum(pred_error, axis=1) / np.arange(1, pred_error.shape[1]+1)[None]
                pred_error_mean = np.mean(pred_error, axis=0)
                pred_error_std = np.std(pred_error, axis=0)
                # cum_pred_error = np.cumsum(pred_error_mean) / np.arange(1, pred_error_mean.shape[0]+1)
                cum_pred_error = pred_error_mean
                ax.plot(time_steps, cum_pred_error, 
                            **{**general_style, **mean_style,
                                'label' : curve_plot_style[model_name]['label'] if _i == 0 else None
                            }
                )
                ax.fill_between(time_steps, cum_pred_error - pred_error_std, cum_pred_error + pred_error_std,
                                    linewidth=0.0, alpha=alpha_std, color=curve_plot_style[model_name]['color_std']
                                )
                continue

            if 'ylim' in density_cfg and density_cfg['ylim'][_i] is not None:
                ax.set_ylim(density_cfg['ylim'][_i])


            # Plot the groundtruth
            if first_model:
                ax.plot(time_steps, xevol_converted[indx2plot, :, _i], 
                            **{**general_style, **curve_plot_style['groundtruth'], 
                                    'label' : curve_plot_style['groundtruth']['label'] if _i == 0 and first_model else None
                                } 
                        )

            # Plot the mean prediction accuracy
            # print(std_style)
            if std_style is None:
                for _j in range(x_pred.shape[1]):
                    ax.plot(time_steps, x_pred[indx2plot, _j, :, _i], 
                                **{**general_style, **mean_style, 
                                        'label' : curve_plot_style[model_name]['label'] if _i == 0 and _j == 0 else None
                                    } 
                            )
            else:
                ax.plot(time_steps, x_pred_mean[indx2plot, :, _i], 
                            **{**general_style, **mean_style, 
                                    'label' : curve_plot_style[model_name]['label'] if _i == 0 else None
                                } 
                        )
            
            # Plot the std depending on the std_style
            if std_style == 'std':
                ax.fill_between(time_steps, x_pred_mean[indx2plot, :, _i] - x_pred_std[indx2plot, :, _i], 
                                            x_pred_mean[indx2plot, :, _i] + x_pred_std[indx2plot, :, _i], 
                                            linewidth=0.0, alpha=alpha_std, color=curve_plot_style[model_name]['color_std']
                                        )
            elif std_style == 'perc':
                state_sorted = np.sort(x_pred[indx2plot, :, :, _i], axis=0)
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * state_sorted.shape[0] )
                    q_bot = state_sorted[idx,:]
                    q_top = state_sorted[-idx,:]
                    ax.fill_between(time_steps, q_bot, q_top, alpha=_alph, linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
            elif std_style == 'perc75':
                percetile25, percentile75 = np.percentile(x_pred[indx2plot, :, :, _i], [25, 75], axis=0)
                ax.fill_between(time_steps, percetile25, percentile75, alpha=alpha_std, linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
            elif std_style == 'minmax':
                min_state = np.min(x_pred[indx2plot, :, :, _i], axis=0)
                max_state = np.max(x_pred[indx2plot, :, :, _i], axis=0)
                ax.fill_between(time_steps, min_state, max_state, alpha=alpha_std, linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
        first_model = False
    
    # Collect all the labels and show them in the legend
    lines = []
    labels = []
    for ax in axs:
        l, l_ = ax.get_legend_handles_labels()
        # Add only non-duplicate labels
        for _l, _l_ in zip(l, l_):
            if _l_ not in labels:
                lines.append(_l)
                labels.append(_l_)
    # fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.legend(lines, labels, **density_cfg.get('extra_args',{}).get('legend_args', {}))

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
        # axs[0].legend(**density_cfg.get('extra_args',{}).get('legend_args', {}))
        tikzplotlib_fix_ncols(fig)
        import tikzplotlib
        density_cfg['save_config_tex']['fname'] = figure_out + density_cfg['save_config_tex']['fname']
        tikzplotlib.clean_figure(fig)
        tikzplotlib.save(density_cfg['save_config_tex']['fname'], figure=fig)

    plt.show()

# def tikzplotlib_fix_ncols(obj):
#     """
#     workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
#     """
#     if hasattr(obj, "_ncols"):
#         obj._ncol = obj._ncols
#     for child in obj.get_children():
#         tikzplotlib_fix_ncols(child)

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    if hasattr(obj, "_dash_pattern"):
        obj._us_dashOffset = obj._dash_pattern[0]
        obj._us_dashSeq = obj._dash_pattern[1]
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
    

def create_uncertainty_plot(cfg_path):
    # Load the density configuration yaml
    density_cfg = load_yaml(cfg_path)
    # Director containing the models
    learned_dir = 'my_models/'
    # The directory where the plots will be stored
    figure_out = 'my_data/figures/'
    if not os.path.exists(figure_out):
        os.makedirs(figure_out)

    # Extract the plot configs. This is a list of dictionaries, where the number of elements is the number of subplots
    model2plot = density_cfg['model2plot']
    
    # Exract the figure and axis specifications
    fig_specs = density_cfg['fig_args']
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    assert nrows*ncols == len(model2plot), "The number of subplots is not the same as the number of files to plot"

    # Check if use_tex is enabled
    use_pgf = density_cfg.get('use_pgf', False)
    use_pdf = density_cfg.get('use_pdf', False)
    
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
    
    # Load the dataset used for evaluation of the models
    if isinstance(density_cfg['train_data'], str):
        density_cfg['train_data'] = [density_cfg['train_data']] * len(model2plot)
    
    train_datas = []
    for _train_data_name in density_cfg['train_data']:
        _train_data = load_trajectory(_train_data_name)
        _train_data = np.array([ _dict_x['y'][:,2:] for _dict_x in _train_data])
        train_datas.append(_train_data)

    # Extract the grid limits for the mesh plot
    theta = density_cfg['thetarange']
    thetadot = density_cfg['thetadotrange']
    # Create a meshgrid using num_grid_q and num_grid_qdot in the density configuration
    theta_grid, thetadot_grid = np.meshgrid(np.linspace(theta[0], theta[1], density_cfg['num_grid_theta']),
                                    np.linspace(thetadot[0], thetadot[1], density_cfg['num_grid_thetadot']))
    
    # Create a random PRNG key
    rng_key = jax.random.PRNGKey(density_cfg['seed_mesh'])

    # Horizon for evaluation and number of particles
    horizon_eval = density_cfg['horizon_eval']
    num_particles_eval = density_cfg['num_particles_eval']

    # Function to convert from x, xdot, theta, thetadot to x, xdot, sin(theta), cos(theta), thetadot
    def _convert_to_cartpole_state(_x, array_lib=jax.numpy):
        return array_lib.array([_x[0], _x[1], array_lib.sin(_x[2]), array_lib.cos(_x[2]), _x[3]])
    
    # Get the curve style for the uncertainty plot
    curve_plot_style = density_cfg['curve_plot_style']

    # Loop file
    itr_count = 0
    for ax, model_name in zip(axs, model2plot):
        model_style = curve_plot_style.get(model_name, {})
        # check if _sde is in the model name or gaussian_mlp
        if '_sde' in model_name:
            # Load an SDE model
            _pred_fn = load_predictor_function(learned_dir+model_name+'.pkl', 
                                            modified_params={
                                                'horizon' : horizon_eval, 
                                                'num_particles' : num_particles_eval,
                                                'stepsize' : 0.02
                                            }
                        )
            
            def _pred_fn_convert(_x, _u, _key):
                res = _pred_fn(_x, _u, _key)
                return jax.vmap(jax.vmap(lambda _v : _convert_to_cartpole_state(_v)))(res)
            
            # Convert the function to a jitted function
            pred_fn = jax.jit(_pred_fn_convert, backend=density_cfg.get('jax_backend', 'cpu'))

        elif 'gaussian_mlp' in model_name:

            # Load an ensemble model
            _pred_fn, _ = load_learned_ensemble_model(learned_dir+model_name, horizon=horizon_eval+1,
                            num_samples=num_particles_eval,
                            propagation_method=density_cfg.get('gaussian_propagation_method', 'fixed_model'),
                            rseed=density_cfg['seed_mesh']
                        )
            
            pred_fn = lambda _x, _u, _key : _pred_fn(_convert_to_cartpole_state(_x, np), _u, _key)

        else:
            raise NotImplementedError(f"Model {model_name} is not implemented")

        # Iterate over the meshgrid and compute the prediction error and std
        _mesh_pred_density = np.zeros((len(thetadot_grid), len(theta_grid)))
        u_act = np.zeros((horizon_eval,1))
        for _i in tqdm(range(len(theta_grid))):
            for _j in range(len(thetadot_grid)):
                rng_key, density_key = jax.random.split(rng_key)
                xcurr = np.array([0.0, 0.0, theta_grid[_j,_i], thetadot_grid[_j, _i]])
                # Predicted trajectory
                xpred = pred_fn(xcurr, u_act, density_key)
                # Compute the uncertainty
                _mesh_pred_density[_j,_i] = float(np.std(xpred, axis=0).sum()) # Second output is the density
        
        # Normalize the mesh
        if density_cfg.get('normalize_mesh', False):
            _mesh_pred_density = (_mesh_pred_density - np.min(_mesh_pred_density)) / (np.max(_mesh_pred_density) - np.min(_mesh_pred_density))

        # Plot the mesh
        vmin = model_style.get('vmin', density_cfg.get('vmin', None))
        vmax = model_style.get('vmax', density_cfg.get('vmax', None))
        pcm = ax.pcolormesh(theta_grid, thetadot_grid, _mesh_pred_density, vmin=vmin, vmax=vmax, **density_cfg['mesh_args'])

        # Set the title if 'title' is in pconf
        if 'title' in model_style:
            ax.set_title(model_style['title'])
        
        # Set the x axis label
        # Add xlabel only to the bottom row
        if itr_count >= (nrows-1)*ncols:
            ax.set_xlabel(r'$\theta$')
        
        # Add ylabel only to the leftmost column
        if itr_count % ncols == 0:
            ax.set_ylabel(r'$\dot{\theta}$')

        if 'title_right' in model_style:
            # Add a twin axis on the right with no ticks and the label given by title_right
            ax2 = ax.twinx()
            ax2.set_ylabel(model_style['title_right'], **density_cfg.get('extra_args',{}).get('title_right_args', {}))
            ax2.tick_params(axis='y', which='both', length=0)
            ax2.set_yticks([])
        
        itr_count += 1

        # Now we plot the training data
        train_data = train_datas[itr_count-1]
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
    plt.show()


if __name__ == '__main__':

    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Plot utils for cartpole')

    # Add the arguments
    parser.add_argument('--fun', type=str, default='dad', help='The function to run')
    parser.add_argument('--density_cfg', type=str, default='config_plot_dad_term.yaml', help='The data generation adn training configuration file')
    parser.add_argument('--learned_dir', type=str, default='', help='The directory where all the models are stored')
    parser.add_argument('--data_dir', type=str, default='', help='The directory where trajectories data are stored')


    # Execute the parse_args() method
    args = parser.parse_args()

    if args.fun == 'dad':
        create_density_mesh_plots(args.density_cfg, args.learned_dir, args.data_dir)
    
    elif args.fun == 'pred':
        plot_prediction_accuracy(args.density_cfg)

    elif args.fun == 'unc':
        create_uncertainty_plot(args.density_cfg)