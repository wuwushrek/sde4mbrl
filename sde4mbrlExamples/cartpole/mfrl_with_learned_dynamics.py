""" Main file to train and evaluate the different RL agents on the cartpole environment
"""

import os

import torch
import numpy as np
import pickle
import random

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env

import sys
sys.path.append('../..')

from mbrlLibUtils.rl_networks import Value, Policy

from sde4mbrl.utils import update_params, load_yaml

# Import the different environments
from sde4mbrlExamples.cartpole.cartpole_sde import cartpole_sde_gym
from sde4mbrlExamples.cartpole.cartpole_gym_mlp import CartPoleGaussianMLPEnv
from sde4mbrlExamples.cartpole.modified_cartpole_continuous import CartPoleEnv

from tqdm.auto import tqdm

import yaml

def load_model_env(
        model_file,
        model_dir,
        cfg_dict,
        seed,
        eval_env=''
    ):
    """ Load the model and the environment
    """
    # Define the full file path
    full_model_path = model_dir + '/' + model_file

    # Check what kind of model to load
    if 'groundtruth' in model_file:
        experiment_name = 'groundtruth'
        env = CartPoleEnv(**cfg_dict.get('env_extra_args'+eval_env, {}))
    elif '_sde' in model_file:
        if not model_file.endswith('.pkl'):
            model_file += '.pkl'
        experiment_name = model_file.split('.pkl')[0]
        env = cartpole_sde_gym(
            filename=model_dir + '/' + model_file,
            **{'jax_seed': seed, **cfg_dict['sde_extra_args'], **cfg_dict.get('env_extra_args'+eval_env, {})}
            )
    elif 'gaussian' in model_file:
        experiment_name = model_file
        env = CartPoleGaussianMLPEnv(
            load_file_name = full_model_path,
            **{'torch_seed': seed, **cfg_dict['GE_extra_args'], **cfg_dict.get('env_extra_args'+eval_env, {})}
            )
    else:
        raise NotImplementedError('The model file is not recognized')
    
    return env, experiment_name


def train_with_ppo(
        env, # the environment to train on
        experiment_name, # name of the experiment
        seed, # seed for reproducibility
        cfg_trainer = {
            "timesteps": int(7e5),
            "headless": True,
            "disable_progressbar": False
        }, # configuration for the trainer
        extra_alg_dict={
            "rollouts": 2048,
        }, # extra arguments for the PPO algorithm
        pol_val_init_params={
            "method_name": "normal_",
            "mean": 0.0,
            "std": 0.1
        } # parameters for the initialization of the policy and value networks
    ):
    """ Train a PPO agent on the given environment
    """
    # Print the experiment name and seed
    print("\n  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print ("Experiment name \t:\t ", experiment_name)
    print ("Seed \t:\t ", seed)

    # Set the seed for reproducibility
    set_seed(seed)

    # Wrap the environment
    env = wrap_env(env)

    # Extract the device from the environment
    device = env.device
    print ("Device \t:\t ", device)

    # Set up the models
    models_ppo = {}
    models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
    models_ppo["value"] = Value(env.observation_space, env.action_space, device)
    for model in models_ppo.values():
        model.init_parameters(**pol_val_init_params)
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo['experiment']['experiment_name'] = experiment_name + "_PPO_seed_" + str(seed)
    assert 'rollouts' in extra_alg_dict.keys(), 'The number of rollouts must be specified'

    # Update the configuration with the kwargs
    cfg_ppo = update_params(cfg_ppo, extra_alg_dict)
    # Pretty print the configuration dict with keys values on each line
    print ("PPO configuration \t:")
    for k, v in cfg_ppo.items():
        print (k, " \t:\t ", v)

    # Instantiate a RandomMemory (without replacement) as experience replay memory
    memory = RandomMemory(memory_size=cfg_ppo['rollouts'], num_envs=env.num_envs, device=device, replacement=False)

    # Instantiate the PPO agent
    agent_ppo = PPO(models=models_ppo,
                    memory=memory,
                    cfg=cfg_ppo,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device
                    )
    
    # Configure and instantiate the RL trainer
    _trainer_cfg = SEQUENTIAL_TRAINER_DEFAULT_CONFIG.copy()
    _trainer_cfg = update_params(_trainer_cfg, cfg_trainer)
    trainer = SequentialTrainer(cfg=_trainer_cfg, env=env, agents=agent_ppo)
    print ("Trainer configuration \t:")
    for k, v in _trainer_cfg.items():
        print (k, " \t:\t ", v)

    # Train the agent
    trainer.train()

def train(cfg_dict):
    """
    Train the models
    """
    # Get the list of seed from the configuration file
    assert 'seeds' in cfg_dict.keys(), 'The seeds must be specified in the configuration file'
    seeds = cfg_dict['seeds']
    print ("Seeds \t:\t ", seeds)

    # Check if model_dir is empty, if so, use the default directory given by the current path
    if len(args.model_dir) == 0:
        model_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_models/'
    else:
        model_dir = args.model_dir
    
    # Now we check if several models are to be loaded
    assert len(args.model_files) > 0, 'The model files must be specified'
    print ("Model files \t:\t ", args.model_files)

    # Loop over the model files
    for model_file in args.model_files:
        # Print the current model file
        print('\n###########################################################')
        print(f"Training on {model_file}")
        # Loop over the seeds
        for seed in seeds:
            # Load the model and the environment
            env, experiment_name = load_model_env(
                model_file,
                model_dir,
                cfg_dict,
                seed,
            )
            # Train the agent
            train_with_ppo(
                env,
                experiment_name,
                seed,
                cfg_trainer = cfg_dict['cfg_trainer_ppo'],
                extra_alg_dict = cfg_dict['extra_alg_dict_ppo'],
                pol_val_init_params = cfg_dict['pol_val_init_params_ppo']
            )

def load_policy_from_checkpoint(checkpoint_name, obs_space, act_space, device='cpu'):
    """
    Load the policy from a checkpoint
    """
    # Load the checkpoint
    with open(checkpoint_name, 'rb') as f:
        state_dict = torch.load(f)
    # Create the policy
    policy = Policy(obs_space, act_space, device)
    # Load the state dict
    policy.load_state_dict(state_dict['policy'])
    # Return the policy
    return policy

def load_all_checkpoints_from_modelname(model_name, runs_dir='runs/'):
    """
    Load all the checkpoints from a given model name
    """
    # Get the list of checkpoints
    model_file = runs_dir + model_name + '/checkpoints/'
    checkpoints = os.listdir(model_file)
    list_checkpoints = []
    checkpoints_iter = []
    for checkpoint in checkpoints:
        if 'agent_' in checkpoint and checkpoint.endswith('.pt'):
            list_checkpoints.append(model_file+checkpoint)
            checkpoints_iter.append(int(checkpoint.split('_')[1].split('.')[0]))
    # Sort the list of checkpoints_iter and retur the idx of the sorted list
    idx_sorted = np.argsort(checkpoints_iter)
    # Sort the list of checkpoints
    list_checkpoints = [list_checkpoints[idx] for idx in idx_sorted]
    checkpoints_iter = [checkpoints_iter[idx] for idx in idx_sorted]
    return list_checkpoints, checkpoints_iter

def simulate_env(env, nn_policy):
    """
    Simulate the environment using the given policy
    """
    reward_list = []
    obs_list = []
    act_list = []

    obs, _ = env.reset()
    obs_list.append(obs)

    done = False
    while not done:
        with torch.no_grad():
            obs_th = torch.tensor(obs, device=nn_policy.device, dtype=torch.float32)
            act, _, _ = nn_policy.act({"states" : obs_th}, role='policy')
        obs, rew, done, _, _ = env.step(act.detach().numpy())
        act_list.append(act.detach().numpy())
        obs_list.append(obs)
        reward_list.append(rew)

    return obs_list, act_list, reward_list

def manual_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simulate_model(model_name, cfg_dict, groundtruth=False):
    """
    Simulate the environment using the given model
    """
    # Some checks
    assert model_name.endswith('_PPO') or model_name.endswith('_SAC'), 'The model name must end with _PPO or _SAC'
    # Load the environment depending on the groundtruth flag
    eval_env = '_eval'
    if groundtruth:
        env = CartPoleEnv(**cfg_dict['env_extra_args'+eval_env])
    else:
        env, _ = load_model_env(
            model_name.split('_PPO')[0] if model_name.endswith('_PPO') else model_name.split('_SAC')[0],
            args.model_dir,
            cfg_dict,
            cfg_dict['seed_eval'],
            eval_env
        )
    # Get the different seeds
    seeds = cfg_dict['seeds']
    # Model + seed list
    model_file_seed = [ model_name + '_seed_' + str(seed) for seed in seeds ]
    rewards_per_seed = []
    checkpoints_iter_per_seed = []
    tqdm.write(f"\nSimulating {model_name} ===========>")
    tqdm.write(f"Groundtruth = {groundtruth}")
    tqdm.write(f"Seeds = {seeds}")
    for model_file in tqdm(model_file_seed):
        list_checkpoints, checkpoints_iter = load_all_checkpoints_from_modelname(model_file)
        checkpoints_reward = []
        # Checkpoints spacer
        list_checkpoints = list_checkpoints[::cfg_dict['checkpoints_spacer']]
        checkpoints_iter = checkpoints_iter[::cfg_dict['checkpoints_spacer']]
        # Loop over the checkpoints
        for checkpoint_name in tqdm(list_checkpoints, leave=False):
            # Load the policy
            _curr_policy = load_policy_from_checkpoint(checkpoint_name, env.observation_space, env.action_space, device=cfg_dict.get('device_eval', 'cpu'))
            # Reset the random seed for identical environment initialization
            if hasattr(env, 'reset_model_seed'):
                env.reset_model_seed()
            manual_set_seed(cfg_dict['seed_eval'])
            # Store total reward per episode
            total_reward_list = []
            for _ in range(cfg_dict['num_eval_episodes']):
                # Simulate the environment
                _, _, reward_list = simulate_env(env, _curr_policy)
                # Store the total reward
                total_reward_list.append(np.sum(reward_list))
            checkpoints_reward.append(total_reward_list)
        # Store the rewards and the checkpoints
        rewards_per_seed.append(checkpoints_reward)
        checkpoints_iter_per_seed.append(checkpoints_iter)
    # Check if the checkpints are saved at the same iteration
    assert np.all([np.all(np.array(checkpoints_iter_per_seed[0]) == np.array(checkpoints_iter_per_seed[i])) for i in range(1, len(checkpoints_iter_per_seed))]), 'The checkpoints are not saved at the same iteration'
    return rewards_per_seed, checkpoints_iter


# Load a pkl file
def _load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def generate_plotData_from_cfg(cfg_dict):
    """
    Generate the (performance) data to plot from the configuration file
    """
    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
    # Get the list of models to evaluate
    model_names = cfg_dict['model_names']
    outfile = cfg_dict['outfile']
    # Check if file exists
    outfile_not_exist = not os.path.isfile(data_dir + outfile)
    outfile_dict = {} if cfg_dict.get('recompute_all', False) or outfile_not_exist else _load_pkl(data_dir + outfile)
    for model_name in tqdm(model_names):
        if model_name in outfile_dict.keys():
            continue
        # Simulate the model
        rewards_per_seed, checkpoints_iter = simulate_model(model_name, cfg_dict)
        # Simulate the model in the groundtruth environment
        rewards_per_seed_gt, _ = simulate_model(model_name, cfg_dict, groundtruth=True)
        # Store the results
        outfile_dict[model_name] = {
            'rewards_per_seed': rewards_per_seed,
            'checkpoints_iter': checkpoints_iter,
            'rewards_per_seed_gt': rewards_per_seed_gt
        }
        # Save the results
        with open(data_dir + outfile, 'wb') as f:
            pickle.dump(outfile_dict, f)

def get_best_agent(model_name, mydata, min_num_interaction=20000):
    """ Get the best mean reward from the reward data
    """
    rew_dict = mydata[model_name.split('_gt')[0]]
    rewValues = np.array(rew_dict['rewards_per_seed']) if not model_name.endswith('_gt') else np.array(rew_dict['rewards_per_seed_gt'])
    xValues = np.array(rew_dict['checkpoints_iter'])
    # Exchange the last two axes
    assert len(rewValues.shape) == 3, 'The shape of the rewards array is not correct'
    assert rewValues.shape[-2] == len(xValues), 'The number of checkpoints is not the same as the number of rewards'
    rewValues = rewValues.transpose((0, 2, 1))
    rewValues = rewValues.reshape((-1, rewValues.shape[-1]))
    rewValues_mean = np.mean(rewValues, axis=0)
    # Find the first index where the number of interactions is greater than min_num_interaction
    idx = [ i for i in range(len(xValues)) if xValues[i] > min_num_interaction ][0]
    # Get the best mean reward
    # print("idx, ", idx, "xValues[idx], ", xValues[idx])
    best_mean_reward_min = np.max(rewValues_mean[:idx])
    best_mean_reward_max = np.max(rewValues_mean)
    return best_mean_reward_min, best_mean_reward_max

def generate_barplot_data(cfg_dict, outbar='barplot_data.yaml'):
    """ Generate the data for the barplot in the paper and save it as a yaml file in my_data
    """
    # Load the data
    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
    outfile = cfg_dict['outfile']
    with open(data_dir + outfile, 'rb') as f:
        mydata = pickle.load(f)
    # Dictionary to store the results
    dict_results = {}
    for model_name in cfg_dict['model_names']:
        # Get the best reward from the groundtruth
        _rewValues = np.array(mydata[model_name]['rewards_per_seed'])
        rewValues = _rewValues.transpose((0, 2, 1))
        mean_rew_numEpisodes = np.mean(rewValues, axis=1)
        rewValues = rewValues.reshape((-1, rewValues.shape[-1]))
        _rewValues_gt = np.array(mydata[model_name]['rewards_per_seed_gt'])
        rewValues_gt = _rewValues_gt.transpose((0, 2, 1))
        mean_rew_numEpisodes_gt = np.mean(rewValues_gt, axis=1)
        rewValues_gt = rewValues_gt.reshape((-1, rewValues_gt.shape[-1]))
        # Get the best mean reward
        mean_reward = np.mean(rewValues, axis=0)
        std_reward = np.std(rewValues, axis=0)
        mean_reward_gt = np.mean(rewValues_gt, axis=0)
        std_reward_gt = np.std(rewValues_gt, axis=0)
        # Get the best mean reward and corresponding std
        best_mean_reward = np.max(mean_reward)
        best_std_reward = std_reward[np.argmax(mean_reward)]
        best_mean_reward_gt = np.max(mean_reward_gt)
        best_std_reward_gt = std_reward_gt[np.argmax(mean_reward_gt)]
        # Max rewards per seed
        max_rew_array = np.max(mean_rew_numEpisodes, axis=1)
        max_rew_per_seed = np.mean(max_rew_array)
        std_max_rew_per_seed = np.std(max_rew_array)

        max_rew_array_gt = np.max(mean_rew_numEpisodes_gt, axis=1)
        max_rew_per_seed_gt = np.mean(max_rew_array_gt)
        std_max_rew_per_seed_gt = np.std(max_rew_array_gt)
        # COmpute gaps between rewValues and rewValues_gt
        gaps = np.mean(np.abs(_rewValues.transpose((0, 2, 1)).mean(axis=1) - _rewValues_gt.transpose((0, 2, 1)).mean(axis=1) ) )

        # Store the results
        dict_results[model_name] = {
            'best_mean_reward': float(best_mean_reward),
            'std_best_reward': float(best_std_reward),
            'best_mean_reward_gt': float(best_mean_reward_gt),
            'std_best_reward_gt': float(best_std_reward_gt),
            '_max_rew_per_seed': float(max_rew_per_seed),
            '_std_max_rew_per_seed': float(std_max_rew_per_seed),
            '_max_rew_per_seed_gt': float(max_rew_per_seed_gt),
            '_std_max_rew_per_seed_gt': float(std_max_rew_per_seed_gt),
            '_gap': float(gaps)
        }
    # Now lets iterate and add a normalized reward and std based on the groundtruth
    scaling_factor = dict_results[cfg_dict['scaling_baseline_model']]['best_mean_reward']
    scaling_factor_max = dict_results[cfg_dict['scaling_baseline_model']]['_max_rew_per_seed']
    for model_name in cfg_dict['model_names']:
        dict_results[model_name]['best_mean_reward_scaled'] = dict_results[model_name]['best_mean_reward'] / scaling_factor
        dict_results[model_name]['std_best_reward_scaled'] = dict_results[model_name]['std_best_reward'] / scaling_factor
        dict_results[model_name]['best_mean_reward_gt_scaled'] = dict_results[model_name]['best_mean_reward_gt'] / scaling_factor
        dict_results[model_name]['std_best_reward_gt_scaled'] = dict_results[model_name]['std_best_reward_gt'] / scaling_factor
        dict_results[model_name]['_max_rew_per_seed_scaled'] = dict_results[model_name]['_max_rew_per_seed'] / scaling_factor_max
        dict_results[model_name]['_std_max_rew_per_seed_scaled'] = dict_results[model_name]['_std_max_rew_per_seed'] / scaling_factor_max
        dict_results[model_name]['_max_rew_per_seed_gt_scaled'] = dict_results[model_name]['_max_rew_per_seed_gt'] / scaling_factor_max
        dict_results[model_name]['_std_max_rew_per_seed_gt_scaled'] = dict_results[model_name]['_std_max_rew_per_seed_gt'] / scaling_factor_max

    # Save the results as a yaml file
    with open(data_dir + outbar, 'w') as f:
        yaml.dump(dict_results, f)


def plot_data(cfg_dict):
    """
    Plot the data
    """
    import matplotlib.pyplot as plt

    # Load the plt style
    texConfig = cfg_dict.get('texConfig', None)
    if texConfig is not None:
        plt.style.use(texConfig)

    data_dir = os.path.dirname(os.path.realpath(__file__)) + '/my_data/'
    outfile = cfg_dict['plot_data_file']
    # Load the data
    with open(data_dir + outfile, 'rb') as f:
        mydata = pickle.load(f)

    plot_configs = cfg_dict['plot_configs']

    # Exract the figure and axis specifications
    fig_specs = cfg_dict['fig_args']
    nrows = fig_specs['nrows']
    ncols = fig_specs['ncols']

    # Check if the number of subplots is the same as the number of files to plot
    assert nrows*ncols == len(plot_configs), "The number of subplots is not the same as the number of files to plot"

    # Percentiles
    alpha_percentiles = cfg_dict['alpha_percentiles']
    percentiles_array = cfg_dict['percentiles_array']

    # Curve plot style
    curve_plot_style = cfg_dict['curve_plot_style']
    general_style = cfg_dict.get('general_style', {})

    # Create the figure
    fig, axs_2d = plt.subplots(**fig_specs)
    if hasattr(axs_2d, 'shape'):
        axs = axs_2d.flatten()
    else:
        axs = [axs_2d]
    
    # Get the best reward from the groundtruth
    dict_best_gt = {}
    for best_rew_gt in cfg_dict.get('best_rew_gt', []):
        gt_reward_equivalent_data, gt_reward_max = get_best_agent(best_rew_gt, mydata, 
                    min_num_interaction=cfg_dict['total_env_interact_for_training_data'])
        dict_best_gt[best_rew_gt] = (gt_reward_equivalent_data, gt_reward_max)

    groundlabel_added = False

    for conf, ax in zip(plot_configs, axs):

        if not isinstance(conf['value'], list):
            conf['value'] = [conf['value']]

        for model_name in conf['value']:
            plot_type = conf.get('type', None)

            rew_dict = mydata[model_name.split('_gt')[0]]
            rewValues = np.array(rew_dict['rewards_per_seed']) if not model_name.endswith('_gt') else np.array(rew_dict['rewards_per_seed_gt'])
            xValues = np.array(rew_dict['checkpoints_iter'])
            
            # Exchange the last two axes
            assert len(rewValues.shape) == 3, 'The shape of the rewards array is not correct'
            assert rewValues.shape[-2] == len(xValues), 'The number of checkpoints is not the same as the number of rewards'
            rewValues = rewValues.transpose((0, 2, 1))
            rewValues = rewValues.reshape((-1, rewValues.shape[-1]))

            if plot_type == 'gap':
                rewValues_gt = np.array(rew_dict['rewards_per_seed_gt'])
                rewValues_gt = rewValues_gt.transpose((0, 2, 1))
                _rewValues_gt = np.mean(rewValues_gt, axis=1)
                # rewValues_gt = rewValues_gt.reshape((-1, rewValues_gt.shape[-1]))
                _rewValues = np.mean(rewValues.reshape((-1, rewValues_gt.shape[1], rewValues_gt.shape[2])), axis=1)
                rewValues = np.abs(_rewValues_gt - _rewValues)
                # SSmooth the data
                # rewValues = np.array([np.convolve(rewValues[i,:], np.ones(3)/3, mode='same') for i in range(rewValues.shape[0])])

            # We do the statistics
            meanRew = np.mean(rewValues, axis=0)
            cfg_dict['std_style'] = cfg_dict.get('std_style', None)
            if cfg_dict['std_style'] == 'perc':
                for _alph, _perc in zip(alpha_percentiles, percentiles_array):
                    idx = int( (1 - _perc) / 2.0 * rewValues.shape[0] )
                    q_bot = rewValues[idx,:]
                    q_top = rewValues[-idx,:]
                    ax.fill_between(xValues, q_bot, q_top, alpha=_alph, linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
            elif cfg_dict['std_style'] == 'std':
                stdRew = np.std(rewValues, axis=0)
                ax.fill_between(xValues, meanRew - stdRew, meanRew + stdRew, alpha=cfg_dict['alpha_std'], linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
            elif cfg_dict['std_style'] == 'minmax':
                minRew = np.min(rewValues, axis=0)
                maxRew = np.max(rewValues, axis=0)
                ax.fill_between(xValues, minRew, maxRew, alpha=cfg_dict['alpha_std'], linewidth=0.0, color=curve_plot_style[model_name]['color_std'])
            # Plot the mean
            mean_style = curve_plot_style[model_name].copy()
            mean_style.pop('color_std', None)
            ax.plot(xValues, meanRew, **{**general_style, **mean_style})

        # Set the x-axis and format it
        if conf.get('xaxis', True):
            ax.set_xlabel(r'Cartpole steps ($\times 10^5$)')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x/100000)))
        
        # Add the groundtruth reward
        for best_rew_gt in dict_best_gt.keys():
            if plot_type == 'gap':
                continue
            gt_reward_equivalent_data, gt_reward_max = dict_best_gt[best_rew_gt]
            ax.axhline(gt_reward_equivalent_data, **{**general_style, **curve_plot_style[best_rew_gt+'_data'], 
                                                     'label': curve_plot_style[best_rew_gt+'_data']['label'] if not groundlabel_added else None
                                                    }
                        )
            # ax.axhline(gt_reward_max, **{**general_style, **curve_plot_style[best_rew_gt+'_max'], 
            #                                          'label': curve_plot_style[best_rew_gt+'_max']['label'] if not groundlabel_added else None
            #                                         }
            #             )
            groundlabel_added = True

        # Set the y-axis
        if conf.get('yaxis', True):
            if plot_type == 'gap':
                ax.set_ylabel(r'Behavior gap')
            else:
                ax.set_ylabel(r'Cartpole total reward ($\times 10^2$)')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x/100)))

        # if cfg_dict.get('global_legend', None) == False:
        #     ax.legend()
        ax.grid(True)
    
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
    fig.legend(lines, labels, **cfg_dict.get('extra_args',{}).get('legend_args', {}))
        

    # # Collect all the labels and show them in the legend
    # if  cfg_dict.get('global_legend', None) == True:
    #     fig.legend(**cfg_dict.get('extra_args',{}).get('legend_args', {}))

    # Save the figure
    if 'save_config' in cfg_dict.keys():
        figure_out = data_dir + 'figures/'
        cfg_dict['save_config']['fname'] = figure_out + cfg_dict['save_config']['fname']
        fig.savefig(**cfg_dict['save_config'])
    
    if 'save_config_tex' in cfg_dict.keys():
        for ax in axs:
            ax.legend()
        tikzplotlib_fix_ncols(fig)
        import tikzplotlib
        cfg_dict['save_config_tex']['fname'] = figure_out + cfg_dict['save_config_tex']['fname']
        tikzplotlib.clean_figure(fig)
        tikzplotlib.save(cfg_dict['save_config_tex']['fname'], figure=fig)

    # Show the figure
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

if __name__ == '__main__':
    import argparse
    # Argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate an RL agent on a given environment')
    parser.add_argument('--fun', type=str, default='train', help='The function to execute: train, evalpol, plot, barplot')
    parser.add_argument('--cfg_path', type=str, default='rl_config.yaml', help='The path to the configuration file')
    parser.add_argument('--model_dir', type=str, default='my_models/', help='The directory where the models are stored')
    # Create model_file argument which can be a string or a list of strings
    parser.add_argument('--model_files', type=str,  nargs='+', default=['groundtruth',], help='The name of the models or model files')
    parser.add_argument('--method', type=str, default='ppo', help='The RL method to use')
    parser.add_argument('--outbar', type=str, default='barplot_data.yaml', help='The name of the output file for the barplot')

    # Parse the arguments
    args = parser.parse_args()

    # Load the configuration file
    cfg_dict = load_yaml(args.cfg_path)

    if args.fun == 'train':
        train(cfg_dict)
    elif args.fun == 'evalpol':
        generate_plotData_from_cfg(cfg_dict)
    elif args.fun == 'plot':
        plot_data(cfg_dict)
    elif args.fun == 'barplot':
        generate_barplot_data(cfg_dict, args.outbar)
