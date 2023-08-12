import pickle, os

import mbrl.models as models
import mbrl.util.common as common_utils

from mbrlLibUtils.replay_buffer_utils import generate_sample_trajectories

import torch
import numpy as np
from tqdm import tqdm

def save_model_and_config(model, config, save_dir):
    """
    Save the model and config to the save_dir.

    Parameters
    ----------
    model : mbrl.models.OneDTransitionRewardModel
        Model to be saved.
    config : omegaconf.dictconfig.DictConfig
        Config to be saved.
    save_dir : str
        Directory to save the model and config.
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    config_file_str = os.path.join(save_dir, 'config.pkl')

    with open(config_file_str, 'wb') as f:
        pickle.dump(config, f)
    
    model.save(save_dir)

def load_model_and_config(save_dir, propagation_method=None):
    """
    Load the model and config from the save_dir.

    Parameters
    ----------
    save_dir : str
        Directory to load the model and config from.
    propagation_method : str
        Propagation method to use for the model. If None, the model will return the output
        of each model in the ensemble. Other valid options (for the gaussian MLP ensemble model) are:
            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    Returns
    -------
    model : mbrl.models.OneDTransitionRewardModel
        Loaded model.
    config : omegaconf.dictconfig.DictConfig
        Loaded config.
    """
    config_file_str = os.path.join(save_dir, 'config.pkl')

    with open(config_file_str, 'rb') as f:
        config = pickle.load(f)

    config['dynamics_model']['propagation_method'] = propagation_method

    model = common_utils.create_one_dim_tr_model(config, config['obs_shape'], config['action_shape'])

    model.load(save_dir)

    return model, config

def load_learned_ensemble_model(
    model_path, 
    horizon=1, 
    num_samples=1, 
    ufun=None, 
    propagation_method="fixed_model", 
    rseed=1000, 
    prior_dist=False,
    device=None
):
    """ Load the learned model from the path

        Args:
            model_path (str): The path to the learned model
            horizon (int, optional): The horizon of the model
            num_samples (int, optional): The number of samples to generate
            ufun (function, optional): The control function
            propagation_method (str, optional): The propagation method to used to generate trajectories.
            Valid propagation options are:
            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.
            rseed (int, optional): The random seed for the sampling (default: 1000)
            prior_dist (bool, optional): If True, the function will sample from the prior knowledge of the system + prior diffusion

        Returns:
            sampling (function): A function to sample from the learned model
                sampling(y, rng) -> yevol
            
            _time_evol (np.array): The time evolution of the model
        
    """
    # Load the model
    model, _ = load_model_and_config(model_path, propagation_method=propagation_method)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    generator = torch.Generator(device=torch.device('cuda'))
    generator.manual_seed(rseed)

    def sampling(y, u, rng):
        return generate_sample_trajectories(y, num_samples, model, generator, time_horizon=horizon, ufun=ufun, control_inputs=u, device=device).cpu().numpy()

    _time_evol = np.arange(0, horizon + 1, 0.01)

    return sampling, _time_evol
