import pickle, os

import mbrl.models as models
import mbrl.util.common as common_utils

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

    config['dynamics_model']['model']['propagation_method'] = propagation_method

    model = common_utils.create_one_dim_tr_model(config, config['obs_shape'], config['action_shape'])

    model.load(save_dir)

    return model, config



