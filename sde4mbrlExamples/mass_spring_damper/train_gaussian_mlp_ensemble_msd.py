import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.optim as optim

import mbrl.models as models
import mbrl.util.common as common_utils
from mbrl.util.replay_buffer import ReplayBuffer

import pickle
import os, sys

import yaml

sys.path.append('../../mbrl_lib_utils')
from mbrl_lib_utils import save_model_and_config, \
    load_model_and_config, populate_replay_buffers, ProgressBarCallback

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

seed = 42

generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Construct and populate the replay buffers
data_config_file_name = 'MSD_MeasLow_Full_2000_config.yaml'

data_path = os.path.abspath(os.path.join(os.path.curdir, 'my_data'))
with open(os.path.abspath(os.path.join(data_path, data_config_file_name))) as f:
    data_config = yaml.safe_load(f)
with open(data_config['outfile'], 'rb') as f:
    data = pickle.load(f)

train_test_split = 0.8

train_trajectories = data[:int(len(data) * train_test_split)]
test_trajectories = data[int(len(data) * train_test_split):]

num_train_datapoints = len(train_trajectories) * data_config['horizon']
num_test_datatpoints = len(test_trajectories) * data_config['horizon']

train_buffer = populate_replay_buffers(train_trajectories, num_train_datapoints)
test_buffer = populate_replay_buffers(test_trajectories, num_test_datatpoints)

train_obs = train_buffer.obs[:train_buffer.num_stored]
test_obs = test_buffer.obs[:test_buffer.num_stored]

print('Number of training observations: {}'.format(len(train_obs)))
print('Number of testing observations {}'.format(len(test_obs)))

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1,1,1)
ax.plot(train_obs[:,0], train_obs[:,1], 'o', markersize=4, color='blue', label='Train data')
ax.plot(test_obs[:,0], test_obs[:,1], 'x', markersize=4, color='red', label='Test data')
ax.legend(fontsize=15)
plt.show()

# ReplayBuffer generates its own training/validation split, but we probably want to
# keep our own manually generated split, so instead we use two replay buffers. 

num_members = 5 # Number of models in the ensemble

# Build the ensemble of Gaussian MLPs
cfg_dict = {
    # dynamics model configuration
    "obs_shape" : (2,),
    "action_shape" : (0,),
    "trainer_setup" : {
        "optim_lr" : 0.001,
        "weight_decay" : 5e-5,
        "num_epochs" : 200,
        "patience" : 200,
        "batch_size" : 32,
    },
    "dynamics_model": {
        "model" : {
            "_target_": "mbrl.models.GaussianMLP",
            "device": device_str,
            "num_layers": 3,
            "ensemble_size": num_members,
            "hid_size": 64,
            "in_size": 2,
            "out_size": 2,
            "deterministic": False,
            # "propagation_method": "fixed_model",
            "activation_fn_cfg": {
                "_target_": "torch.nn.SiLU",
                # "negative_slope": 0.01
        }
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": False,
        "normalize": True,
    },
    "overrides": {
    }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

dynamics_model = common_utils.create_one_dim_tr_model(cfg, cfg['obs_shape'], cfg['action_shape'])# train_buffer.obs.shape[-1:], train_buffer.action.shape[-1:])
dynamics_model.update_normalizer(train_buffer.get_all()) # Normalizer gets called automatically in dynamics_model._process_batch()

# Train the model

train_dataset, _ = common_utils.get_basic_buffer_iterators(
    train_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=num_members, shuffle_each_epoch=True)
val_dataset, _ = common_utils.get_basic_buffer_iterators(
    test_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=1)

pbar = ProgressBarCallback(cfg['trainer_setup']['num_epochs'])

trainer = models.ModelTrainer(
                dynamics_model, 
                optim_lr=cfg['trainer_setup']['optim_lr'], 
                weight_decay=cfg['trainer_setup']['weight_decay']
            )

train_losses, val_losses = trainer.train(
                                train_dataset, 
                                val_dataset, 
                                num_epochs=cfg['trainer_setup']['num_epochs'], 
                                patience=cfg['trainer_setup']['patience'], 
                                callback=pbar.progress_bar_callback
                            )

# Save the learned model
experiment_name = 'gaussian_mlp_ensemble' + '_' +  data_config_file_name[:data_config_file_name.index('_config')]
save_folder = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
save_model_and_config(dynamics_model, cfg, save_folder)

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
ax[0].plot(train_losses)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train loss (gaussian nll)")
ax[1].plot(val_losses)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("val loss (mse)")
plt.show()

dynamics_model, cfg = load_model_and_config(save_folder)

init_state = np.array([0.1, 0.1])

model_state = dynamics_model.reset(
            init_state.astype(np.float32), rng=generator
        )

pred_state_list = []
pred_var_epi = []
pred_var_ale = []
std = []

pred_state_list.append(init_state)
pred_var_epi.append(np.array([0.0]))
pred_var_ale.append(np.array([0.0]))
std.append(np.array([0.0, 0.0]))

steps_in_horizon = 200

pred = torch.from_numpy(init_state).unsqueeze(0).float().to(device)
pred = dynamics_model.input_normalizer.normalize(pred)

for t in range(steps_in_horizon):
    with torch.no_grad():
        pred, pred_logvar = dynamics_model(pred)

    pred_state = pred.mean(dim=0)

    pred_state_list.append(pred_state[0].cpu().numpy())
    pred_var_epi.append(pred.var(dim=0)[0].cpu().numpy())
    pred_var_ale.append(pred_logvar.exp().mean(dim=0)[0].cpu().numpy())
    std.append(np.sqrt(pred_var_epi[-1] + pred_var_ale[-1]))

    pred = dynamics_model.input_normalizer.normalize(pred_state)

pred_state_list = np.array(pred_state_list)
pred_var_epi = np.array(pred_var_epi)
pred_var_ale = np.array(pred_var_ale)
std = np.array(std)

X = np.arange(pred_state_list.shape[0])

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(2,1,1)
plt.plot(X, pred_state_list[:, 0], 'r')
plt.fill_between(X, pred_state_list[:, 0], pred_state_list[:, 0] + 2 * std[:,0], color='b', alpha=0.2)
plt.fill_between(X, pred_state_list[:, 0] - 2 * std[:,0], pred_state_list[:, 0], color='b', alpha=0.2)
# plt.axis([-12, 12, -2.5, 2.5])

ax = fig.add_subplot(2,1,2)
plt.plot(X, pred_state_list[:, 1], 'r')
plt.fill_between(X, pred_state_list[:, 1], pred_state_list[:, 1] + 2 * std[:,1], color='b', alpha=0.2)
plt.fill_between(X, pred_state_list[:, 1] - 2 * std[:,1], pred_state_list[:, 1], color='b', alpha=0.2)

plt.show()