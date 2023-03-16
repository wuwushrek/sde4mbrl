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

sys.path.append('../../')
from mbrlLibUtils.save_and_load_models import save_model_and_config, \
    load_model_and_config
from mbrlLibUtils.replay_buffer_utils import populate_replay_buffers, generate_sample_trajectories
from mbrlLibUtils.sgd_model_trainer import SGDModelTrainer, TrainCallback

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

seed = 42

generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Construct and populate the replay buffers
data_config_file_name = 'MSD_MeasLow_Full_700_config.yaml'
test_data_config_file_name = 'MSD_TestData_config.yaml'

data_path = os.path.abspath(os.path.join(os.path.curdir, 'my_data'))
with open(os.path.abspath(os.path.join(data_path, data_config_file_name))) as f:
    data_config = yaml.safe_load(f)
with open(data_config['outfile'], 'rb') as f:
    data = pickle.load(f)

with open(os.path.abspath(os.path.join(data_path, test_data_config_file_name))) as f:
    test_data_config = yaml.safe_load(f)
with open(test_data_config['outfile'], 'rb') as f:
    test_data = pickle.load(f)

# train_test_split = 0.8

# train_trajectories = data[:int(len(data) * train_test_split)]
# test_trajectories = data[int(len(data) * train_test_split):]

num_train_datapoints = len(data) * data_config['horizon']
num_test_datatpoints = len(test_data) * test_data_config['horizon']

train_buffer = populate_replay_buffers(data, num_train_datapoints)
test_buffer = populate_replay_buffers(test_data, num_test_datatpoints)

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

from config.gaussian_mlp_ensemble_msd_config import ensemble_cfg as cfg
dynamics_model = common_utils.create_one_dim_tr_model(cfg, cfg['obs_shape'], cfg['action_shape'])
dynamics_model.update_normalizer(train_buffer.get_all()) # Normalizer gets called automatically in dynamics_model._process_batch()

# Train the model
train_dataset, _ = common_utils.get_basic_buffer_iterators(
    train_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=cfg['dynamics_model']['model']['ensemble_size'], shuffle_each_epoch=True)
val_dataset, _ = common_utils.get_basic_buffer_iterators(
    test_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=1)

trainer = SGDModelTrainer(
                dynamics_model,
                optim_lr=cfg['trainer_setup']['optim_lr'],
                weight_decay=cfg['trainer_setup']['weight_decay']
            )

train_callback = TrainCallback(
    num_training_epochs=cfg['trainer_setup']['num_epochs'],
    model_checkpoint_frequency=cfg['trainer_setup']['model_checkpoint_frequency'],
)

train_losses, val_losses = trainer.train(
                                train_dataset, 
                                val_dataset, 
                                num_epochs=cfg['trainer_setup']['num_epochs'], 
                                patience=cfg['trainer_setup']['patience'], 
                                # callback=pbar.progress_bar_callback,
                                callback=train_callback.train_callback,
                                num_steps_per_epoch=cfg['trainer_setup']['num_steps_per_epoch'],
                            )

# Save the learned model
experiment_name = 'gaussian_mlp_ensemble' + '_' +  data_config_file_name[:data_config_file_name.index('_config')]
save_folder = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
save_model_and_config(dynamics_model, cfg, save_folder)
train_callback.save_training_results(save_folder)
train_callback.save_model_checkpoints(save_folder)

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
ax[0].plot(train_losses)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train loss (gaussian nll)")
ax[1].plot(val_losses)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("val loss (mse)")
plt.show()

dynamics_model, cfg = load_model_and_config(save_folder, propagation_method="expectation")

init_state = np.array([0.1, 0.1])

# model_state = dynamics_model.reset(
#             init_state.astype(np.float32), rng=generator
#         )

time_horizon = 500
num_particles = 200

sample_trajectory = generate_sample_trajectories(init_state, num_particles, dynamics_model, generator, time_horizon=time_horizon, device=device).cpu().numpy()

percentile_75 = np.percentile(sample_trajectory, 75, axis=0)
percentile_25 = np.percentile(sample_trajectory, 25, axis=0)
mean = np.mean(sample_trajectory, axis=0)

# Plot the results
T = np.arange(mean.shape[0])

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(2,1,1)
plt.fill_between(T, percentile_25[:,0], percentile_75[:,0], color='green', alpha=0.2)
plt.plot(T, mean[:,0], color='green', linewidth=3, label='Mean predicted trajectory, N_data = 2000')
# plt.plot(T, gtruth_data[:, 0], color='black', linewidth=3, label='Ground truth trajectory')
plt.legend(fontsize=15)

ax = fig.add_subplot(2,1,2)
plt.fill_between(T, percentile_25[:,1], percentile_75[:,1], color='green', alpha=0.2)
plt.plot(T, mean[:,1], color='green', linewidth=3)
# plt.plot(T, gtruth_data[:, 1], color='black', linewidth=3)

plt.show()