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
from mbrlLibUtils.sgd_model_trainer import SGDModelTrainer, ProgressBarCallback

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

seed = 42

generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Construct and populate the replay buffers

data_file_name = 'trajs.pkl'

data_path = os.path.abspath(os.path.join(os.path.curdir, 'iris_sitl', 'my_data'))
with open(os.path.abspath(os.path.join(data_path, data_file_name)), 'rb') as f:
    data = pickle.load(f)

dataset_train = [traj for traj in data['train']]
dataset_test = [traj for traj in data['test']]

num_train_datapoints = sum([dataset_train[i][0].shape[0] for i in range(len(dataset_train))])
num_test_datatpoints = sum([dataset_test[i][0].shape[0] for i in range(len(dataset_test))])

train_buffer = populate_replay_buffers(dataset_train, num_train_datapoints, save_actions=True)
test_buffer = populate_replay_buffers(dataset_test, num_test_datatpoints, save_actions=True)

train_obs = train_buffer.obs[:train_buffer.num_stored]
test_obs = test_buffer.obs[:test_buffer.num_stored]

print('Number of training observations: {}'.format(len(train_obs)))
print('Number of testing observations {}'.format(len(test_obs)))

from config.gaussian_mlp_ensemble_rotor_config import ensemble_cfg as cfg
dynamics_model = common_utils.create_one_dim_tr_model(cfg, cfg['obs_shape'], cfg['action_shape'])
dynamics_model.update_normalizer(train_buffer.get_all()) # Normalizer gets called automatically in dynamics_model._process_batch()

# Train the model
train_dataset, _ = common_utils.get_basic_buffer_iterators(
    train_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=cfg['dynamics_model']['model']['ensemble_size'], shuffle_each_epoch=True)
val_dataset, _ = common_utils.get_basic_buffer_iterators(
    test_buffer, cfg['trainer_setup']['batch_size'], 0, ensemble_size=1)

pbar = ProgressBarCallback(cfg['trainer_setup']['num_epochs'])

trainer = SGDModelTrainer(
                dynamics_model,
                optim_lr=cfg['trainer_setup']['optim_lr'],
                weight_decay=cfg['trainer_setup']['weight_decay']
            )

train_losses, val_losses = trainer.train(
                                train_dataset, 
                                val_dataset, 
                                num_epochs=cfg['trainer_setup']['num_epochs'], 
                                patience=cfg['trainer_setup']['patience'], 
                                callback=pbar.progress_bar_callback,
                                num_steps_per_epoch=cfg['trainer_setup']['num_steps_per_epoch'],
                            )

# Save the learned model
experiment_name = 'gaussian_mlp_ensemble_rotor_model'
save_folder = os.path.abspath(os.path.join(os.path.curdir, 'iris_sitl', 'my_models', experiment_name))
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

dynamics_model, cfg = load_model_and_config(save_folder, propagation_method="expectation")

# init_state = np.array([0.1, 0.1])

# time_horizon = 500
# num_particles = 200

# sample_trajectory = generate_sample_trajectories(init_state, num_particles, dynamics_model, generator, time_horizon=time_horizon, device=device).cpu().numpy()

# percentile_75 = np.percentile(sample_trajectory, 75, axis=0)
# percentile_25 = np.percentile(sample_trajectory, 25, axis=0)
# mean = np.mean(sample_trajectory, axis=0)

# # Plot the results
# T = np.arange(mean.shape[0])

# fig = plt.figure(figsize=(16, 8))

# ax = fig.add_subplot(2,1,1)
# plt.fill_between(T, percentile_25[:,0], percentile_75[:,0], color='green', alpha=0.2)
# plt.plot(T, mean[:,0], color='green', linewidth=3, label='Mean predicted trajectory, N_data = 2000')
# # plt.plot(T, gtruth_data[:, 0], color='black', linewidth=3, label='Ground truth trajectory')
# plt.legend(fontsize=15)

# ax = fig.add_subplot(2,1,2)
# plt.fill_between(T, percentile_25[:,1], percentile_75[:,1], color='green', alpha=0.2)
# plt.plot(T, mean[:,1], color='green', linewidth=3)
# # plt.plot(T, gtruth_data[:, 1], color='black', linewidth=3)

# plt.show()