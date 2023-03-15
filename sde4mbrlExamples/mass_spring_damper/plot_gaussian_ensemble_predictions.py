import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../../mbrl_lib_utils')
from mbrlLibUtils.save_and_load_models import save_model_and_config, load_model_and_config
    
from mbrlLibUtils.replay_buffer_utils import generate_sample_trajectories

from mass_spring_model import load_data_generator

import jax

# Setup torch stuff
device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_str = 'cpu'
device = torch.device(device_str)
seed = 42


seed_rng = jax.random.PRNGKey(seed)

# Load the model
experiment_name_100 = 'gaussian_mlp_ensemble_MSD_MeasLow_Full_100'
experiment_name_700 = 'gaussian_mlp_ensemble_MSD_MeasLow_Full_700'
experiment_name_2000 = 'gaussian_mlp_ensemble_MSD_MeasLow_Full_2000'

load_dir_100 = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name_100))
load_dir_700 = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name_700))
load_dir_2000 = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name_2000))

dynamics_model_100, cfg_100 = load_model_and_config(load_dir_100, propagation_method="expectation")
dynamics_model_700, cfg_700 = load_model_and_config(load_dir_700, propagation_method="expectation")
dynamics_model_2000, cfg_2000 = load_model_and_config(load_dir_2000, propagation_method="expectation")

time_horizon = 500

# Generate the sample trajectories
init_state = np.array([0.1, 0.1])

# Generate the true trajectory
model_groundtruth_dir = "mass_spring_damper.yaml"
# Generate a trajectory starting from xinit and length HORIZON
groundtruth_sampler, _ = load_data_generator(model_groundtruth_dir, noise_info={}, horizon=time_horizon, ufun=None)
gtruth_data, _ = groundtruth_sampler(init_state, seed_rng) # Second output is the control input
gtruth_data = np.array(gtruth_data)[:-1, :]

# Generate the learned model predictions
num_particles = 200
generator = torch.Generator(device=dynamics_model_100.device)

generator.manual_seed(seed)
sample_trajectories_100 = generate_sample_trajectories(init_state, num_particles, dynamics_model_100, generator, time_horizon=time_horizon, device=device).cpu().numpy()

generator = torch.Generator(device=dynamics_model_700.device)
generator.manual_seed(seed)
sample_trajectories_700 = generate_sample_trajectories(init_state, num_particles, dynamics_model_700, generator, time_horizon=time_horizon, device=device).cpu().numpy()

generator = torch.Generator(device=dynamics_model_2000.device)
generator.manual_seed(seed)
sample_trajectories_2000 = generate_sample_trajectories(init_state, num_particles, dynamics_model_2000, generator, time_horizon=time_horizon, device=device).cpu().numpy()

# Post-process the predictions
percentile_75_100 = np.percentile(sample_trajectories_100, 75, axis=0)
percentile_25_100 = np.percentile(sample_trajectories_100, 25, axis=0)
mean_100 = np.mean(sample_trajectories_100, axis=0)

percentile_75_700 = np.percentile(sample_trajectories_700, 75, axis=0)
percentile_25_700 = np.percentile(sample_trajectories_700, 25, axis=0)
mean_700 = np.mean(sample_trajectories_700, axis=0)

percentile_75_2000 = np.percentile(sample_trajectories_2000, 75, axis=0)
percentile_25_2000 = np.percentile(sample_trajectories_2000, 25, axis=0)
mean_2000 = np.mean(sample_trajectories_2000, axis=0)

# Plot the results
T = np.arange(mean_100.shape[0])

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(2,1,1)
plt.fill_between(T, percentile_25_100[:,0], percentile_75_100[:,0], color='blue', alpha=0.2)
plt.plot(T, mean_100[:,0], color='blue', linewidth=3, label='Mean predicted trajectory, N_data = 100')
plt.fill_between(T, percentile_25_700[:,0], percentile_75_700[:,0], color='orange', alpha=0.2)
plt.plot(T, mean_700[:,0], color='orange', linewidth=3, label='Mean predicted trajectory, N_data = 700')
plt.fill_between(T, percentile_25_2000[:,0], percentile_75_2000[:,0], color='green', alpha=0.2)
plt.plot(T, mean_2000[:,0], color='green', linewidth=3, label='Mean predicted trajectory, N_data = 2000')
# for i in range(num_particles):
#     plt.plot(X, sample_trajectories[i, :, 0], 'r')
plt.plot(T, gtruth_data[:, 0], color='black', linewidth=3, label='Ground truth trajectory')
plt.legend(fontsize=15)

ax = fig.add_subplot(2,1,2)
plt.fill_between(T, percentile_25_100[:,1], percentile_75_100[:,1], color='blue', alpha=0.2)
plt.plot(T, mean_100[:,1], color='blue', linewidth=3)
plt.fill_between(T, percentile_25_700[:,1], percentile_75_700[:,1], color='orange', alpha=0.2)
plt.plot(T, mean_700[:,1], color='orange', linewidth=3)
plt.fill_between(T, percentile_25_2000[:,1], percentile_75_2000[:,1], color='green', alpha=0.2)
plt.plot(T, mean_2000[:,1], color='green', linewidth=3)
# for i in range(num_particles):
#     plt.plot(X, sample_trajectories[i, :, 1], 'r')
plt.plot(T, gtruth_data[:, 1], color='black', linewidth=3)

plt.show()