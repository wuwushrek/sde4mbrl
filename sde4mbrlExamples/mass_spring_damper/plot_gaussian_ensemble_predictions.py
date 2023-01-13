from mbrl_lib_utils import save_model_and_config, load_model_and_config, generate_sample_trajectories

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Setup torch stuff
device_str = 'cpu'
device = torch.device(device_str)
seed = 42
generator = torch.Generator(device=device)
generator.manual_seed(seed)

# Load the model
experiment_name = 'gaussian_mlp_ensemble_MSD_MeasLow_Full_100'
load_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))

dynamics_model, cfg = load_model_and_config(load_dir, propagation_method="expectation")

# Generate the sample trajectories
init_state = np.array([0.1, 0.1])

num_particles = 20

sample_trajectories = generate_sample_trajectories(init_state, num_particles, dynamics_model, generator, time_horizon=100, device=device)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(2,1,1)
X = np.arange(sample_trajectories.shape[1])
for i in range(num_particles):
    plt.plot(X, sample_trajectories[i, :, 0], 'r')

ax = fig.add_subplot(2,1,2)
for i in range(num_particles):
    plt.plot(X, sample_trajectories[i, :, 1], 'r')
plt.show()