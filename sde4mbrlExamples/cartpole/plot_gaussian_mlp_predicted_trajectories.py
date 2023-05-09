import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../..')
from mbrlLibUtils.save_and_load_models import save_model_and_config, load_model_and_config
    
from mbrlLibUtils.replay_buffer_utils import generate_sample_trajectories
import pickle

# Setup torch stuff
device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_str = 'cpu'
device = torch.device(device_str)
seed = 42

# Load the model
experiment_name = 'gaussian_mlp_ensemble_cartpole_random'
load_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
dynamics_model, cfg = load_model_and_config(load_dir, propagation_method="expectation")

# Load the dataset
data_path = os.path.abspath(os.path.join(os.path.curdir, 'my_data'))
with open(os.path.abspath(os.path.join(data_path, 'learned.pkl')), 'rb') as f:
    data = pickle.load(f)

traj_int = 0
gtruth_data = data[traj_int]
actions = gtruth_data[1][:]    

gtruth_traj = gtruth_data[0]

time_horizon = gtruth_data[0].shape[0]

# Generate the sample trajectories
init_state = gtruth_data[0][0, :]

# Generate the learned model predictions
num_particles = 50
generator = torch.Generator(device=dynamics_model.device)

generator.manual_seed(seed)
sample_trajectories = generate_sample_trajectories(
    init_state, 
    num_particles, 
    dynamics_model, 
    generator, 
    control_inputs=actions,
    time_horizon=time_horizon, 
    device=device
).cpu().numpy()

# Post-process the predictions
percentile_75 = np.percentile(sample_trajectories, 75, axis=0)
percentile_25 = np.percentile(sample_trajectories, 25, axis=0)
mean = np.mean(sample_trajectories, axis=0)

# Plot the results
T = np.arange(mean.shape[0])

plot_horizon = 100

fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot(2,3,1)
plt.fill_between(T[0:plot_horizon], percentile_25[0:plot_horizon,0], percentile_75[0:plot_horizon,0], color='blue', alpha=0.2)
plt.plot(T[0:plot_horizon], mean[0:plot_horizon,0], color='blue', linewidth=3, label='Mean predicted trajectory, N_data = 100')
plt.plot(T[0:plot_horizon], gtruth_traj[0:plot_horizon, 0], color='black', linewidth=3, label='Ground truth trajectory')
plt.legend(fontsize=15)

ax = fig.add_subplot(2,3,2)
plt.fill_between(T[0:plot_horizon], percentile_25[0:plot_horizon,1], percentile_75[0:plot_horizon,1], color='blue', alpha=0.2)
plt.plot(T[0:plot_horizon], mean[0:plot_horizon,1], color='blue', linewidth=3)
plt.plot(T[0:plot_horizon], gtruth_traj[0:plot_horizon, 1], color='black', linewidth=3)

ax = fig.add_subplot(2,3,3)
plt.fill_between(T[0:plot_horizon], percentile_25[0:plot_horizon,2], percentile_75[0:plot_horizon,2], color='blue', alpha=0.2)
plt.plot(T[0:plot_horizon], mean[0:plot_horizon,2], color='blue', linewidth=3)
plt.plot(T[0:plot_horizon], gtruth_traj[0:plot_horizon, 2], color='black', linewidth=3)

ax = fig.add_subplot(2,3,4)
plt.fill_between(T[0:plot_horizon], percentile_25[0:plot_horizon,3], percentile_75[0:plot_horizon,3], color='blue', alpha=0.2)
plt.plot(T[0:plot_horizon], mean[0:plot_horizon,3], color='blue', linewidth=3)
plt.plot(T[0:plot_horizon], gtruth_traj[0:plot_horizon, 3], color='black', linewidth=3)

ax = fig.add_subplot(2,3,5)
plt.fill_between(T[0:plot_horizon], percentile_25[0:plot_horizon,4], percentile_75[0:plot_horizon,4], color='blue', alpha=0.2)
plt.plot(T[0:plot_horizon], mean[0:plot_horizon,4], color='blue', linewidth=3)
plt.plot(T[0:plot_horizon], gtruth_traj[0:plot_horizon, 4], color='black', linewidth=3)

plt.show()