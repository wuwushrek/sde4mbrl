import pickle

import numpy as np
import matplotlib.pyplot as plt

import os, sys

import torch

from modified_cartpole_continuous import CartPoleEnv
from time import sleep

save_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_data'))
load_file_name = 'learned.pkl'

with open(os.path.join(save_dir, load_file_name), 'rb') as f:
    data = pickle.load(f)

max_vals_x = []
max_vals_x_dot = []
max_vals_theta_dot = []
for traj_ind in range(len(data)):
    max_vals_x.append(np.max(np.abs(data[traj_ind][0][:, 0]), axis=0))
    max_vals_x_dot.append(np.max(np.abs(data[traj_ind][0][:, 1]), axis=0))
    max_vals_theta_dot.append(np.max(np.abs(data[traj_ind][0][:, 4]), axis=0))

max_x_ind = np.argmax(max_vals_x)
max_x_dot_int = np.argmax(max_vals_x_dot)
max_theta_dot_ind = np.argmax(max_vals_theta_dot)

print('Max_x_ind: {}, max_x_dot_ind: {}, max_theta_dot_ind: {}'.format(max_x_ind, max_x_dot_int, max_theta_dot_ind))

# fig = plt.figure()
# ax = fig.add_subplot(311)
# ax.plot(max_vals_x, label='x')
# ax = fig.add_subplot(312)
# ax.plot(max_vals_x_dot, label='x_dot')
# ax = fig.add_subplot(313)
# ax.plot(max_vals_theta_dot, label='theta_dot')

traj_ind = 10

traj = data[traj_ind][0]

s = traj[:, 2]
c = traj[:, 3]

theta = np.arctan2(s, c)

fig = plt.figure()
ax = fig.add_subplot(241)
ax.plot(data[traj_ind][0][:, 0], label='x')
ax.set_ylabel(r'$x$ (m)')

ax = fig.add_subplot(242)
ax.plot(data[traj_ind][0][:, 1], label='x_dot')
ax.set_ylabel(r'$\dot{x}$ (m/s)')

ax = fig.add_subplot(243)
ax.plot(data[traj_ind][0][:, 2], label='sin(theta)')
ax.set_ylabel(r'$\sin(\theta)$')

ax = fig.add_subplot(244)
ax.plot(data[traj_ind][0][:, 3], label='cos(theta)')
ax.set_ylabel(r'$\cos(\theta)$')

ax = fig.add_subplot(245)
ax.plot(theta, label='theta')
ax.set_ylabel(r'$\theta$ (rad)')

ax = fig.add_subplot(246)
ax.plot(data[traj_ind][0][:, 4], label='theta_dot')
ax.set_ylabel(r'$\dot{\theta}$ (rad/s)')

ax = fig.add_subplot(247)
ax.plot(data[traj_ind][1], label='u')
ax.set_ylabel(r'$u$')

plt.show() 

env = CartPoleEnv(render_mode='human')

for t in range(len(traj)):
    env.state = (traj[t][0], traj[t][1], theta[t], traj[t][4])
    env.render()
    sleep(0.02)

# save_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_models'))
# model_name = 'gaussian_mlp_ensemble_cartpole'
# model_dir = os.path.join(save_dir, model_name)

# load_file = 'training_results.pkl'
# load_file_str = os.path.join(model_dir, load_file)

# with open(load_file_str, 'rb') as f:
#     training_results = pickle.load(f)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.plot(training_results['epoch'], training_results['training_loss'], label='train loss')
# ax.plot(training_results['epoch'], [torch.mean(i).cpu().numpy() for i in training_results['validation_score']], label='val loss')
# # ax.set_yscale('log')

# plt.show()