#!/usr/bin/env python

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np

from sde4mbrlExamples.rotor_uav.sde_rotor_model import load_predictor_function, load_trajectory

# get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt

# Path for the learned sde parameters
ode_path = 'my_models/iris_sitl_ode_sde.pkl'
sde_path = 'my_models/iris_sitl_sde.pkl'
modified_params = {'horizon' : 50, 'num_particles' : 100, 'stepsize': 0.05}

# Load a trajectory to compare the models on
path_traj = 'my_data/iris_gazebo_traj1.ulg'
_traj_x, _traj_u = load_trajectory(path_traj, outlier_cond=lambda d: d['z'] > 0.3) # Only the first trajectory

# Extract the data
traj_data = {'y' : _traj_x[0], 'u' : _traj_u[0]}

# The time step in the data
data_stepsize = 0.01

# The time evolution of the trajectory from the time step
traj_time_evol = np.array([i*data_stepsize for i in range(traj_data['y'].shape[0])])

# Load the system identification-based model 
_nominal_model, nom_times = load_predictor_function(sde_path, prior_dist=True, nonoise=True, modified_params= {**modified_params, 'num_particles' : 1}, return_time_steps=True)
sysId_model = jax.jit(_nominal_model)

# Load the neural SDE based model
_sde_model, sde_times = load_predictor_function(sde_path, prior_dist=False, modified_params= modified_params, return_time_steps=True)
sde_model = jax.jit(_sde_model)

# Load the ODE-based model
_ode_model, ode_times = load_predictor_function(ode_path, prior_dist=False, nonoise=True, modified_params= {**modified_params, 'num_particles' : 1}, return_time_steps=True)
ode_model = jax.jit(_ode_model)


# Main function to split the test trajectories into chunk and evaluate the learned models on each chunks
def n_steps_analysis(xtraj, utraj, jit_sampling_fn, time_evol):
    """Compute the time evolution of the mean and variance of the SDE at each time step

    Args:
        xtraj (TYPE): The trajectory of the states
        utraj (TYPE): The trajectory of the inputs
        jit_sampling_fn (TYPE): The sampling function return an array of size (num_particles, horizon, state_dim)
        time_evol (TYPE): The time evolution of the sampling technique

    Returns:
        TYPE: The multi-sampled state evolution
        TYPE: The time step evolution for plotting
    """
    sampler_horizon = len(time_evol) - 1
    dt_sampler = time_evol[1] - time_evol[0]
    # Check if dt_sampler and data_stepsize are close enough
    if abs(dt_sampler - data_stepsize) < 1e-5:
        quot = 1
    else:
        assert dt_sampler > data_stepsize-1e-5, "The time step of the sampling function must be larger than the data step size"
        assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
        quot = dt_sampler / data_stepsize

    # print(dt_sampler, data_stepsize, dt_sampler % sampler_horizon, sampler_horizon % dt_sampler)
    # assert dt_sampler > data_stepsize-1e-6, "The time step of the sampling function must be larger than the data step size"
    # assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
    quot = dt_sampler / data_stepsize
    # Take the closest integer to quot
    num_steps2data  = int(quot + 0.5)
    # Compute the actual horizon for splitting the trajectories
    traj_horizon = num_steps2data * sampler_horizon
    # Split the trajectory into chunks of size num_steps2data
    total_traj_size = (xtraj.shape[0] // traj_horizon) * traj_horizon
    xevol = xtraj[:total_traj_size+1]
    uevol = utraj[:total_traj_size]
    uevol = uevol.reshape(-1, sampler_horizon, num_steps2data, uevol.shape[-1])
    xevol = xevol[::traj_horizon]
    # Reshape the time evolution
    m_tevol = traj_time_evol[:total_traj_size+1][::traj_horizon]
    print(xevol.shape)
    print(uevol.shape)
    # assert xevol.shape[0] == uevol.shape[0], "The number of trajectories must be the same for the states and inputs"
    # Initial random number generator
    rng = jax.random.PRNGKey(0)
    rng, s_rng = jax.random.split(rng)
    xres = []
    tres = []
    for i in range(uevol.shape[0]):
        rng, s_rng = jax.random.split(rng)
        # _curr_u = np.mean(uevol[i], axis=-2)
        _curr_u = uevol[i,:,0,:]
        _curr_x = xevol[i]
        _xpred = np.array(jit_sampling_fn(_curr_x, _curr_u, s_rng)) # (num_particles, horizon+1, state_dim)
        _tevol = m_tevol[i] + time_evol
        if i < xevol.shape[0]-1:
            _xpred = _xpred[:,:-1,:]
            _tevol = _tevol[:-1]
        xres.append(_xpred)
        tres.append(_tevol)
    # Merge the results along the horizon axis
    xres = np.concatenate(xres, axis=1)
    tres = np.concatenate(tres, axis=0)
    print(xres.shape, tres.shape)
    return xres, tres


# Compute trajectory by the nominal model
xevol_sysid, tsysid = n_steps_analysis(traj_data['y'], traj_data['u'], sysId_model, nom_times)
# xevol_nominal, tnominal = n_steps_analysis(traj_data['y'], traj_data['u'], nominal_model, nom_times)

# Compute trajectory by the ODE model
xevol_ode, tode = n_steps_analysis(traj_data['y'], traj_data['u'], ode_model, ode_times)

# Compute trajectory by the SDE model
xevol_sde, tsde = n_steps_analysis(traj_data['y'], traj_data['u'], sde_model, sde_times)

# # COmpute trajectory by the nominal model
# xevol_posterior, tposterior = n_steps_analysis(traj_data['y'], traj_data['u'], posterior_model, post_times)


# The states in the x array
name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
fig, axs = plt.subplots(5,3, figsize=(14,16), sharex=True)
axs = axs.flatten()
for i in range(len(name_states)):
    # Let's plot the groundtruth
    axs[i].plot(traj_time_evol, traj_data['y'][:,i], color='k', linestyle='--', label='Measured', zorder=100)

    # Let's plot the ODE learned model
    axs[i].plot(tode, xevol_ode[0,:,i], color='b', label='Neural ODE' if i==0 else None, zorder=70)

    # Let's plot the SDE learned model
    for k in range(xevol_sde.shape[0]):
        axs[i].plot(tsde, xevol_sde[k,:,i], color='r', label='Neural SDE' if k==0 else None, zorder=10)
    
    # Let's plot the system identification model
    axs[i].plot(tsysid, xevol_sysid[0,:,i], color='g', label='SysID' if i==0 else None, zorder=40)

    axs[i].set_ylabel(name_states[i])
    axs[i].set_xlabel('Time [s]')
    # axs[i].set_title(name_states[i])
    # axs[i].legend()
    axs[i].grid()

# Set a single legend for all subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5)

plt.show()

