import os
# MPC seems to be faster on cpu because of the loop
# TODO: check if this is still true, and investiage how to make it faster on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt

import time
from tqdm.auto import tqdm

from sde4mbrlExamples.rotor_uav.sde_mpc_design import load_mpc_from_cfgfile

from sde4mbrlExamples.rotor_uav.utils import quat_from_euler, quat_to_euler

cfg_file = 'mpc_test_cfg.yaml'
cfg_dict, (m_reset, m_mpc), state_from_traj, _one_step_function = load_mpc_from_cfgfile(cfg_file, convert_to_enu=False)

m_reset_jit = jax.jit(m_reset)
# opt_state = m_reset_jit()
# print(opt_state)

# Define the initial state
t_init = 0.0
if state_from_traj is not None:
    x_init = state_from_traj(t_init)
    print(x_init)
    print('-------')
# x_init = state_from_traj(t_init) if state_from_traj is not None else cfg_dict['sim_scen']['pos_init']

# Run the mpc
rng = jax.random.PRNGKey(0)
rng, rng_next = jax.random.split(rng)
m_mpc_jit = jax.jit(m_mpc)

stepsize = 0.01 # Integration step size

def extract_policy(opt_state):
    return {'avg_linesearch': opt_state.avg_linesearch, 'delta' : opt_state.x_opt[0,0], 'eng' : opt_state.x_opt[0,1],
            'num_steps' : opt_state.num_steps, 'grad_norm': opt_state.grad_sqr,
            'avg_stepsize': opt_state.avg_stepsize, 'cost0': opt_state.init_cost, 'costT': opt_state.opt_cost }

def reset_env(rng):
    """ Reset the environment """
    # global x_init

    rng_next, rng = jax.random.split(rng)

    if state_from_traj is not None:
        # Grab the initial state
        _x, _y, _z, _vx, _vy, _vz, _, _, _, _, _wx, _wy, _wz = x_init
        # Get the current pitch, roll, yaw
        roll_, pitch_, yaw_ = quat_to_euler(x_init[6:10], jnp)
        pos_euler = jnp.array([_x, _y, _z, _vx, _vy, _vz, roll_, pitch_, yaw_, _wx, _wy, _wz])
    else:
        pos_euler = jnp.array(cfg_dict['sim_scen']['pos_init'])

    pos_euler_pert = pos_euler + jax.random.normal(rng, pos_euler.shape) * jnp.array(cfg_dict['sim_scen']['init_std'])

    # Convert back to xyz ...
    x_, y_, z_, vx_, vy_, vz_, roll_, pitch_, yaw_, wx_, wy_, wz_ = pos_euler_pert

    # Create quaternion from the angles
    quat_init = quat_from_euler(roll_, pitch_, yaw_, jnp)

    # Normalize the quaternion
    quat_init = quat_init / jnp.sum(jnp.square(quat_init))
    _x_init = jnp.array([x_, y_, jnp.abs(z_), vx_, vy_, vz_, quat_init[0], quat_init[1], quat_init[2], quat_init[3], wx_, wy_, wz_])

    # Target state
    xtarget = None
    if state_from_traj is None: # This happen when a trajectory is not given but it is a step
        # Get the initial state and the target state
        # x_, y_, z_ = x_init[:3]
        xtarget_, ytarget_, ztarget_, yaw_target_ = jnp.array(cfg_dict['sim_scen']['pos_goal']) + jax.random.normal(rng_next, (4,)) * jnp.array(cfg_dict['sim_scen']['pos_goal_std'])

        # Get quaternion from yaw
        quat_target = quat_from_euler(0., 0., yaw_target_, jnp)

        # Normalize the quaternion
        quat_target = quat_target / jnp.sum(jnp.square(quat_target))
        xtarget = jnp.array([xtarget_, ytarget_, jnp.abs(ztarget_), 0., 0., 0., quat_target[0], quat_target[1], quat_target[2], quat_target[3], 0., 0., 0.])

    return _x_init, xtarget

def init_env(rng=None):
    return reset_env(rng)

def step_env(xt, ut, rng):
    return jax.jit(_one_step_function)(xt, ut, rng) # First sample of next state

def play_policy(num_iteration, rng):
    # First split
    rng, init_rng, _reset_rng = jax.random.split(rng, 3)

    # Initialize the state in the environment
    state0, _target = init_env(init_rng)

    # Extract the initial state of the solver
    opt_state = m_reset_jit(x=state0, rng=_reset_rng)
    feats = extract_policy(opt_state)
    _, _opt_state, _, _ = m_mpc_jit(state0, rng, opt_state, curr_t=jnp.array(0.), xdes=_target)
    m_mpc_jit(state0, rng, _opt_state, curr_t=jnp.array(0.), xdes=_target)
    tqdm.write(' | '.join([ '{} : {:.3f}'.format(k,v) for k, v in feats.items() if k != 'u']))
    cost_total = 0

    # Store the solution of the problem
    sol = {'state': [state0], 't': [jnp.array(0.)], 'u' : []}

    # Iterate through the number steps in the environment
    for _ in tqdm(range(num_iteration)):
        rng, c_rng = jax.random.split(rng)

        # Measure the time spent in solving the problem
        curr_time = time.time()
        _uopt, opt_state, _, vehicle_states = m_mpc_jit(state0, c_rng, opt_state, curr_t=sol['t'][-1], xdes=_target)
        _uopt = _uopt[0]
        # Block until ready
        opt_state.yk.block_until_ready()
        elapsed_time = time.time() - curr_time

        # costv = jax.jit(cost_fn)(state0, _uopt) * 0.01
        cost_total += float(vehicle_states[0, 1, -1])

        # Extract the interesting features from the optimizer state
        feats = extract_policy(opt_state)
        feats['time'] = elapsed_time
        feats['Tcost'] = cost_total
        # feats['s']  = state0[-3]
        tqdm.write(' | '.join([ '{} : {:.3e}'.format(k,v) for k, v in feats.items() if k != 'u']))

        # Store the feature from the current optimization step
        for k, v in feats.items():
            if k not in sol:
                sol[k] = []
            sol[k].append(v)

        # Step in the environment to get the next state
        rng, c_rng = jax.random.split(rng)
        state0 = step_env(state0, _uopt, c_rng)
        state0.block_until_ready()
        sol['state'].append(state0)
        sol['t'].append(sol['t'][-1]+stepsize)
        sol['u'].append(_uopt)
        # if opt_state.opt_cost >= 10000:
        #     break
    return sol, _target

m_rng = jax.random.PRNGKey(2050)
apg_opt_sol, targetSp = play_policy(cfg_dict['sim_scen']['num_steps'], m_rng)

# Plot the results
state_label = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz', 'M1', 'M2', 'M3', 'M4']

state_evol_ = np.array(apg_opt_sol['state'])
u_evol = np.array(apg_opt_sol['u'])

state_evol = [ state_evol_[:, i] for i in range(state_evol_.shape[-1])]
for i in range(u_evol.shape[-1]):
    state_evol.append(u_evol[:, i])

print(len(state_evol))

fig, axs = plt.subplots(4, 4, figsize=(20, 20), sharex=True)
axs = axs.flatten()
for i in range(len(state_evol)-1):
    axs[i].plot(state_evol[i])
    if i < 13:
        if 'state_traj' in cfg_dict:
            axs[i].plot(cfg_dict['state_traj'][:,i], 'r--')
        else:
            assert targetSp is not None, 'Target state is not defined'
            axs[i].plot(np.ones_like(state_evol[i]) * targetSp[i], 'r--')
    if i == 15:
        axs[i].plot(state_evol[i+1], '--')
    axs[i].set_ylabel(state_label[i])
    axs[i].set_xlabel('t')
    axs[i].grid()

plt.show()
