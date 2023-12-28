import numpy as np
import os

import jax
import jax.numpy as jnp


from sde4mbrlExamples.rotor_uav.sde_rotor_model import load_predictor_function, load_mpc_solver
from sde4mbrlExamples.rotor_uav.utils import load_trajectory, quat_from_euler, quat_to_euler, quatmult, quatinv
from sde4mbrlExamples.rotor_uav.utils import ned_to_enu_position, ned_to_enu_orientation, frd_to_flu_conversion

# Accelerated proximal gradient import
from sde4mbrl.apg import init_apg, apg, init_opt_state
from sde4mbrl.nsde import compute_timesteps

import pickle

from functools import partial


state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
state_name2axis = {name: i for i, name in enumerate(state_names)}


def parse_trajectory(_traj):
    """ Return the array of time and concatenate the other states
        _traj: a dictionary with the keys: t, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz
    """
    # List of states in order
    states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    time_val = jnp.array(_traj['t'])
    # stack the states
    state_val = jnp.stack([_traj[state] for state in states], axis=1)
    return time_val, state_val


def ned2enu(x):
    """ Convert the state from NED to ENU
    """
    return jnp.concatenate((ned_to_enu_position(x[:3], jnp),
                            ned_to_enu_position(x[3:6],jnp),
                            ned_to_enu_orientation(x[6:10], jnp),
                            frd_to_flu_conversion(x[10:13], jnp),
                            x[13:]
                            )
            )

def extract_targets(curr_t, _time_evol, _state_evol, time_steps):
    """ Extract the targets trajectory from the map (reference target used in the MPC algorithm)
    """
    # Closest index from s [right index]
    indx_next = jnp.searchsorted(_time_evol, curr_t)

    # Clip the closest index to be valid
    indx_next = jnp.clip(indx_next-1, 0, _time_evol.shape[0]-2)

    # Extrapolated time
    curr_state = _state_evol[indx_next] + ( (_state_evol[indx_next+1] - _state_evol[indx_next]) * (curr_t - _time_evol[indx_next]) / (_time_evol[indx_next+1] - _time_evol[indx_next]) )

    # Get the sequence of next time steps
    next_time_steps = curr_t + time_steps

    # Here compute the index awhere to fetch the target states and clip these indexes values
    indx_tnext = jnp.searchsorted(_time_evol, next_time_steps)
    indx_tnext = jnp.clip(indx_tnext-1, 0, _time_evol.shape[0]-2)

    extr_fn = lambda _i, _t, _arr: _arr[_i] + (_arr[_i+1] - _arr[_i]) * (_t - _time_evol[_i]) / (_time_evol[_i+1] - _time_evol[_i])

    xnext = jax.vmap(extr_fn, in_axes=(0,0,None))(indx_tnext, next_time_steps, _state_evol)
    return curr_state, jnp.concatenate((curr_state[None],xnext))

# Define a cost function
def one_step_cost_f(x, u, extra_args, _cost_params):
    """ Generic cost function
        x is a (N, 13) array
        u is a (N, 4) array
        The cost function is a scalar
    """
    xref = extra_args
    # Compute the error on the quaternion
    qerr = 0.0
    if 'qerr' in _cost_params:
        qerr = jnp.square(quatmult(xref[6:10], quatinv(x[6:10], jnp), jnp)[1:]) * jnp.array(_cost_params['qerr'])
        if 'qerr_sig' in _cost_params:
            qerr = jax.nn.sigmoid(qerr - 7.0) * jnp.array(_cost_params['qerr_sig'])
        qerr = jnp.sum(qerr)
    
    # Compute the error on the position
    perr = 0.0
    if 'perr' in _cost_params:
        perr = jnp.square(x[:3] - xref[:3]) * jnp.array(_cost_params['perr'])
        if 'perr_sig' in _cost_params:
            perr = jax.nn.sigmoid(perr - 7.0) * jnp.array(_cost_params['perr_sig'])
        perr = jnp.sum(perr)
    
    # Compute the error on the velocity
    verr = 0.0
    if 'verr' in _cost_params:
        verr = jnp.square(x[3:6] - xref[3:6]) * jnp.array(_cost_params['verr'])
        if 'verr_sig' in _cost_params:
            verr = jax.nn.sigmoid(verr - 7.0) * jnp.array(_cost_params['verr_sig'])
        verr = jnp.sum(verr)

    # Compute the error on the angular velocity
    werr = 0.0
    if 'werr' in _cost_params:
        werr = jnp.square(x[10:13] - xref[10:13]) * jnp.array(_cost_params['werr'])
        if 'werr_sig' in _cost_params:
            werr = jax.nn.sigmoid(werr - 7.0) * jnp.array(_cost_params['werr_sig'])
        werr = jnp.sum(werr)
    
    # Compute the cost on the control
    uerr = 0.0
    if 'uerr' in _cost_params:
        # u_shape = u.shape[0]
        u = x[13:] if 'urate_err' in _cost_params else u
        uerr = jnp.square(u - jnp.array(_cost_params['uref'])) * jnp.array(_cost_params['uerr'])
        if 'uerr_sig' in _cost_params:
            uerr = jax.nn.sigmoid(uerr - 7.0) * jnp.array(_cost_params['uerr_sig'])
        uerr = jnp.sum(uerr)
    
    urate_err = 0.0
    if 'urate_err' in _cost_params:
        # By default, 
        urate_err = jnp.square(u) * jnp.array(_cost_params['urate_err'])
        urate_err = jnp.sum(urate_err)

    if 'res_sig' in _cost_params:
        _res_low, _res_mult = _cost_params['res_sig']
        return jnp.tanh((perr + verr + qerr + werr + uerr - _res_low)) * _res_mult
    
    return (perr + verr + qerr + werr + uerr + urate_err) * jnp.array(_cost_params.get('res_mult',1.0))

def augmented_cost(u_params, _multi_sampling_cost, _cost_params, _time_steps, n_u):
    """ This function augment the cost with slew rate penalty and etc ...
    """
    total_cost, _state_evol = _multi_sampling_cost(u_params)

    # Add the slew rate penalty
    cost_u_slew_rate = jnp.array(0.)
    u_slew = None
    if 'u_slew_coeff' in _cost_params and (not 'urate_err' in _cost_params):
        __time_steps = _time_steps[:-1].reshape(-1, 1)
        u_slew = (u_params[1:, :n_u] - u_params[:-1, :n_u]) / __time_steps
        cost_u_slew_rate = (jnp.square(u_slew)) * jnp.array(_cost_params['u_slew_coeff']).reshape(1, -1)
        cost_u_slew_rate = jnp.sum(cost_u_slew_rate) * _cost_params.get('res_mult', 1.0)
    
    if 'u_slew_constr' in _cost_params and (not 'urate_err' in _cost_params):
        assert u_slew is not None, 'The slew rate must be computed before'
        u_slew_bounds = jnp.array(_cost_params['u_slew_constr'])
        u_slew_constr_coeff = jnp.array(_cost_params['u_slew_constr_coeff'])
        diff_lower = jnp.minimum(u_slew - u_slew_bounds[:, 0][None], 0.0)
        diff_upper = jnp.maximum(u_slew - u_slew_bounds[:, 1][None], 0.0)
        cost_u_slew_rate += jnp.sum( (jnp.square(diff_lower) + jnp.square(diff_upper)) * u_slew_constr_coeff.reshape(1,-1) ) * _cost_params.get('res_mult', 1.0)

    # State slew rate
    state_slew_rate = jnp.array(0.)
    if 'state_slew_active' in _cost_params:
        # Only valid for vx,vy,vz, and wx,wy,wz
        assert len(_cost_params['state_slew_active']) == len(_cost_params['state_slew_coeffs']), 'The number of active states and the number of slew rate coefficients must be the same'
        indx_active_params = jnp.array([ state_name2axis[_name] for _name in _cost_params['state_slew_active'] ])
        _time_steps = _time_steps.reshape(1, -1, 1)
        slew_state = (_state_evol[:, 1:, indx_active_params] - _state_evol[:, :-1, indx_active_params]) / _time_steps
        slew_coeffs = jnp.array(_cost_params['state_slew_coeffs'])
        state_slew_rate = jnp.sum( jnp.square(slew_state) * slew_coeffs.reshape(1,1,-1) ) * _cost_params.get('res_mult', 1.0)

    return total_cost + cost_u_slew_rate + state_slew_rate

def reset_apg(x, apg_mpc_params, _constructor_uslack, rng=None, uref=None, 
              _pol_fn=None, convert_to_enu=True, sample_sde=None):
    """ This function takes as imput the current observation, the constructor of the uslack function
        and the policy function and returns an initial state of the apg algorithm
    """
    if _pol_fn is not None:
        if convert_to_enu:
            x = ned2enu(x)
        if sample_sde is not None:
            assert rng is not None, "If the SDE is sampled, the random number generator must be provided"
            x_init, u_init = sample_sde(x, _pol_fn, rng)
            x_init, u_init = x_init[0], u_init[0] # Take only the first sample
        else:
            u_init = _pol_fn(x)
            x_init = x
    else:
        # assert uref is not None, "If no policy is provided, the reference input must be provided"
        u_init = uref
        x_init = x
    # u_init = uref
    # x_init = None
    # Get the full u + slack variable
    opt_init = _constructor_uslack(u=u_init, x=x_init)
    # Get the initial mpc state
    opt_state = init_opt_state(opt_init, apg_mpc_params)
    return opt_state

def create_full_cost_function(xt, rng, target_x, multi_cost_sampling, cfg_dict, _sde_learned,
                _terminal_cost=None, extra_dyn_args=None, convert_x=False, mod_cost_params={}):
    """ Create the full cost function"""

    if convert_x:
        xt = ned2enu(xt)

    # Update the cost parameters dictionary
    cfg_dict['cost_params'].update(mod_cost_params)

    red_cost_f = lambda _x, _u, extra_args: one_step_cost_f(_x, _u, extra_args, cfg_dict['cost_params'])
    
    # TODO: Check the etra_cost_terminal_args
    # assert target_x.shape[0] == len(cfg_dict['_time_steps'])+1, 'The target_x must be of size (N+1, n_x)'
    _cost_xt = lambda u_params: multi_cost_sampling(_sde_learned, xt, u_params, rng,
                                    red_cost_f, _terminal_cost=_terminal_cost,
                                    extra_dyn_args=extra_dyn_args,
                                    extra_cost_args=target_x,
                                    extra_cost_terminal_args=target_x[-1])
    
    _timesteps_arr = jnp.array(cfg_dict['_time_steps'])
    cost_xt = lambda u_params: augmented_cost(u_params, _cost_xt, cfg_dict['cost_params'], _timesteps_arr, cfg_dict['model']['n_u'])
    
    return cost_xt, red_cost_f

def apg_mpc(xt, rng, past_solution, target_x, multi_cost_sampling, proximal_fn, cfg_dict,
                _sde_learned, _terminal_cost=None, extra_dyn_args=None, convert_to_enu=True, 
                mod_cost_params={}):
    """ Run the mpc given current state and
    """
    # Split the key
    rng_next, rng = jax.random.split(rng)

    if convert_to_enu:
        xt = ned2enu(xt)

    cost_xt, red_cost_f = create_full_cost_function(xt, rng, target_x, multi_cost_sampling, cfg_dict, _sde_learned,
                                        _terminal_cost, extra_dyn_args, convert_x=False, mod_cost_params=mod_cost_params)

    # First reinitialize the optimizer
    opt_state = init_apg(past_solution.x_opt, cost_xt,
                        cfg_dict['apg_mpc'],
                        momentum=past_solution.momentum,
                        stepsize=past_solution.avg_stepsize)


    # new_opt_state = one_step_apg(opt_state, cost_xt, cfg_dict['apg_mpc'], proximal_fn)
    new_opt_state = apg(opt_state, cost_xt, cfg_dict['apg_mpc'], proximal_fn=proximal_fn)

    # Compute the next states of the system
    _, vehicle_states = multi_cost_sampling(_sde_learned, xt, new_opt_state.x_opt, rng, red_cost_f,
                                _terminal_cost=_terminal_cost,
                                extra_dyn_args=extra_dyn_args,
                                extra_cost_args=target_x,
                                extra_cost_terminal_args=target_x[-1])

    uopt = new_opt_state.x_opt[:, :cfg_dict['model']['n_u']]

    # Update the internal state of the optimizer
    # new_solution = new_opt_state._replace(x_opt=new_opt_state.x_opt.at[:-1].set(new_opt_state.x_opt[1:]))
    new_solution = new_opt_state

    # TODO; Change this for the enu case with Gazebo as the control is dependent on if controlling rate or not
    if convert_to_enu:
        return uopt, new_solution, rng_next, vehicle_states[0, :, :xt.shape[0]] # Remove the cost value here as it is not needed

    return uopt, new_solution, rng_next, vehicle_states

def load_mpc_from_cfgfile(path_config, convert_to_enu=True, nominal_model=False, 
                          one_step_params={}, return_full_cost=False):
    """ Load the mpc from a configuration file
    """
    (_sde_learned, cfg_dict), multi_cost_sampling, vmapped_prox, construct_opt_params = \
            load_mpc_solver(path_config, modified_params={}, nominal_model=nominal_model)
    
    # Do some clean up depending on if we are doing rate control or not
    if cfg_dict['model'].get('control_augmented_state', False):
        assert 'uref' in cfg_dict['cost_params'], 'The uref must be provided in the cost parameters'
        assert 'urate_err' in cfg_dict['cost_params'], 'The urate_err must be provided in the cost parameters'
        cfg_dict['cost_params'].pop('u_slew_coeff', None)
    else:
        cfg_dict['cost_params'].pop('urate_err', None)

    # Write the proximal function in the right format
    proximal_fn = None if vmapped_prox is None else lambda x, stepsize=None: vmapped_prox(x)

    # Construct the value and policy functions if they are gievn
    # contain_valuepol = 'valuepol_model' in cfg_dict
    contain_valuepol = False
    contain_best_params = False

    # TODO: This was not used for the paper. valuepol_params, val_pure, pol_pure are not defined but won't be used
    # To be removed
    # if contain_valuepol:
    #     # Update the value and model params to contains size details
    #     cfg_dict['valuepol_model']['n_x'] = cfg_dict['model']['n_x']
    #     cfg_dict['valuepol_model']['n_y'] = cfg_dict['model']['n_y']
    #     cfg_dict['valuepol_model']['n_u'] = cfg_dict['model']['n_u']
    #     (_, val_pure), (_, pol_pure) = create_value_n_policy_fn(cfg_dict['valuepol_model'], RotorValuePolicy, seed=0)

    #     # Check if there is a file with values function parameters
    #     if 'valuepol_params' in cfg_dict:
    #         contain_best_params = True
    #         # Load the pickle file
    #         print('Loading the best value function parameters from: ', cfg_dict['valuepol_params'])
    #         with open(os.path.expanduser(cfg_dict['valuepol_params']), 'rb') as f:
    #             valuepol_params = pickle.load(f)
    #         print(valuepol_params)
    #         valuepol_params = valuepol_params[cfg_dict.get('valuepol_type', 'opt')]
    #         valuepol_params = jax.tree_map(lambda x: jnp.array(x), valuepol_params)

    # Extract the timesteps used in the mpc
    cfg_dict['_time_steps'] = np.array(compute_timesteps(cfg_dict['model']), dtype=np.float32)

    # Print the configuration dictionary
    print(cfg_dict)

    # N-step predictor
    _n_step_function = load_predictor_function(cfg_dict['learned_model_params'],
                        modified_params={'num_particles' : 1, **{k : cfg_dict['model'][k] for k in ['horizon', 'num_short_dt', 'short_step_dt', 'long_step_dt']} },
                        prior_dist=nominal_model, return_control=True)

    # Load trajectory from the configuration file
    # Two cases here -? Either a trajectory is given or setpoint-based tracking
    if 'trajectory_path' in cfg_dict:
        print('TRAJECTORY TRACKING')
        traj_vector = load_trajectory(cfg_dict['trajectory_path'])
        time_evol_traj, state_evol_traj = parse_trajectory(traj_vector)
        cum_time_steps = np.cumsum(np.array(cfg_dict['_time_steps']))

        # Define the mpc function
        def m_mpc(xt, rng, past_solution, curr_t = 0.0, xdes=None,
                _val_params=None if not contain_best_params else valuepol_params['value'],
                return_ref=False, mod_cost_params={}):
            """ Run the mpc given current state and
            """
            # Extract the current s value
            _, xref = extract_targets(curr_t, time_evol_traj, state_evol_traj, cum_time_steps)
            # Get the value function
            val_fn = None if (_val_params is None or not contain_valuepol) else lambda x, feat: val_pure(_val_params, x, feat)
            _res_mpc =  apg_mpc(xt, rng, past_solution, xref, multi_cost_sampling, proximal_fn, cfg_dict, _sde_learned,
                            _terminal_cost=val_fn, extra_dyn_args=None, convert_to_enu=convert_to_enu, mod_cost_params=mod_cost_params)
            if return_ref:
                return _res_mpc, xref
            return _res_mpc

        # Define the reset function
        def m_reset(x=None, xdes=None, uref = jnp.array(cfg_dict['cost_params']['uref']), rng=None,
                    _pol_params=None if not contain_best_params else valuepol_params['policy']):
            """ Reset the apg algorithm """
            pol_fn = None if (_pol_params is None or not contain_valuepol) else lambda x: pol_pure(_pol_params, x)
            return reset_apg(x, cfg_dict['apg_mpc'], construct_opt_params, rng=rng, uref=uref, _pol_fn=pol_fn,
                        convert_to_enu=convert_to_enu, sample_sde=_n_step_function)

        def get_state_at_t(curr_t, _cum_time_steps=None, _return_evol=False):
            """ Get the state at a given s value """
            if _cum_time_steps is None:
                _cum_time_steps = cum_time_steps
            xcurr, _xevol = extract_targets(curr_t, time_evol_traj, state_evol_traj, _cum_time_steps)
            if _return_evol:
                return xcurr, _xevol
            return xcurr

        cfg_dict['state_traj'] = state_evol_traj
        cfg_dict['time_traj'] = time_evol_traj
    else:
        print('SETPOINT-BASED TRACKING')
        # We need to check if cost_params_end is given. This is the cost to go function
        user_cost2go = False
        if 'cost_params_end' in cfg_dict:
            assert 'uerr' not in cfg_dict['cost_params_end'], 'uerr not supported in cost_params_end'
            assert 'uref' not in cfg_dict['cost_params_end'], 'uref not supported in cost_params_end'
            # assert len(cfg_dict['x_xdes_ref']) == cfg_dict['model']['n_x'], 'xref has wrong dimension'
            # Now we redefine _val_fn to be the cost to go function
            user_cost2go = True
            _val_fn_user = lambda x, feat : one_step_cost_f(x, None, feat, cfg_dict['cost_params_end'])
        # Define the reset function
        default_uref = jnp.array(cfg_dict['cost_params']['uref']) if 'control_augmented_state' not in cfg_dict['model'] else None
        def m_reset(x=None, xdes=None, uref = default_uref, rng=None,
                    _pol_params=None if not contain_best_params else valuepol_params['policy']):
            """ Reset the apg algorithm """
            pol_none = (_pol_params is None or not contain_valuepol)
            pol_fn = None if pol_none else lambda x: pol_pure(_pol_params, x, xdes)
            return reset_apg(x, cfg_dict['apg_mpc'], construct_opt_params, rng=rng, uref=uref, _pol_fn=pol_fn,
                    convert_to_enu=convert_to_enu, sample_sde=_n_step_function)

        # Define the mpc function
        def m_mpc(xt, rng, past_solution, curr_t = 0.0, xdes=None,
                        _val_params=None if not contain_best_params else valuepol_params['value'],
                        return_ref=False, mod_cost_params={}):
            """ Run the mpc given current state and
            """
            # Build the next target
            # xtarget = jnp.tile(xdes, (cfg_dict['_time_steps'].shape[0], 1))
            xtarget = jnp.tile(xdes, (cfg_dict['_time_steps'].shape[0]+1, 1))
            # xt_conv = xt
            # if convert_to_enu:
            #     xt_conv = ned2enu(xt)
            # xtarget = jnp.concatenate((xt_conv[None], xtarget))

            # Get the value function
            # TODO: Fix the x-xdes to consider also quaternion
            if not user_cost2go:
                val_none = (_val_params is None or not contain_valuepol)
                val_fn = None if val_none else lambda x, feat: val_pure(_val_params, x, feat)
            else:
                val_fn = _val_fn_user

            _res_mpc = apg_mpc(xt, rng, past_solution, xtarget, multi_cost_sampling, proximal_fn, cfg_dict, _sde_learned,
                            _terminal_cost=val_fn, extra_dyn_args=None, convert_to_enu=convert_to_enu, mod_cost_params=mod_cost_params)
            
            if return_ref:
                return _res_mpc, xtarget
            return _res_mpc

        get_state_at_t = None

    # Import a step function
    one_step_params = {} if 'one_step_params' not in cfg_dict else cfg_dict['one_step_params']
    _one_step_function = load_predictor_function(cfg_dict['learned_model_params'],
                        modified_params={'horizon' : 1, 'num_particles' : 1, 
                                         'control_augmented_state' : cfg_dict['model'].get('control_augmented_state', False), 
                                         **one_step_params}, 
                        prior_dist=nominal_model)

    def one_step_prediction(xt, ut, rng):
        """ One step prediction of the system """
        return _one_step_function(xt, ut, rng)[0,1]

    # # Save the sde loaded inside the cfg_dict
    # cfg_dict['sde_learned'] = _sde_learned

    if return_full_cost:
        full_cost_fn = lambda y, u, rng, terminal_cost_fn, extra_dyn_args=None, extra_cost_args=None : create_full_cost_function(y, rng, extra_cost_args, 
                                multi_cost_sampling, cfg_dict, _sde_learned, terminal_cost_fn, extra_dyn_args, convert_to_enu)[0](u)
        return (cfg_dict, (m_reset, m_mpc), get_state_at_t, one_step_prediction), full_cost_fn
    return cfg_dict, (m_reset, m_mpc), get_state_at_t, one_step_prediction
    

def extract_policy(opt_state):
    return {'avg_linesearch': opt_state.avg_linesearch, 'M1' : opt_state.yk[0,0], 'M2' : opt_state.yk[0,1],
                'num_steps' : opt_state.num_steps, 'grad_norm': opt_state.grad_sqr,
                'avg_stepsize': opt_state.avg_stepsize, 'cost0': opt_state.init_cost,
                'costF': opt_state.opt_cost }