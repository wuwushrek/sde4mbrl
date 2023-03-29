import numpy as np
import os

import jax
import jax.numpy as jnp

import haiku as hk

from sde4mbrlExamples.rotor_uav.sde_rotor_model import load_predictor_function, load_mpc_solver
from sde4mbrlExamples.rotor_uav.utils import load_trajectory, quat_from_euler, quat_to_euler, quatmult, quatinv
from sde4mbrlExamples.rotor_uav.utils import ned_to_enu_position, ned_to_enu_orientation, frd_to_flu_conversion

# Accelerated proximal gradient import
from sde4mbrl.apg import init_apg, apg, init_opt_state
from sde4mbrl.nsde import ValuePolicy, create_value_n_policy_fn, compute_timesteps

import pickle

from functools import partial


state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
state_name2axis = {name: i for i, name in enumerate(state_names)}

def get_list_ofsize_d_summing_to_n(n, d):
    """ Given a number n and a number d,
        return all arrays of size d of integers that sum up to n
    """
    if d == 1:
        return [[n]]
    else:
        return [[i] + rest for i in range(n+1) for rest in get_list_ofsize_d_summing_to_n(n-i, d-1)]


class RotorValuePolicy(ValuePolicy):
    """ A value and a policy functions for the rotor dynamics.
    """
    def __init__(self, params, name=None):
        super().__init__(params, name=name)

    def value_nn_init(self):
        if self.params['value_fn']['type'] == 'quad':
            self.quad_value()
        elif self.params['value_fn']['type'] == 'sos':
            self.sos_value()
        elif self.params['value_fn']['type'] == 'expnn':
            self.expnn_value()
        else:
            raise NotImplementedError


    def quad_value(self):
        val_fn_params = self.params['value_fn']['hidden_layers']
        val_fn_act = self.params['value_fn']['activation_fn']
        active_indx = jnp.array(self.params['value_fn']['active_indx'])

        self.L_nn = hk.nets.MLP(output_sizes=[*val_fn_params, active_indx.shape[0]],
                                        activation = getattr(jnp, val_fn_act) if hasattr(jnp, val_fn_act) else getattr(jax.nn, val_fn_act),
                                        w_init=hk.initializers.RandomUniform(minval=-1e-5, maxval=1e-5),
                                        b_init=jnp.zeros,  name='L_nn'
                                    )
        self.value_fn = lambda x: jnp.sum(jnp.square(self.L_nn(x[active_indx]) * (x[active_indx] - hk.get_parameter(name="xf", shape=(active_indx.shape[0],), init=hk.initializers.RandomUniform(minval=-1e-5,maxval=1e-5))) ) )
        # self.value_fn = lambda x : 0.0

    def expnn_value(self):
        val_fn_params = self.params['value_fn']['hidden_layers']
        val_fn_act = self.params['value_fn']['activation_fn']
        active_indx = jnp.array(self.params['value_fn']['active_indx'])

        self.L_nn = hk.nets.MLP(output_sizes=[*val_fn_params,1],
                                        activation = getattr(jnp, val_fn_act) if hasattr(jnp, val_fn_act) else getattr(jax.nn, val_fn_act),
                                        w_init=hk.initializers.RandomUniform(minval=-1e-5, maxval=1e-5),
                                        b_init=jnp.zeros,  name='L_nn'
                                    )
        self.value_fn = lambda x: 1e-3 * jnp.exp(self.L_nn(x[active_indx])[0])
        # self.value_fn = lambda x : 0.0

    def sos_value(self):
        """ Value functions defined as a sum of square polynomials
        """
        # Extract the polynomial order
        poly_orders = self.params['value_fn']['poly_order']
        # Extract the active indices
        active_indx = jnp.array(self.params['value_fn']['active_indx'])
        # Get the number of active states
        n_active = active_indx.shape[0]
        # Now, we need a function to returns all pairs of indices of size n_active that sum up to poly_order
        # This is not a recursive function
        self.power_arry = [jnp.array(get_list_ofsize_d_summing_to_n(poly_order, n_active)) for poly_order in poly_orders]
        # Now, we need a function to compute the value
        def val_fn(x):
            # First, we need to extract the active states
            x_active = x[active_indx]
            # Now, we need to compute the value
            val = 0.0
            for i, power_arry in enumerate(self.power_arry):
                xpower = jax.vmap(lambda _x, _p : jnp.prod(jnp.power(_x, _p)), in_axes=(None, 0))(x_active, power_arry)
                val += jnp.sum( jnp.square( hk.get_parameter(name=f"coeff_{i}", shape=(power_arry.shape[0], power_arry.shape[0]), init=hk.initializers.RandomUniform(minval=-1e-5,maxval=1e-5)) @ xpower ) )
            return val

        # Store the value function
        self.value_fn = val_fn


    def policy_nn_init(self):
        pol_fn_params = self.params['policy_fn']['hidden_layers']
        pol_fn_act = self.params['policy_fn']['activation_fn']
        self.pol_nn = hk.nets.MLP(output_sizes=[*pol_fn_params, 4],
                    w_init=hk.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                    b_init=jnp.zeros,
                    activation = getattr(jnp, pol_fn_act) if hasattr(jnp, pol_fn_act) else getattr(jax.nn, pol_fn_act),
                    name='pol_nn'
                   )

    def policy_fn(self, x):
        # uval = jnp.tanh(self.pol_nn(x)) * jnp.array(self.params['policy_fn'].get('max_u', 1.))
        uval = self.pol_nn(x)
        thrust = jnp.exp(uval[0])
        moments = jnp.tanh(uval[1:])
        ures = jnp.array([thrust, moments[0], moments[1], moments[2]]) * jnp.array(self.params['policy_fn'].get('max_u', 1.))
        # uval = uval.at[0].set(jnp.clip(uval[0], 0.0, 1.0))
        return ures

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
    return jnp.concatenate((ned_to_enu_position(x[:3], jnp),
                            ned_to_enu_position(x[3:6],jnp),
                            ned_to_enu_orientation(x[6:10], jnp),
                            frd_to_flu_conversion(x[10:], jnp)
                            )
            )

def extract_targets(curr_t, _time_evol, _state_evol, time_steps):
    """ Extract the targets trajectory from the map
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
    return curr_state, xnext

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
        werr = jnp.square(x[10:] - xref[10:]) * jnp.array(_cost_params['werr'])
        if 'werr_sig' in _cost_params:
            werr = jax.nn.sigmoid(werr - 7.0) * jnp.array(_cost_params['werr_sig'])
        werr = jnp.sum(werr)
    
    # Compute the cost on the control
    uerr = 0.0
    if 'uerr' in _cost_params:
        uerr = jnp.square(u - jnp.array(_cost_params['uref'])) * jnp.array(_cost_params['uerr'])
        if 'uerr_sig' in _cost_params:
            uerr = jax.nn.sigmoid(uerr - 7.0) * jnp.array(_cost_params['uerr_sig'])
        uerr = jnp.sum(uerr)

    if 'res_sig' in _cost_params:
        _res_low, _res_mult = _cost_params['res_sig']
        return jnp.tanh((perr + verr + qerr + werr + uerr - _res_low)) * _res_mult
    
    return (perr + verr + qerr + werr + uerr) * jnp.array(_cost_params.get('res_mult',1.0))

def augmented_cost(u_params, _multi_sampling_cost, _cost_params, _time_steps, n_u):
    """ This function augment the cost with slew rate penalty and etc ...
    """
    total_cost, _state_evol = _multi_sampling_cost(u_params)

    # Add the slew rate penalty
    cost_u_slew_rate = jnp.array(0.)
    if 'u_slew_coeff' in _cost_params:
        __time_steps = _time_steps[:-1].reshape(-1, 1)
        cost_u_slew_rate = (jnp.square(u_params[1:, :n_u] - u_params[:-1, :n_u]) / __time_steps) * jnp.array(_cost_params['u_slew_coeff']).reshape(1, -1)
        cost_u_slew_rate = jnp.sum(cost_u_slew_rate) * _cost_params.get('res_mult', 1.0)

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

def reset_apg(x, apg_mpc_params, _constructor_uslack, rng=None, uref=None, _pol_fn=None, convert_to_enu=True, sample_sde=None):
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
        assert uref is not None, "If no policy is provided, the reference input must be provided"
        u_init = uref
        x_init = x
    # u_init = uref
    # x_init = None
    # Get the full u + slack variable
    opt_init = _constructor_uslack(u=u_init, x=x_init)
    # Get the initial mpc state
    opt_state = init_opt_state(opt_init, apg_mpc_params)
    return opt_state


def apg_mpc(xt, rng, past_solution, target_x, multi_cost_sampling, proximal_fn, cfg_dict,
                _sde_learned, _terminal_cost=None, extra_dyn_args=None, convert_to_enu=True):
    """ Run the mpc given current state and
    """
    if convert_to_enu:
        xt = ned2enu(xt)

    # Split the key
    rng_next, rng = jax.random.split(rng)

    red_cost_f = lambda _x, _u, extra_args: one_step_cost_f(_x, _u, extra_args, cfg_dict['cost_params'])

    _cost_xt = lambda u_params: multi_cost_sampling(_sde_learned, xt, u_params, rng,
                                    red_cost_f, _terminal_cost=_terminal_cost,
                                    extra_dyn_args=extra_dyn_args,
                                    extra_cost_args=target_x)

    _timesteps_arr = jnp.array(cfg_dict['_time_steps'])
    cost_xt = lambda u_params: augmented_cost(u_params, _cost_xt, cfg_dict['cost_params'], _timesteps_arr, cfg_dict['model']['n_u'])

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
                                extra_cost_args=target_x)

    uopt = new_opt_state.x_opt[:, :cfg_dict['model']['n_u']]

    # Update the internal state of the optimizer
    # new_solution = new_opt_state._replace(x_opt=new_opt_state.x_opt.at[:-1].set(new_opt_state.x_opt[1:]))
    new_solution = new_opt_state

    if convert_to_enu:
        return uopt, new_solution, rng_next, vehicle_states[0, :, :-1] # Remove the cost value here as it is not needed

    return uopt, new_solution, rng_next, vehicle_states

def load_mpc_from_cfgfile(path_config, convert_to_enu=True, nominal_model=False, one_step_params={}):
    """ Load the mpc from a configuration file
    """
    (_sde_learned, cfg_dict), multi_cost_sampling, vmapped_prox, construct_opt_params = \
            load_mpc_solver(path_config, modified_params={}, nominal_model=nominal_model)

    # Write the proximal function in the right format
    proximal_fn = None if vmapped_prox is None else lambda x, stepsize=None: vmapped_prox(x)

    # Construct the value and policy functions if they are gievn
    contain_valuepol = 'valuepol_model' in cfg_dict
    contain_best_params = False
    if contain_valuepol:
        # Update the value and model params to contains size details
        cfg_dict['valuepol_model']['n_x'] = cfg_dict['model']['n_x']
        cfg_dict['valuepol_model']['n_y'] = cfg_dict['model']['n_y']
        cfg_dict['valuepol_model']['n_u'] = cfg_dict['model']['n_u']
        (_, val_pure), (_, pol_pure) = create_value_n_policy_fn(cfg_dict['valuepol_model'], RotorValuePolicy, seed=0)

        # Check if there is a file with values function parameters
        if 'valuepol_params' in cfg_dict:
            contain_best_params = True
            # Load the pickle file
            print('Loading the best value function parameters from: ', cfg_dict['valuepol_params'])
            with open(os.path.expanduser(cfg_dict['valuepol_params']), 'rb') as f:
                valuepol_params = pickle.load(f)
            print(valuepol_params)
            valuepol_params = jax.tree_map(lambda x: jnp.array(x), valuepol_params)

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
            _val_params=None if not contain_best_params else valuepol_params['value']):
            """ Run the mpc given current state and
            """
            # Extract the current s value
            _, xref = extract_targets(curr_t, time_evol_traj, state_evol_traj, cum_time_steps)
            # Get the value function
            val_fn = None if (_val_params is None or not contain_valuepol) else lambda x: val_pure(_val_params, x)
            return apg_mpc(xt, rng, past_solution, xref, multi_cost_sampling, proximal_fn, cfg_dict, _sde_learned,
                            _terminal_cost=val_fn, extra_dyn_args=None, convert_to_enu=convert_to_enu)

        # Define the reset function
        def m_reset(x=None, xdes=None, uref = jnp.array(cfg_dict['cost_params']['uref']), rng=None,
                    _pol_params=None if not contain_best_params else valuepol_params['policy']):
            """ Reset the apg algorithm """
            pol_fn = None if (_pol_params is None or not contain_valuepol) else lambda x: pol_pure(_pol_params, x)
            return reset_apg(x, cfg_dict['apg_mpc'], construct_opt_params, rng=rng, uref=uref, _pol_fn=pol_fn,
                        convert_to_enu=convert_to_enu, sample_sde=_n_step_function)

        def get_state_at_t(curr_t):
            """ Get the state at a given s value """
            xcurr, _ = extract_targets(curr_t, time_evol_traj, state_evol_traj, cum_time_steps)
            return xcurr

        cfg_dict['state_traj'] = state_evol_traj
        cfg_dict['time_traj'] = time_evol_traj
    else:
        print('SETPOINT-BASED TRACKING')
        # We need to check if cost_params_end is given. This is the cost to go function
        user_cost2go = False
        if 'cost_params_end' in cfg_dict:
            assert 'uerr' not in cfg_dict['cost_params_end'], 'uerr not supported in cost_params_end'
            # assert 'x_xdes_ref' in cfg_dict['cost_params_end'], 'xref not given in cost_params_end'
            assert 'uref' not in cfg_dict['cost_params_end'], 'uref not supported in cost_params_end'
            # assert len(cfg_dict['x_xdes_ref']) == cfg_dict['model']['n_x'], 'xref has wrong dimension'
            # Now we redefine _val_fn to be the cost to go function
            user_cost2go = True
            _val_fn_user = lambda x, xdes: one_step_cost_f(x,None, xdes, cfg_dict['cost_params_end'])
        # Define the reset function
        def m_reset(x=None, xdes=None, uref = jnp.array(cfg_dict['cost_params']['uref']), rng=None,
                    _pol_params=None if not contain_best_params else valuepol_params['policy']):
            """ Reset the apg algorithm """
            pol_none = (_pol_params is None or not contain_valuepol)
            pol_fn = None if pol_none else lambda x: pol_pure(_pol_params, x-xdes)
            return reset_apg(x, cfg_dict['apg_mpc'], construct_opt_params, rng=rng, uref=uref, _pol_fn=pol_fn,
                    convert_to_enu=convert_to_enu, sample_sde=_n_step_function)

        # Define the mpc function
        def m_mpc(xt, rng, past_solution, curr_t = 0.0, xdes=None,
                        _val_params=None if not contain_best_params else valuepol_params['value']):
            """ Run the mpc given current state and
            """
            # Get the value function
            # TODO: Fix the x-xdes to consider also quaternion
            if not user_cost2go:
                val_none = (_val_params is None or not contain_valuepol)
                val_fn = None if val_none else lambda x: val_pure(_val_params, x-xdes)
            else:
                val_fn = lambda x: _val_fn_user(x, xdes)
            # Build the next target
            xtarget = jnp.tile(xdes, (cfg_dict['_time_steps'].shape[0], 1))

            return apg_mpc(xt, rng, past_solution, xtarget, multi_cost_sampling, proximal_fn, cfg_dict, _sde_learned,
                            _terminal_cost=val_fn, extra_dyn_args=None, convert_to_enu=convert_to_enu)

        get_state_at_t = None

    # Import a step function
    one_step_params = {} if 'one_step_params' not in cfg_dict else cfg_dict['one_step_params']
    _one_step_function = load_predictor_function(cfg_dict['learned_model_params'],
                        modified_params={'horizon' : 1, 'num_particles' : 1, **one_step_params}, 
                        prior_dist=nominal_model)

    def one_step_prediction(xt, ut, rng):
        """ One step prediction of the system """
        return _one_step_function(xt, ut, rng)[0,1]

    return cfg_dict, (m_reset, m_mpc), get_state_at_t, one_step_prediction


# def create_simulation_tools(path_config):
#     """ Create the functions for full size simulation using mpc
#     """
#     cfg_dict, (m_reset, m_mpc), get_state_at_t, one_step_prediction = load_mpc_from_cfgfile(path_config, convert_to_enu=False)

#     # Get the initial time and the end time of the simulation
#     t_init = jnp.array(cfg_dict['sim_scen']['t_init'])

#     # Get the number of time steps
#     if get_state_at_t is None:
#         num_time_steps = cfg_dict['sim_scen']['num_steps']
#     else:
#         num_time_steps = len(cfg_dict['time_traj'])
#     # num_time_steps = cfg_dict['sim_scen']['num_steps']


#     if get_state_at_t is not None:
#         state_init = get_state_at_t(t_init)
#     else: # It is a point-based tracking
#         _x, _y, _z, _vx, _vy, _vz, _roll, _pitch, _yaw, _wx, _wy, _wz = cfg_dict['sim_scen']['pos_init']
#         # Create quaternion from yaw
#         quat_init = quat_from_euler(_roll, _pitch, _yaw, jnp)
#         # Normalize the quaternion
#         quat_init = quat_init / jnp.sum(jnp.square(quat_init))
#         state_init = jnp.array([_x, _y, _z, _vx, _vy, _vz, quat_init[0], quat_init[1], quat_init[2], quat_init[3], _wx, _wy, _wz])

#     def reset_env(rng):
#         """ Reset the environment """
#         rng_next, rng = jax.random.split(rng)

#         # Grab the initial state
#         _x, _y, _z, _vx, _vy, _vz, _, _, _, _, _wx, _wy, _wz = state_init

#         # Get the current pitch, roll, yaw
#         roll_, pitch_, yaw_ = quat_to_euler(state_init[6:10], jnp)
#         pos_euler = jnp.array([_x, _y, _z, _vx, _vy, _vz, roll_, pitch_, yaw_, _wx, _wy, _wz])
#         pos_euler_pert = pos_euler + jax.random.normal(rng, pos_euler.shape) * jnp.array(cfg_dict['sim_scen']['init_std'])

#         # Convert back to xyz ...
#         x_, y_, z_, vx_, vy_, vz_, roll_, pitch_, yaw_, wx_, wy_, wz_ = pos_euler_pert

#         # Create quaternion from the angles
#         quat_init = quat_from_euler(roll_, pitch_, yaw_, jnp)

#         # Normalize the quaternion
#         quat_init = quat_init / jnp.sum(jnp.square(quat_init))
#         x_init = jnp.array([x_, y_, jnp.abs(z_), vx_, vy_, vz_, quat_init[0], quat_init[1], quat_init[2], quat_init[3], wx_, wy_, wz_])

#         # Target state
#         xtarget = None
#         if get_state_at_t is None:
#             # Get the initial state and the target state
#             x_, y_, z_ = x_init[:3]
#             xtarget_, ytarget_, ztarget_, yaw_target_ = jnp.array([x_, y_, z_, yaw_]) + jax.random.normal(rng_next, (4,)) * jnp.array(cfg_dict['sim_scen']['target_pos'])

#             # Get quaternion from yaw
#             quat_target = quat_from_euler(0., 0., yaw_target_, jnp)

#             # Normalize the quaternion
#             quat_target = quat_target / jnp.sum(jnp.square(quat_target))
#             xtarget = jnp.array([xtarget_, ytarget_, jnp.abs(ztarget_), 0., 0., 0., quat_target[0], quat_target[1], quat_target[2], quat_target[3], 0., 0., 0.])

#         return x_init, xtarget

#     # Step function
#     def step_env(xt, ut, rng):
#         """ Step the environment """
#         return one_step_prediction(xt, ut, rng) # First sample of next state

#     def one_step_simulation(carry, xs, _val_params=None, _xtarget=None):
#         """ One step of the simulation """

#         # Unpack the carry
#         rng, curr_y, past_solution, _t = carry

#         # Compute the optimal control
#         uopt, next_solution, next_rng, x_traj = m_mpc(curr_y, rng, past_solution, curr_t=_t, xdes=_xtarget,_val_params=_val_params)
#         # curr_cost = jnp.mean(x_traj[:,:cfg_dict['sim_scen']['horizon_cost'],-1])
#         curr_cost = x_traj[0, 1, -1] # 0 should be o -> To test
#         extra_feat = extract_policy(next_solution)

#         # Step the environment
#         rng, n_rng, pert_u_rng = jax.random.split(next_rng, 3)

#         # Check if needs to be greedy
#         # explore = jax.random.choice(pert_u_rng, 2, p=jnp.array([cfg_dict['sim_scen']['expl_prob'], 1.0-cfg_dict['sim_scen']['expl_prob']]))
#         # _u_noise = jax.random.normal(pert_u_rng, (uopt.shape[1],)) * jnp.array(cfg_dict['sim_scen']['expl_std']) + uopt[0]
#         # uval = jnp.where(explore == 0, _u_noise, uopt[0])
#         uval = uopt[0]

#         # Perform the control
#         xs_next = step_env(curr_y, uval, n_rng)

#         # Return the new carry and the new state
#         return (rng, xs_next, next_solution, _t + cfg_dict['model']['stepsize']), (xs_next, uval, curr_cost, extra_feat)

#     @partial(jax.jit, backend='cpu')
#     def simulation_loop(nn_params, rng):
#         """ Run the simulation loop """
#         # Reset the environment
#         rng, n_rng, _mpc_reset_rng = jax.random.split(rng, 3)
#         xs, xtarget = reset_env(n_rng)

#         # Initialize opt
#         opt_state = m_reset(xs, xdes=xtarget, _pol_params=nn_params['policy'], rng = _mpc_reset_rng)
#         # extra_feats = extract_policy(opt_state)
#         # print('Extra features: ', extra_feats)

#         # Run the simulation loop
#         _, (xevol, uopt, curr_cost, extra_feats) = jax.lax.scan(lambda carry, xs : one_step_simulation(carry, xs, _val_params=nn_params['value'], _xtarget=xtarget),
#                         (rng, xs, opt_state, t_init), None, length=num_time_steps)

#         # Concatenate the state without the last state
#         yres = jnp.concatenate((xs[None], xevol))[:-1]

#         # Sum the total cost
#         # cost_eval = jnp.sum(curr_cost) / cfg_dict['sim_scen']['horizon_cost']
#         # curr_cost = jnp.cumsum(curr_cost)[::-1]
#         cost_eval = jnp.sum(curr_cost) # [0]
#         # extra_feats['costT'] = curr_cost
#         # extra_feats['costT'] = extra_feats['costF']

#         extra_feats['costT'] = curr_cost

#         return yres, (uopt, cost_eval, extra_feats)

#     # Build the simulation tool packet
#     sim_tools = {'n_y' : 13, 'n_u': 4, 'opt2state_fn': lambda x, rng : x,
#                 'episode_length' : num_time_steps,
#                 'mpc_sim' : simulation_loop}

#     return cfg_dict, sim_tools

# def extract_policy(opt_state):
#     return {'avg_linesearch': opt_state.avg_linesearch, 'Thro' : opt_state.yk[0,0], 'Elev' : opt_state.yk[0,1],
#                 'num_steps' : opt_state.num_steps, 'grad_norm': opt_state.grad_sqr,
#                 'avg_stepsize': opt_state.avg_stepsize, 'cost0': opt_state.init_cost,
#                 'costF': opt_state.opt_cost }

# # If main
# if __name__ == '__main__':
#     import argparse
#     import os

#     from sde4mbrl.train_sde import train_value_policy

#     # Read command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, help='Path to the yaml config file')
#     parser.add_argument('--out', type=str, help='Path to the output folder')
#     # GEt the arguments
#     args = parser.parse_args()

#     # Load the simulation tools
#     cfg_dict, sim_tools = create_simulation_tools(args.cfg)

#     # Train the model
#     train_value_policy(cfg_dict, args.out, RotorValuePolicy, sim_tools)
