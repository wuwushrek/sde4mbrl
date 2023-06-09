import jax
import jax.numpy as jnp

import numpy as np

from sde4mbrlExamples.rotor_uav.utils import quatmult as quatmult_gen
from sde4mbrlExamples.rotor_uav.utils import quat_rotatevector as quat_rotatevector_gen
from sde4mbrlExamples.rotor_uav.utils import quat_rotatevectorinv as quat_rotatevectorinv_gen

from sde4mbrlExamples.rotor_uav.motor_models import *

from sde4mbrl.nsde import ControlledSDE, create_sampling_fn, create_online_cost_sampling_fn, compute_timesteps
from sde4mbrl.utils import load_yaml, update_params, apply_fn_to_allleaf

import haiku as hk

import os
import pickle

# Make sure jax is used for all these function calls
quatmult = lambda a, b: quatmult_gen(a, b, jnp)
quat_rotatevector = lambda q, v: quat_rotatevector_gen(q, v, jnp)
quat_rotatevectorinv = lambda q, v: quat_rotatevectorinv_gen(q, v, jnp)

def get_pos(state):
    """Get the position from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The pose of the drone.
    """
    return state[0:3]

def set_pos(state, pos):
    """Set the position from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The pose of the drone.
    """
    return state.at[0:3].set(pos)

def get_vel(state):
    """Get the velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The velocity of the drone.
    """
    return state[3:6]

def set_vel(state, vel):
    """Set the velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The velocity of the drone.
    """
    return state.at[3:6].set(vel)

def get_quat(state):
    """Get the quaternion from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The quaternion of the drone.
    """
    return state[6:10]

def set_quat(state, q):
    """Set the quaternion from the state.
       The convention is q0 + q1*i + q2*j + q3*k
    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The quaternion of the drone.
    """
    return state.at[6:10].set(q)

def get_ang_vel(state):
    """Get the angular velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The angular velocity of the drone.
    """
    return state[10:13]

def set_ang_vel(state, w):
    """Set the angular velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The angular velocity of the drone.
    """
    return state.at[10:13].set(w)

class SDERotorModel(ControlledSDE):
    """SDE model of the UAV rotors
    """
    def __init__(self, params, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name=name)

        # Parameters initialization values -> Values for HK PARAMETERS initialization
        self.init_params = params['init_params']
        assert 'fm_model' in self.init_params, "The motor models must be specified"
        # Tese could be polynomial models as specified in motor_models.py

        # Initialization of the residual networks
        self.init_residual_networks()

        self.state_scaling = jnp.array(self.params.get('state_scaling', [1.0] * self.n_x))
        max_scaling = jnp.max(self.state_scaling)

        # In case scaling factor is give, we also need to ensure scaling diffusion network inputs
        if 'state_scaling' in self.params:
            # self.reduced_state = lambda x : x / self.state_scaling
            self.reduced_state = lambda x : x / max_scaling

    def get_param(self, param_name):
        """Get the value of the parameter with the given name.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            float: The value of the parameter.
        """
        init_value = self.init_params.get(param_name, 1e-3) # Small non-zero value
        if param_name in self.init_params and self.params.get('fixed_init_params', False):
            return init_value
        return hk.get_parameter(param_name, shape=(), init=hk.initializers.Constant(init_value))

    # Define a projection function specifically for the quaternions
    def projection_fn(self, x):
        """Projection function for the quaternion of the state
            Args:
                x (jax.numpy.ndarray): The state of the drone.
            Returns:
                jax.numpy.ndarray: The projected state of the drone.
        """
        # TODO: Differentiation issues here???
        quat_val = get_quat(x)
        norm_q = quat_val / jnp.linalg.norm(quat_val)
        return set_quat(x, norm_q)

    def translational_dynamics(self, x, Fz, Fres=None):
        """ Given the state x and the thrust and residual forces not explained by model,
            This is ENU coordinate frame with z up. return the derivative of the state
            Args:
                x (jax.numpy.ndarray): The state of the drone.
                Fz (jax.numpy.ndarray): The thrust force.
                Fres (jax.numpy.ndarray): The residual force -> Unmodelled Aerodynamics force
            Returns:
                jax.numpy.ndarray: The derivative of the position.
                jax.numpy.ndarray: The derivative of the velocity.

        """
        # Get the mass -> This is a parameter that can be learned
        m = self.get_param('mass')
        # Get the gravity
        g = self.init_params.get('gravity', 9.81)

        # Get the total thrust
        F = jnp.array([0., 0., Fz])
        if Fres is not None:
            F += Fres

        # Rotate the thrust into the inertial frame
        q = get_quat(x)
        F = quat_rotatevector(q, F)

        # Compute the acceleration
        a = (F / m) + jnp.array([0., 0., -g])

        # Get the velocity
        pos_dot = get_vel(x)

        # Return the derivative of the state
        return pos_dot, a

    def rotational_dynamics(self, x, Mxyz, Mres=None):
        """ Given the state x and the moments and residual moments not explained by model,
            This is in the body frame. Return the derivative of the state
            Args:
                x (jax.numpy.ndarray): The state of the drone.
                Mxyz (jax.numpy.ndarray): The moments.
                Mres (jax.numpy.ndarray): The residual moments -> Unmodelled Aerodynamics moments
            Returns:
                jax.numpy.ndarray: The derivative of the quaternion.
                jax.numpy.ndarray: The derivative of the angular velocity.
        """
        # Get the inertia -> These are parameters that can be learned
        I = jnp.array([self.get_param('Ixx'), self.get_param('Iyy'), self.get_param('Izz')])

        # Get the total moment
        M = Mxyz
        if Mres is not None:
            M += Mres

        # Get the angular velocity
        ang_vel = get_ang_vel(x)

        # Compute the cross product between the angular velocity and the inertia and the angular velocity
        ang_vel_cross = jnp.cross(ang_vel, I * ang_vel)

        # Update the total moment
        M -= ang_vel_cross

        # Compute the angular acceleration
        ang_acc = M / I

        # Get the quaternion derivative
        quat_dot = 0.5 * quatmult(get_quat(x), jnp.array([0., ang_vel[0], ang_vel[1], ang_vel[2]]))

        return quat_dot, ang_acc

    def fm_model(self, u):
        """Compute the thrust and moment generated by the motors.

        Args:
            u (jax.numpy.ndarray): The input of the quadrotor.

        Returns:
            jax.numpy.ndarray: The thrust generated by the motors.
            jax.numpy.ndarray: The moment generated by the motors.
        """
        return motor_model(self.get_param, u, **self.init_params['fm_model'])

    def aero_drag(self, v_b):
        """Aerodynamic drag prior model
            Args:
                v_b (jax.numpy.ndarray): The velocity of the drone in the body frame.
            Returns:
                jax.numpy.ndarray: The residual force resulting from aerodynamic drag.
        """
        # Now we can compute the drag force
        # We use the linear drag force assumption -> Initial estimate close to zero
        kdx = hk.get_parameter('aero_kdx', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kdy = hk.get_parameter('aero_kdy', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kdz = hk.get_parameter('aero_kdz', shape=(), init=hk.initializers.RandomUniform(0., 0.001))
        kh = hk.get_parameter('aero_kh', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
        return jnp.array([-kdx*v_b[0], -kdy*v_b[1], -kdz*v_b[2] + kh*(jnp.sum(jnp.square(v_b[:2])))])

    def ground_effect(self, x):
        """Ground effect modelization
            Args:
                x (jax.numpy.ndarray): The state of the drone.
        """
        # TODO: Implement the ground effect
        # Project the position in the local frame
        # Then the results is dependent on the projected height
        # This is a hack for the thrust response to see if the static learning worked well
        # k3 = hk.get_parameter('Ft', shape=(), init=hk.initializers.RandomUniform(0., 0.0001))
        # return jnp.array([1.0+k3, 0.])
        return None

    def compute_force_residual(self, x, v_b):
        """Compute the residual aerodynamic forces
            Args:
                x (jax.numpy.ndarray): The state of the drone.
                v_b (jax.numpy.ndarray): The velocity of the drone in the body frame.
            Returns:
                jax.numpy.ndarray: The residual force.
        """
        # Check if the force_residual is enabled
        if not ('residual_forces' in self.params or 'aero_drag_effect' in self.params):
            return None

        # Create an MLP to compute the residual
        # The parameters are store in params dictionary under residual_forces
        Fres = jnp.zeros(3)

        # The residual terms are function of the velocity and angular velocity in the body frame
        if 'residual_forces' in self.params:
            in_vect = jnp.array([v_b[0], v_b[1], v_b[2], x[10], x[11], x[12]])
            in_vect_scale = jnp.array([self.state_scaling[3], self.state_scaling[4], self.state_scaling[5],
                                       self.state_scaling[10], self.state_scaling[11], self.state_scaling[12]])
            in_vect = in_vect / in_vect_scale
            active_indx = self.params['residual_forces'].get('active_indx', None)
            in_vect = in_vect[jnp.array(active_indx)] if active_indx is not None else in_vect
            stab_coeff = 1.0
            if self.params.get('stability', False):
                stab_coeff = jnp.sum(jnp.square(in_vect))
            Fres += self.residual_forces(in_vect) * stab_coeff

        if self.params.get('aero_drag_effect', False):
            Fres += self.aero_drag(v_b)

        return Fres

    def compute_moment_residual(self, x, v_b):
        """Compute the residual aerodynamic moments
            Args:
                x (jax.numpy.ndarray): The state of the drone.
                v_b (jax.numpy.ndarray): The velocity of the drone in the body frame.
            Returns:
                jax.numpy.ndarray: The residual moment.
        """
        if not 'residual_moments' in self.params:
            return None
        # Create an MLP to compute the residual
        # The parameters are store in params dictionary under residual_forces
        in_vect = jnp.array([v_b[0], v_b[1], v_b[2], x[10], x[11], x[12]])
        in_vect_scale = jnp.array([self.state_scaling[3], self.state_scaling[4], self.state_scaling[5],
                                       self.state_scaling[10], self.state_scaling[11], self.state_scaling[12]])
        in_vect = in_vect / in_vect_scale
        active_indx = self.params['residual_moments'].get('active_indx', None)
        in_vect = in_vect[jnp.array(active_indx)] if active_indx is not None else in_vect
        stab_coeff = 1.0
        if self.params.get('stability', False):
            stab_coeff = jnp.sum(jnp.square(in_vect))
        Mres = self.residual_moments(in_vect) * stab_coeff
        return Mres

    def get_effective_thrust(self, thrust_, ge_effect):
        """Compute the effective thrust
            Args:
                thrust_ (jax.numpy.ndarray): The thrust generated by the motors.
                x (jax.numpy.ndarray): The state of the drone.
                ge_effect (jax.numpy.ndarray): The ground effect multiplicative value that is learned
            Returns:
                jax.numpy.ndarray: The effective thrust.
        """
        if ge_effect is not None:
            thrust_ *= ge_effect
        return thrust_

    def vector_field(self, x, u, Fres=None, Mres=None, ge_effect=None):
        """Compute the vector field of the dynamics of the quadrotor.
            The geometric parameters of the drone are unknown as well as the aerodynamics forces and moments.

        Args:
            x (jax.numpy.ndarray): The state of the quadrotor.
            u (jax.numpy.ndarray): The input of the quadrotor.
            Fres (jax.numpy.ndarray, optional): The residual force. Defaults to None -> means zero
            Mres (jax.numpy.ndarray, optional): The residual moment. Defaults to None -> means zero

        Returns:
            jax.numpy.ndarray: The derivative of the state.
        """
        # COmpute the thrust generated by the motors and the corresponding moments
        thrust_, moment = self.fm_model(u) # fm_model is parameterized and learned too
        thrust = self.get_effective_thrust(thrust_, ge_effect) # Account for learned ground model

        # Compute the translational dynamics using side information if enabled
        pos_dot, v_dot = self.translational_dynamics(x, thrust, Fres)

        # Compute the rotational dynamics
        quat_dot, omega_dot = self.rotational_dynamics(x, moment, Mres)

        # Return the derivative
        return jnp.concatenate((pos_dot, v_dot, quat_dot, omega_dot))

    def init_residual_networks(self):
        """Initialize the residual deep neural networks
        """
        # Create the residual MLP
        # The parameters are store in params dictionary under residual_forces
        # The residual MLP is a function of the state and the control
        if 'residual_forces' in self.params:
            _act_fn = self.params['residual_forces']['activation_fn']
            self.residual_forces = hk.nets.MLP([*self.params['residual_forces']['hidden_layers'], 3],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-0.001, 0.001),
                                                name = 'res_forces')

        # The residual MLP is a function of the state and the control
        if 'residual_moments' in self.params:
            _act_fn = self.params['residual_moments']['activation_fn']
            self.residual_moments = hk.nets.MLP([*self.params['residual_moments']['hidden_layers'], 3],
                                                activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                                w_init=hk.initializers.RandomUniform(-0.001, 0.001),
                                                name = 'res_moments')

        # Add ground effect here if a NN is used

    def prior_diffusion(self, x, u, extra_args=None):
        # Set the prior to a constant noise as defined in the yaml file
        return jnp.array(self.params['noise_prior_params'])
    

    def compositional_drift(self, x, u, extra_args=None):
        """Drift of the posterior dynamics
        """
        # We need to build the residual terms and ext_thrust terms
        # First we need to rotate the velocity vector in the body frame
        v_b = quat_rotatevectorinv(get_quat(x), get_vel(x))

        # Compute Fres and Mres and ground effect if it is taken into account
        Fres = self.compute_force_residual(x, v_b)
        Mres = self.compute_moment_residual(x, v_b)
        ge_effect = self.ground_effect(x)

        # Now we can compute the drift
        return self.vector_field(x, u, Fres, Mres, ge_effect)


############# SET OF FUNCTIONS TO TRAIN THE MODEL #############

def load_trajectory(log_dir, outlier_cond=lambda d : d['z']>0.1, min_length=500, mavg_dict={}):
    """Load the trajectories from the file
        Args:
            log_dir (str): Directory where the log file is stored
            outlier_cond (function): Function that returns True if the data point is an outlier
            min_length (int): Minimum length of the trajectory when splitting using outlier
        Returns: (as a tuple)
            x (list): List of ndarray of shape (N, 13) containing the states
            u (list): List of ndarray of shape (N, 4) or (N, 6) containing the controls
    """
    from sde4mbrlExamples.rotor_uav.utils import parse_ulog
    log_dir = os.path.expanduser(log_dir)
    # Load the data from the ULog
    log_data = parse_ulog(log_dir, outlier_cond=outlier_cond, min_length=min_length, mavg_dict=mavg_dict)

    # Ordered state names
    name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']
    name_controls = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

    # Extract the states and controls
    x = [ np.stack([_log_data[_name] for _name in name_states], axis=1) for _log_data in log_data]
    # Build the control action ndarray
    u = [ np.stack([_log_data[_name] for _name in name_controls if _name in _log_data], axis=1) for _log_data in log_data]
    # Return the data
    return (x, u)

def main_generate_trajectories(cfg_yaml_dir):
    """ Generate and save trajectories fron the configuration file cfg_yaml.
        When splitting, only the longest trajectory is taken
        Args:
            cfg_yaml_dir (str): Directory of the configuration file
    """
    cfg_yaml = load_yaml(cfg_yaml_dir)
    vehicle_dir = os.path.expanduser(cfg_yaml['vehicle_dir']) + '/my_data/' + cfg_yaml['outfile'] + '.pkl'

    # Condition for the outliers
    def _cond_fun(d):
        """ Parsing data criteritrain_trajectorya function """
        res = None
        for _state, (_min, _max) in cfg_yaml['criteria']['states'].items():
            _res = np.logical_and(d[_state] >= _min, d[_state] <= _max)
            if res is None:
                res = _res
            else:
                res = np.logical_and(res, _res)
        return res
    
    # Check if a dictionary with moving average parameters are given
    mavg_dict = cfg_yaml.get('mavg_dict', {})

    train_traj_list = []
    print('Loading train trajectories')
    for _data_path in tqdm(cfg_yaml['train_trajectory'], leave=False):
        (xlist, ulist) = load_trajectory(_data_path, outlier_cond = _cond_fun, min_length = cfg_yaml['criteria']['TRAJ_MIN_LEN'], mavg_dict=mavg_dict)
        # Check the size of the x value
        for _xtraj, _utraj in zip(xlist, ulist):
            train_traj_list.append((np.array(_xtraj), np.array(_utraj)))

    test_traj_list = []
    print('Loading test trajectories')
    for _data_path in tqdm(cfg_yaml['test_trajectory'], leave=False):
        (xlist, ulist) = load_trajectory(_data_path, outlier_cond = _cond_fun, min_length = cfg_yaml['criteria']['TRAJ_MIN_LEN'], mavg_dict=mavg_dict)
        # Check the size of the x value
        for _xtraj, _utraj in zip(xlist, ulist):
            test_traj_list.append((np.array(_xtraj), np.array(_utraj)))

    # Dump the file in a pkl
    print('Saving the data')
    with open(vehicle_dir, 'wb') as f:
        pickle.dump({'train': train_traj_list, 'test': test_traj_list}, f)


def load_predictor_function(learned_params_dir, prior_dist=False, nonoise=False, modified_params ={}, return_control=False, return_time_steps=False):
    """ Create a function to sample from the prior distribution or
        to sample from the posterior distribution
        Args:
            learned_params_dir (str): Directory where the learned parameters are stored
            prior_dist (bool): If True, the function will sample from the prior distribution
            nonoise (bool): If True, the function will return a function without diffusion term
            modified_params (dict): Dictionary of parameters to modify
        Returns:
            function: Function that can be used to sample from the prior or posterior distribution
    """
    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)

    # vehicle parameters
    _model_params = learned_params['nominal']

    # SDE learned parameters -> All information are saved using numpy array to facilicate portability
    # of jax accross different devices
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)

    # If prior distribution, set the diffusion to zero
    if prior_dist:
        params_model.pop('residual_forces', None)
        params_model.pop('residual_moments', None)
        params_model.pop('aero_drag_effect', None)
        # TODO: Remove ground effect if present
        # Remove the learned density function
        params_model.pop('diffusion_density_nn', None)

    # If no_noise
    if nonoise:
        params_model['noise_prior_params'] = [0] * len(params_model['noise_prior_params'])
    
    # Compute the timestep of the model the extract the time evolution starting t0 = 0
    time_steps = compute_timesteps(params_model)
    time_evol = np.array([0] + jnp.cumsum(time_steps).tolist())

    # Create the model
    _prior_params, m_sampling = create_sampling_fn(params_model, sde_constr=SDERotorModel)

    _sde_learned = _prior_params if prior_dist else _sde_learned
    if not return_time_steps:
        return lambda *x : m_sampling(_sde_learned, *x)[1] if not return_control else m_sampling(_sde_learned, *x)[1:]
    else:
        res_fn = lambda *x : m_sampling(_sde_learned, *x)[1] if not return_control else m_sampling(_sde_learned, *x)[1:]
        return (res_fn, time_evol)


def load_mpc_solver(mpc_config_dir, modified_params={}, nominal_model=False):
    """ Create an MPC solver that can be used at each time step for control
        Args:
            mpc_config_dir (str): Directory where the MPC configuration file is stored
            modified_params (dict): Dictionary of parameters to modify
            nominal_model (bool): If True, the MPC solver will use the nominal model
    """
    # Load the yaml configuration file
    _mpc_params = load_yaml(mpc_config_dir)

    # Get the path to the model parameters
    mpc_params = _mpc_params
    learned_params_dir = mpc_params['learned_model_params']

    # Load the pickle file
    with open(os.path.expanduser(learned_params_dir), 'rb') as f:
        learned_params = pickle.load(f)

    # vehicle parameters
    _model_params = learned_params['nominal']

    # SDE learned parameters
    _sde_learned = apply_fn_to_allleaf(jnp.array, np.ndarray, learned_params['sde'])

    # Load the sde model given the sde learned and the vector field utils
    # Update the parameters with a user-supplied dctionary of parameters
    params_model = update_params(_model_params, modified_params)

    # Update other parameters provided in the yaml file
    if 'model_update' in mpc_params:
        params_model = update_params(params_model, mpc_params['model_update'])
    
    if nominal_model:
        # Remove the learned density function
        params_model.pop('diffusion_density_nn', None)
        params_model['noise_prior_params'] = [0] * len(params_model['noise_prior_params'])

    # Create the function that defines the cost of MPC and integration
    # This function modifies params_model
    _, _multi_cost_sampling, vmapped_prox, _, construct_opt_params = create_online_cost_sampling_fn(params_model, mpc_params, sde_constr=SDERotorModel)

    # multi_cost_sampling =  lambda *x : _multi_cost_sampling(_sde_learned, *x)
    # Set the actual model params
    _mpc_params['model'] = params_model
    return (_sde_learned, _mpc_params), _multi_cost_sampling, vmapped_prox, construct_opt_params


def main_train_sde(yaml_cfg_file, output_file=None):
    """ Main function to train the SDE
    """
    # Load the yaml file
    cfg_train = load_yaml(yaml_cfg_file)
    # Extract the vehicle directory
    vehicle_dir = os.path.expanduser(cfg_train['vehicle_dir'])

    # Obtain the path to the ulog files
    logs_dir = cfg_train['data_dir']
    # Load the pkl file
    with open(vehicle_dir + '/my_data/' + logs_dir, 'rb') as f:
        data = pickle.load(f)
    full_data = [{ 'y' : x, 'u' : u} for x, u in data['train']]
    test_data = [{ 'y' : x, 'u' : u} for x, u in data['test']]
    # Print the maximum in absolute value of the full data
    print('Maximum in absolute value of each state of the full data')
    print(np.max(np.abs(np.concatenate([x['y'] for x in full_data])), axis=0))


    if 'data_state_scaling' in cfg_train['model']:
        cfg_train['model']['state_scaling'] = list(np.max(np.abs(np.concatenate([x['y'] for x in full_data])), axis=0))
        # Set the quaternion scaling to 1
        cfg_train['model']['state_scaling'][6:10] = 1.0
        print('State scaling is set to {}'.format(cfg_train['model']['state_scaling']))

    # Check if the control input match the model
    assert cfg_train['model']['n_u'] == full_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_u'] == test_data[0]['u'].shape[-1], 'The control input dimension does not match the model'
    assert cfg_train['model']['n_y'] == full_data[0]['y'].shape[-1], 'The state dimension does not match the model'
    assert cfg_train['model']['n_y'] == test_data[0]['y'].shape[-1], 'The state dimension does not match the model'

    # Here we make sure that init_params are saved in the model
    # which is then going to be saved on the disk for future use
    if 'learned_nominal' in cfg_train and cfg_train['learned_nominal'] is not None:
        # Load the learned nominal model
        learned_nominal = load_yaml(vehicle_dir + '/my_models/' + cfg_train['learned_nominal'])
        cfg_train['model']['init_params'] = learned_nominal['learned']
    else:
        cfg_train['model']['fixed_init_params'] = False

    # Need to expand the model by impossing penalty on deviation to the nominal model
    # This enforce that the learned special parameters in init_params should deviate minimally from these values when applying penalty
    cfg_train['sde_loss']['special_parameters_val'] =  cfg_train['model']['init_params']

    output_file = vehicle_dir + '/my_models/' + output_file

    # Train the model
    train_model(cfg_train, full_data, test_data, output_file, SDERotorModel)

if __name__ == '__main__':
    from sde4mbrl.train_sde import train_model

    import argparse
    from tqdm.auto import tqdm

    # Argument parser
    parser = argparse.ArgumentParser(description='Train the SDE model')
    parser.add_argument('--fun', type=str, default='gen_traj', help='Path to the yaml training configuration file')
    parser.add_argument('--cfg', type=str, default='iris_sitl/data_generation.yaml', help='Path to the yaml training configuration file')
    parser.add_argument('--out', type=str, default='iris_sitl', help='Path to the output file')
    # Parse the arguments
    args = parser.parse_args()
    # Check the function type
    if args.fun == 'gen_traj':
        # Generate the trajectories
        main_generate_trajectories(args.cfg)
    elif args.fun == 'train_sde':
        # Call the main function
        main_train_sde(args.cfg, args.out)
