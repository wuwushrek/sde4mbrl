import jax
import jax.numpy as jnp
import haiku as hk

# Store the SDE solvers and their corresponding names
from sde4mbrl.sde_solver import sde_solver_name

from sde4mbrl.utils import set_values_all_leaves, update_same_struct_dict
from sde4mbrl.utils import get_penalty_parameters, get_non_negative_params

# from sde4mbrl.utils import initialize_problem_constraints

import copy


def compute_timesteps(params):
    """ Compute the timesteps for the numerical integration of the SDE
    The timesteps are computed as follows:
        - params is a dictionary with that must contains the keys: horizon, stepsize; optional:  num_short_dt, short_step_dt, long_step_dt
        - horizon = total number of timesteps
        - stepsize = size of the timestep dt
        - The first num_short_dt timesteps are of size short_step_dt if given else stepsize
        - The last remaining timesteps are of size long_step_dt if given else stepsize
    
    Args:
        params (dict): The dictionary of parameters

    Returns:
        jnp.array: The array of timesteps
        
    """
    horizon = params['horizon']
    stepsize = params['stepsize']
    num_short_dt = params.get('num_short_dt', horizon)
    assert num_short_dt <= horizon, 'The number of short dt is greater than horizon'
    num_long_dt = horizon - num_short_dt
    short_step_dt = params.get('short_step_dt', stepsize)
    long_step_dt = params.get('long_step_dt', stepsize)
    return jnp.array([short_step_dt] * num_short_dt + [long_step_dt] * num_long_dt)


def initialize_problem_constraints(n_x, n_u, params_model):
    """Check if there are any constraints involved on the hidden states or the inputs
       of the sde model. The constraints are going to be enforced via nonsmooth optimization
       or a change of variable with smooth optimization and proximal projection

    Args:
        params_model (TYPE): The parameters of the model

    Returns:
        TYPE: Description
    """

    # n_u = params_model['n_u']
    # Check if some bounds on u are present
    has_ubound = 'input_constr' in params_model
    input_lb, input_ub = None, None
    if has_ubound:
        print('Found input bound constraints...\n')
        input_lb = [-jnp.inf for _ in range(n_u)]
        input_ub =[jnp.inf for _ in range(n_u)]
        input_dict = params_model['input_constr']
        assert len(input_dict['input_id']) <= n_u and \
                len(input_dict['input_id']) == len(input_dict['input_bound']),\
                "The number of constrained inputs identifier does not match the number of bounds"
        for idx, (u_lb, u_ub) in zip(input_dict['input_id'], input_dict['input_bound']):
            input_lb[idx], input_ub[idx] = u_lb, u_ub
        input_lb = jnp.array(input_lb)
        input_ub = jnp.array(input_ub)

    # Check if bounds on the state are present
    has_xbound = 'state_constr' in params_model
    weight_constr = 1
    slack_scaling = 1
    # By default we impose bounds constraint on the states using nonsmooth penalization
    slack_proximal, state_idx, penalty_coeff, state_lb, state_ub = False, None, None, None, None
    if has_xbound:
        weight_constr = params_model['state_constr'].get('constr_pen', 1)
        print('Found states bound constraints...\n')
        slack_dict = params_model['state_constr']
        assert len(slack_dict['state_id']) <= n_x and \
                len(slack_dict['state_id']) == len(slack_dict['state_bound']),\
                'The number of the constrained states identifier does not match the number of bounds'
        state_idx = jnp.array(slack_dict['state_id'])
        slack_proximal = slack_dict['slack_proximal']
        penalty_coeff = slack_dict['state_penalty']
        state_lb = jnp.array([x[0] for x in slack_dict['state_bound']])
        state_ub = jnp.array([x[1] for x in slack_dict['state_bound']])
        slack_scaling = jnp.array(params_model['state_constr'].get('slack_scaling', 1)) if slack_proximal else 1
        # Scale down state_lb and state_ub by slack_scaling
        state_lb = state_lb / slack_scaling
        state_ub = state_ub / slack_scaling

    return (has_ubound, input_lb, input_ub), \
            (has_xbound, slack_proximal, state_idx, penalty_coeff, state_lb, state_ub, weight_constr, slack_scaling)


class ControlledSDE(hk.Module):
    """Define an SDE object (stochastic dynamical system) with observation and state (latent) variable
    and which is controlled via a control input. 
    Typically \dot{x} = f(x, u) dt + sigma(x;u) dW, where the SDE is defined by the drift f and the diffusion sigma.
    The SDE could be in the Ito or Stratonovich sense. The choice of the sde solver will determine if the SDE is an Ito or Stratonovich SDE.

    This class implements several functions for deep NNs modeling and control of dynamical systems with conservative uncertainty estimation.
    The class is designed to be inherited by the user to define the SDE model and the control policy.
    
    The class will the following functionalities:
        (a) Train an SDE to fit data given a priori knowledge on the dynamical systems as inductive bias 
        (b) The learned SDE must be able to provide uncertainty estimates on what they don't know
        (c) We provide a way to control the SDE using a gradient-based MPC controller + NN policy + Value function
        (d) Sample from the learned SDE model

    A user should create a class that inherits the properties of this class while redefining the functions below:

        - [Required] compositional_drift : Define the drift term of the SDE model to learn.
            The compositional drift is a function of the state and the control input.
            We write it as compositional_drift(x,u) = f(x,u) =  F(x, u, g_1(x), g_2(x), ..., g_n(x)), 
            where g_i are the functions that define the drift and that are parametrized by NNs in our framework.
            This form on the drift informs the learning with prior knowledge on the dynamical system.
            Such prior knowledge is typically obtained as an ODE from physics or engineering. And with such prior knowledge,
            we sometimes can characterize the uncertainty (or region where the ODE is wrong) in the state space. (see next poiint)

        - [Required] prior_diffusion : Define a prior diffusion on the stochastic distribution (SDE) to learn. 
            This is consdered as a priori knowledge on where in the state space the above physics knowledge could be uncertain and wrong. 
            It characterizes the approximations and limited knowledge that our prior ode model relies on.
            If such a prior is not known, then the prior should be set to a constant noise value.
            Typically, our SDE learning algorithm will learn distributions whose uncertainty is smaller than the prior around regionns of the state space
            where we have data. And increasingly revert back to the prior "as we move away from the data".
        
        - [Optional] prior_constraints : Define the constraints that may be known on the unknown terms of the compositional drift.
            These constraints, if present, are going to be enforced during the learning process.
            And they are tipically in the form of a function h(g_1, g_2, ..., g_n) <= 0 for x in a given region of the state space stored in params_model['constraints']

        - [Optional] init_encoder : 
            Initialize the encoder function -> Provide a way to go from observation to state and its log probability
    """
    def __init__(self, params, name=None):
        """Initialize the parameterized sde prior and posterior dynamics

        Args:
            params (dict, optional): The set of params used when defining
                                     each of the parameters of the model.
                                     These parameters uniquely define a model and
                                     will usually be used to reset a model from a file
            name (None, optional): The name to prefix to the parameters description
        """
        # Need to be done for haiku initialization module
        super().__init__(name=name)

        # Save the parameters
        self.params = params

        # Some checks on the parameters
        assert 'n_x' in params and 'n_u' in params and 'n_y' in params, \
            'The number of state, control and observation dimensions must be specified in the params'
        assert 'horizon' in params, 'The prediction horizon must be specified in the params'
        assert 'stepsize' in params, 'The stepsize must be specified in the params'

        # Save some specific function parameters
        self.n_x = params['n_x'] # State dimension
        self.n_u = params['n_u'] # Control dimension
        self.n_y = params['n_y'] # Observation dimension

        # The prediction horizon
        self.horizon = params.get('horizon', 1)

        # Compute the time steps for the SDE solver
        self.time_step = compute_timesteps(params)

        # Initialize the encoder and decoder if they are defined
        self.init_encoder()

        # Construct the diffusion density
        self.construct_diffusion_density_nn()

        # Initialize the SDE solver
        self.sde_solve = sde_solver_name[params.get('sde_solver', 'euler_maruyama')]


    def prior_diffusion(self, x, u, extra_args=None):
        """ Define the prior noise function over the knowledge of the dynamics of the system
        The prior noise function is a function of the state and possibly the control input.
        This work assumes diagonal noise, but the user can define a more complex noise function

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A noise vector of the same size as the state vector (latent space)
        """
        raise NotImplementedError
    
    def compositional_drift(self, x, u, extra_args=None):
        """Define the drift term of the SDE model to learn.
        The compositional drift is a function of the state and the control input.
        We write it as compositional_drift(x,u) = f(x,u) =  F(x, u, g_1(x), g_2(x), ..., g_n(x)), 
        where g_i are the functions that define the drift and that are parametrized by NNs in our framework.
        This form on the drift informs the learning with prior knowledge on the dynamical system.
        Such prior knowledge is typically obtained as an ODE from physics or engineering.

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A vector of the same size as the state vector (latent space)
        """
        raise NotImplementedError

    def prior_constraints(self, x, u):
        """Define the constraints that may be known on the unknown terms of the compositional drift.
        These constraints, if present, are going to be enforced during the learning process.
        And they are tipically in the form of a function h(g_1, g_2, ..., g_n) <= 0 for x in a given region of the state space stored in params_model['constraints']

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A vector encoding the constraints
        """
        return None

    def reduced_state(self, _x):
        """ User-defined function to reduce the state / latent space for noise estimation
        """
        return _x
    
    def construct_diffusion_density_nn(self):
        """ Define the neural network that parametrizes the diffusion's density over the training dataset.
            The density could be a function of the state and the control input, or only of the state.
            The density is a scalar function that is used to compute the prior diffusion multiplier.
            Around the data, the density function would be close to zero or encloses the aleatoric uncertainty.
            And far from the data, the density function would be close to one and the noise would be close to the prior diffusion.
            density : \eta(x, u; \theta) or \eta(x, \theta) -> A real-valued function
            Total diffusion : \sigma(x, u) = [ \eta(x, u; \theta) * scaler(\theta) ] \sigma_{prior}(x, u)
            Few parameters' names:
                - diffusion_density_nn : The name of the key with the parameters of the density NN and scaler. If not present, the density and scaler is assumed to be 1
                - diffusion_density_nn -> indx_noise_in : The indexes of the states that contribute to the noise (if not given, all states contribute)
                - diffusion_density_nn -> indx_noise_out : The indexes of the outputs that contribute to the noise (if not given, all outputs contribute)
                
                - diffusion_density_nn -> scaler_nn : The name of the key with the parameters of the scaler NN. If not present, the scaler is assumed to be 1
                - diffusion_density_nn -> scaler_nn -> type : {nn, scaler} default is scaler
                - diffusion_density_nn -> scaler_nn -> init_value : The parameters are uniformly initialized in the range [-init_value, init_value]
                - diffusion_density_nn -> scaler_nn -> activation_fn, hidden_layers -> The parameters of the NN in case type is nn
                
                - diffusion_density_nn -> density_nn : The name of the key with the parameters of the density NN
                - diffusion_density_nn -> density_nn -> init_value : The parameters are uniformly initialized in the range [-init_value, init_value]
                - diffusion_density_nn -> density_nn -> activation_fn, hidden_layers -> The parameters of the NN

        """
        # Define the default eta and scaler functions if they are not gievn in the params
        self.scaler_fn = None
        self.diff_density = lambda _x, _u : 1.0

        if 'diffusion_density_nn' not in self.params:
            return

        # Extract the indexes of the states that contribute to the noise (if not given, all states contribute)
        self.noise_inputs = jnp.array(self.params['diffusion_density_nn'].get('indx_noise_in', jnp.arange(self.n_x)))
        
        # Extract the indexes of the outputs that contribute to the noise (if not given, all outputs contribute)
        self.noise_outputs = jnp.array(self.params['diffusion_density_nn'].get('indx_noise_out', jnp.arange(self.n_x)))

        # Define the function that concatenates the state and control input if needed for the density NN
        self.noise_relevant_state = lambda _x : self.reduced_state(_x)[self.noise_inputs] if self.noise_inputs.shape[0] < self.n_x else self.reduced_state(_x)
        self.noise_aug_state_ctrl = lambda _x, _u : jnp.concatenate([self.noise_relevant_state(_x), _u], axis=-1) if self.params.get('control_dependent_noise', False) else self.noise_relevant_state(_x)
        self.noise_aug_output = lambda _z : jnp.ones(self.n_x).at[self.noise_outputs].set(_z) if self.noise_outputs.shape[0] < self.n_x else _z
        
        # Lets design the scaler term first
        if 'scaler_nn' in self.params['diffusion_density_nn'] and self.params['diffusion_density_nn']['scaler_nn'].get('type', 'none') != 'none':
            scaler_type = self.params['diffusion_density_nn']['scaler_nn'].get('type', 'scaler')
            init_value = self.params['diffusion_density_nn']['scaler_nn'].get('init_value', 0.001)
            if scaler_type == 'scaler':
                # The choice of sigmoid is to make sure that the scaler is always between 0 and 1
                # The choice of +7.0 is to make sure that regularization (params close to 0) provides a value close to 1
                # self.diff_scaler = lambda _x, _u: self.noise_aug_output(jax.nn.sigmoid(hk.get_parameter('scaler', shape=self.noise_outputs.shape, 
                #                                     init = hk.initializers.RandomUniform(-init_value, init_value)) + 7.0 ))
                self.scaler_fn =  lambda _ : hk.get_parameter('scaler', shape=(self.noise_outputs.shape[0]*2,), 
                                                    init = hk.initializers.RandomUniform(-init_value, init_value))
            elif scaler_type == 'nn':
                _act_fn = self.params['diffusion_density_nn']['scaler_nn'].get('activation_fn', 'tanh')
                self.scaler_fn = hk.nets.MLP([*self.params['diffusion_density_nn']['scaler_nn']['hidden_layers'], self.noise_outputs.shape[0]*2],
                                    activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                    w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                    name = 'scaler')
                # self.diff_scaler = lambda _x, _u : self.noise_aug_output(jax.nn.sigmoid(self.scaler_net(self.noise_aug_state_ctrl(_x, _u)) + 7.0))
            else:
                raise ValueError('The scaler type should be either scaler or nn')
        
        # Now lets design the density term
        if 'density_nn' in self.params['diffusion_density_nn']:
            init_value = self.params['diffusion_density_nn']['density_nn'].get('init_value', 0.01)
            _act_fn = self.params['diffusion_density_nn']['density_nn'].get('activation_fn', 'tanh')
            self.density_net = hk.nets.MLP([*self.params['diffusion_density_nn']['density_nn']['hidden_layers'], 1],
                                    activation = getattr(jnp, _act_fn) if hasattr(jnp, _act_fn) else getattr(jax.nn, _act_fn),
                                    w_init=hk.initializers.RandomUniform(-init_value, init_value),
                                    name = 'density')
            # The choice of sigmoid is to make sure that the density is always between 0 and 1
            # The choice of -7.0 is to make sure that regularization (params close to 0) provides a value close to 0
            self.density_function = lambda xu_red: jax.nn.sigmoid((self.density_net(xu_red)[0] - 7.0))
            # Extract aggressiveness coefficient
            aggr = self.params['diffusion_density_nn']['density_nn'].get('aggressiveness', 1.0)

            # Create the diff_density function
            def __diff_density(_x, _u):
                _xu_red = self.noise_aug_state_ctrl(_x, _u)
                _density = self.density_net(_xu_red)[0]
                if self.scaler_fn is not None:
                    # 1e-7 is to make sure that the scaler is close to 0 when the density is close to 0
                    _scaler_val = jnp.exp(self.scaler_fn(_xu_red)) * 1e-7
                    scaler_coeff, scaler_offset = _scaler_val[:self.noise_outputs.shape[0]], _scaler_val[self.noise_outputs.shape[0]:]
                else:
                    scaler_coeff, scaler_offset = 0.0, 0.0
                return jax.nn.sigmoid((aggr + scaler_coeff) * _density - 7.0 + scaler_offset)
            
            self.diff_density = lambda _x, _u : self.noise_aug_output(__diff_density(_x, _u))
            self.density_function_v2 =  lambda _x, _u : self.density_net(self.noise_aug_state_ctrl(_x, _u))[0]
            self.density_function_v1 =  lambda _x, _u : self.density_function(self.noise_aug_state_ctrl(_x, _u))

    
    def diffusion(self, x, u, extra_args=None):
        """ This function defines the diffusion term of the SDE model to learn.
            It is a function of the prior diffusion, the density distribution of the latent space

        Args:
            x (TYPE): The current state of the system (aka latent state)
            u (TYPE, optional): The current control signal applied to the system
            extra_args (TYPE, optional): Extra arguments to pass to the function
        
        Returns:
            TYPE: A noise vector of the same size as the state vector (latent space)
        """
        return self.diff_density(x, u) * self.prior_diffusion(x, u, extra_args)
    

    def init_encoder(self):
        """ Define a function to encode from the current observation to the
            latent space. This function should be accessed by a call to self.obs2state. Besides, self.obs2state is a function with the following prototype
            obs2state and state2obs are functions that are already vectorized -> can take 1d or 2d arrays
            self.obs2state = lambda obs, randval: latent var
            self.state2obs = lambda latentvar, randval: obs
        """
        self.obs2state = self.identity_obs2state
        self.state2obs = self.identity_state2obs
    
    def normal_dist_compute(self, y, rng, mean_fn, noise_fn):
        """ Compute a Gaussian estimate given a mean and standard deviation function
        """
        x_mean = mean_fn(y)
        x_sigma = noise_fn(y)
        z_dist = jax.random.normal(rng, x_mean.shape)
        return x_mean + x_sigma * z_dist, x_mean, x_sigma
    

    def normal_obs2state(self, y, rng, mean_fn, noise_fn):
        """Return an estimated latent state based on a normal distribution
        with mean_fn and standard deviation given by noise_fn

        Args:
            y (TYPE): The current observation of the system
            rng (TYPE): A random number generator key
            mean_fn (TYPE): A function to compute the mean of a Gaussian process
            noise_fn (TYPE): A function to compute the var of a Gaussian process

        Returns:
            TYPE: The estimated (latent) state given the observation y
        """
        return self.normal_dist_compute(y, rng, mean_fn, noise_fn)[0]
    
    def normal_state2obs(self, x, rng, mean_fn, noise_fn):
        """Return an estimated observation based on a normal distribution
        with mean_fn and standard deviation given by noise_fn

        Args:
            x (TYPE): The current state of the system
            rng (TYPE): A random number generator key
            mean_fn (TYPE): A function to compute the mean of a Gaussian process
            noise_fn (TYPE): A function to compute the var of a Gaussian process

        Returns:
            TYPE: The estimated observation given the latent state x
        """
        return self.normal_dist_compute(x, rng, mean_fn, noise_fn)[0]

    def identity_obs2state(self, y, rng=None):
        """ Return the identity fuction when the observation is exactly the
            state of the system
        """
        return y
    
    def identity_state2obs(self, x, rng=None):
        """ Return the identity fuction when the observation is exactly the
            state of the system
        """
        return x
    
    def sample_sde(self, y0, uVal, rng, extra_scan_args=None):
        """Sample trajectory from the SDE distribution

        Args:
            y0 (TYPE): The observation of the system at time at initial time
            uVal (TYPE): The control signal applied to the system. This could be an array or a function of observation
            rng (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and the drift and diffusion functions
        """
        return self.sample_general((self.obs2state, self.state2obs), self.compositional_drift, self.diffusion, y0, uVal, rng, extra_scan_args)

    def sample_general(self, obs2state_fns, drift_term, diff_term, y0, uVal, rng_brownian, extra_scan_args=None):
        """A general function for sampling from a drift fn and a diffusion fn
        given times at which to sample solutions and control values to be applied

        Args:
            obs2state_fns (TYPE): A tuple of functions to convert observations to states and vice versa
            drift_term (TYPE): A function to compute the drift term of the SDE
            diff_term (TYPE): A function to compute the diffusion term of the SDE
            y0 (TYPE): The observation of the system at time at initial time
            uVal (TYPE): The control signal applied to the system. This could be an array or a function of observation
            rng_brownian (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and the drift and diffusion functions
        
        Returns:
            TYPE: A tuple of arrays containing the sampled states, observations and the control applied
        """

        # Finally solve the stochastic differential equation
        if hk.running_init():
            # Dummy return in this case -> This case is just to initialize NNs
            # Initialize the obs2state and state2obs parameters
            x0 = obs2state_fns[0](y0, rng_brownian)
            _ = obs2state_fns[1](x0, rng_brownian)
            # Initialize the drift and diffusion parameters
            _ = drift_term(x0, uVal if uVal.ndim ==1 else uVal[0], extra_scan_args)
            _ = diff_term(x0, uVal if uVal.ndim ==1 else uVal[0], extra_scan_args)

            #[TODO Franck] Maybe make the return type to be the same after sde_solve
            #Fine if the returned types are different at initilization
            # 2D array to be similar with sde_solve
            return jnp.zeros((self.params['horizon']+1, self.n_x)), jnp.zeros((self.params['horizon']+1, self.n_y)), jnp.zeros((self.params['horizon'], self.n_u))
        else:
            # Solve the sde and return its output (latent space)
            return self.sde_solve(obs2state_fns, self.time_step, y0, uVal, 
                        rng_brownian, drift_term, diff_term, 
                        projection_fn= self.projection_fn if hasattr(self, 'projection_fn') else None,
                        extra_scan_args=extra_scan_args
                    )
    
    def density_loss(self, y, u, rng, mu_coeff):
        """ Given an observation, control, and a random number generator key,
            This function computes the following terms that define the loss on the density function:
                1. The observation y is converted to a state x
                2. x and u are combined [x,u] then reduced to the relevant components as defined by noise_aug_state_ctrl
                3. The reduced state and control are passed to density_function to compute both the density and its gradient
                4. Now, params['density_loss'] contains 4 keys: ball_radius, mu_coeff, learn_mucoeff, ball_nsamples
                    a. ball_radius: The radius of the ball around the observation [y,u] where we sample points to enforce local density/ strong convexity
                    b. mu_coeff: The coefficient of the strong convexity term
                    c. learn_mucoeff: If True, then mu_coeff is a learnable parameter that will be optimized and initialize with the value in mu_coeff
                    d. ball_nsamples: The number of points to sample from the ball to enforce the strong convexity term
                5. We generate a ball of radius ball_radius around [y,u] and sample ball_nsamples points from it
                6. We compute the density at each of the sampled points
            
            Args:
                y (TYPE): The observation of the system
                u (TYPE): The control signal applied to the system
                rng (TYPE): A random number generator key

            Returns:
                TYPE: The gradient norm
                TYPE: the strong convexity loss
        """

        # Split the random number generator into a key for observation to state conversion and a key for sampling the ball
        rng_obs2state, rng_ball = jax.random.split(rng)

        # Convert the observation to a state
        x = self.obs2state(y, rng_obs2state)

        # Combine the state and the control
        xu_red = self.noise_aug_state_ctrl(x, u)

        # Compute the density and its gradient
        den_xu, grad_den_xu = jax.value_and_grad(self.density_function)(xu_red)

        # Check if the ball_radius is an array or a scalar
        # [TODO] Make the radius to be a proportion of the magnitude of the state
        radius = jnp.array(self.params['density_loss']['ball_radius'])
        if radius.ndim > 0:
            assert radius.shape == xu_red.shape, "The ball_radius should be a scalar or an array of size xu_red.shape[0]"

        # Sample ball_nsamples points from the ball of radius ball_radius around xu_red
        ball_dist = jax.random.normal(rng_ball, (self.params['density_loss']['ball_nsamples'], xu_red.shape[0])) * radius[None]
        xball = xu_red[None] + ball_dist

        # Compute the density at each of the sampled points
        den_xball = jax.vmap(self.density_function)(xball)

        # Compute the strong convexity loss given mu_coeff
        sconvex_cond = den_xball - den_xu - jnp.sum(grad_den_xu[None] * ball_dist, axis=1) - 0.5 * mu_coeff * jnp.sum(jnp.square(ball_dist), axis=1)
        # sconvex_cond = den_xball - den_xu - 0.5 * mu_coeff * jnp.sum(jnp.square(ball_dist), axis=1)
        
        sconvex_loss = jnp.sum(jnp.square(jnp.minimum(sconvex_cond, 0)))
        # sconvex_loss = jnp.sum(jnp.square(sconvex_cond)) # [TODO Franck] Check if this is better than summing

        return jnp.sum(jnp.square(grad_den_xu)), sconvex_loss, den_xu


    def sample_for_loss_computation(self, ymeas, uVal, rng, extra_scan_args=None):
        """ Sample the SDE and compute the loss between the estimated and the measured observation

        Args:
            ymeas (TYPE): The measured observation of the system
            uVal (TYPE): The control signal applied to the system. This is an array
            rng (TYPE): A random number generator key
            extra_scan_args (None, optional): Extra arguments to be passed to the scan function and the drift and diffusion functions
        
        Returns:
            TYPE: The estimated latent x0 at y0 = ymeas[0]
            TYPE: The error between the estimated and the measured observation
            TYPE: A dictionary containing extra information about the loss
        """

        # Check if the trajectory horizon matches
        assert ymeas.shape[0] == uVal.shape[0]+1, 'Trajectory horizon must match'
        num_steps2data = self.params['num_steps2data']
        assert  self.params['horizon'] * num_steps2data == uVal.shape[0], 'Trajectory horizon must match'

        # Split the random number generator
        rng_brownian, rng_sample2consider, rng_density = jax.random.split(rng, 3)

        # uval is size (horizon * num_steps2data, num_ctrl), lets reshape it to (horizon, num_steps2data, num_ctrl)
        u_values = uVal.reshape((self.params['horizon'], num_steps2data, self.params['n_u']))
        # Let's get the actual y_values
        y_values = ymeas[::num_steps2data]
        # How do we pick u_values? Different strategies stored in params['u_sampling_strategy']
        # By default we pick the first control value, i.e. u_values[:,0,:]
        # Another strategie is the mean of all the control values, i.e. u_values.mean(axis=1)
        # Another strategie is the median of all the control values, i.e. jnp.median(u_values, axis=1)
        # Another strategie is a random control value
        u_sampling_strategy = self.params.get('u_sampling_strategy', 'first')
        if u_sampling_strategy == 'first':
            u_values = u_values[:,0,:]
        elif u_sampling_strategy == 'mean':
            u_values = u_values.mean(axis=1)
        elif u_sampling_strategy == 'median':
            u_values = jnp.median(u_values, axis=1)
        elif u_sampling_strategy == 'random':
            rng_density, rng_u = jax.random.split(rng_density)
            rnd_indx = jax.random.randint(rng_u, shape=(self.params['horizon'],), minval=0, maxval=num_steps2data)
            u_values = u_values[jnp.arange(self.params['horizon']), rnd_indx, :]
        else:
            raise ValueError('Unknown u_sampling_strategy: {}. Choose from first, mean, median, random'.format(u_sampling_strategy))

        # Solve the SDE to obtain state and observation evolution
        y0 = y_values[0]
        _, ynext, _ = self.sample_sde(y0, u_values, rng_brownian, extra_scan_args)

        # Extract the state (ignore the initial state) and the KL divergence
        ynext, meas_y = ynext[1:], y_values[1:]

        # Get indexes of the samples to consider in the norm 2 computation between the estimated and the measured observation
        # This essentially make the fitting similar to a time series with irregular sampling
        _indx = None
        if 'num_sample2consider' in self.params:
            # rng_sample2consider = jnp.array([0, 0], dtype=jnp.uint32)
            if self.params['num_sample2consider'] < ynext.shape[0]:
                _indx = jax.random.choice(rng_sample2consider, ynext.shape[0], 
                            shape=(self.params['num_sample2consider'],), replace=False)
                # Restrain the time indexes to be considered
                ynext = ynext[_indx]
                meas_y = meas_y[_indx]

        # Compute the error between the estimated and the measured observation
        # We sum the error over the integration horizon and then average over the number of coordinates
        # _error_data = jnp.mean(jnp.sum(jnp.square(meas_y-ynext), axis=0))
        _error_data = jnp.sum(jnp.square((meas_y-ynext) / jnp.array(self.params.get('obs_weights', 1.0))))

        # Extra values to print or to penalize the loss function on
        extra = {}

        # Check if density_loss is in the parameters
        if 'density_loss' not in self.params:
            # Error on prediction, Gradient error, strong convexity error, and extra values
            return _error_data, 0.0, 0.0, 0.0, extra

        # Function to get the mu_coeff whether it is learnable or not
        learn_mu = self.params['density_loss']['learn_mucoeff']
        mu_coeff = self.params['density_loss']['mu_coeff']
        get_mu_coeff = lambda:  mu_coeff if not learn_mu else jnp.exp(hk.get_parameter('mu_coeff', (), init= hk.initializers.Constant(jnp.log(mu_coeff)) ))
        
        # Check if haiku is running initialization
        if hk.running_init():
            # Initialize the extra parameters if present
            mu_coeff = get_mu_coeff()
            # Error on prediction, Gradient error, strong convexity error, mu_coeff, and extra values
            return _error_data, 0.0, 0.0, mu_coeff, extra

        # Here haiku is not running initialization
        # We get the mu_coeff if it is learnable
        mu_coeff = get_mu_coeff()
        rng_density = jax.random.split(rng_density, ymeas.shape[0]-1)
        
        den_yinput = ymeas[:-1]
        den_uinput = uVal
        if _indx is not None and self.params.get('density_on_partial', False):
            den_yinput = y_values[:-1][_indx]
            den_uinput = u_values[_indx]
            rng_density = rng_density[_indx]

        # Get the gradient and the convex loss
        grad_norm, sconvex, _density_val = jax.vmap(lambda _y, _u, _rng: self.density_loss(_y, _u, _rng, mu_coeff))(den_yinput, den_uinput, rng_density)
        extra['density_val'] = jnp.mean(_density_val)
        
        # Error on prediction, Gradient error, strong convexity error, mu_coeff, and extra values
        return _error_data, jnp.mean(grad_norm), jnp.sum(sconvex), mu_coeff, extra


    def sample_dynamics_with_cost(self, y, u, rng, cost_fn, slack_index=None, extra_dyn_args=None, extra_cost_args=None):
        """Generate a function that integrate the sde dynamics augmented with a cost
           function evolution (which is also integrated along the dynamics).
           The cost function is generally the cost we want to minimize
           in a typically Nonlinear MPC problem

        Args:
            y (TYPE): The initial observation of the system
            u (TYPE): The sequence of control signal applied to the system. This is an array
            rng (TYPE): A random number generator key
            cost_fn (TYPE): The cost function to be integrated along the dynamics
            slack_index (None, optional): The index of the slack variable in the control signal
            extra_dyn_args (None, optional): Extra arguments to be passed to the drift and diffusion functions
            extra_cost_args (None, optional): Extra arguments to be passed to the cost function

        """
        # Define the augmented dynamics
        def cost_aug_drift(_aug_x, _u, extra_args=None):
            _x = _aug_x[:-1]
            actual_u = _u[:self.n_u]
            drift_fn = self.compositional_drift
            drift_pos_x = drift_fn(_x, actual_u, None if extra_dyn_args is None else extra_args[0])
            slack_t = _u[self.n_u:] if _u.shape[0] > self.n_u else None
            cost_term = cost_fn(_x, actual_u, slack_t, None if extra_cost_args is None else extra_args[-1])
            return jnp.concatenate((drift_pos_x, jnp.array([cost_term])))

        def cost_aug_diff(_aug_x, _u, extra_args=None):
            _x = _aug_x[:-1]
            diff_x = self.diffusion(_x, _u, None if extra_dyn_args is None else extra_args[0])
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Define the augmented obs2state and state2obs functions
        # The last element of the augmented state is the cost and is not affect by obs2state and state2obs
        _temp_obs2state = lambda _y, _rng: jnp.concatenate((self.obs2state(_y[:-1], _rng), _y[-1:]))
        _temp_state2obs = lambda _x, _rng: jnp.concatenate((self.state2obs(_x[:-1], _rng), _x[-1:]))

        # Solve the augmented integration problem
        aug_y = jnp.concatenate((y, jnp.array([0.])) )

        # TODO: This is a hack because of setting the slacj value to the corresponding state value
        # Transform y to state
        xval = self.obs2state(y, rng)

        # The slack variable should be initialized to the corresponding state values
        # The slack variable are shifted -> The slack corresponding to the first time step is actually the last
        # Because control time step and state time step are shifted
        if slack_index is not None and u.shape[1] > self.n_u:
            u = u.at[0,self.n_u:].set(xval[slack_index])
        
        extra_scan_args = None if (extra_dyn_args is None and extra_cost_args is None) else \
                            ( (extra_dyn_args, extra_cost_args) if extra_dyn_args is not None and extra_cost_args is not None else \
                                    ( (extra_dyn_args,) if extra_dyn_args is not None else (extra_cost_args,)
                                    )
                            )
        
        _, _yevol, _ = self.sample_general((_temp_obs2state, _temp_state2obs), cost_aug_drift, cost_aug_diff, aug_y, u, rng, extra_scan_args)
        # return self.sample_general(cost_aug_drift, cost_aug_diff, aug_x, u, rng_brownian, extra_scan_args)
        return _yevol


def create_obs2state_fn(params_model, sde_constr=ControlledSDE, seed=0,
                        **extra_args_sde_constr):
    """ Return a function for estimating the hidden state given an observation of the system.
    The function is a wrapper around the SDE class

    Args:
        params_model (dict): The parameters of the model
        sde_constr (TYPE, optional): The SDE class to use
        seed (int, optional): The seed for the random number generator
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE class

    Returns:
        dict: A dictionary containing the initial parameter models
        function: The function that estimate the state given an observation
                    The function takes as input the some hk model parameters, observation and a random key
                    and returns the estimated state
                    obs2state_fn(params: dict, obs: ndarray, rng: ndarray) -> state : ndarray
    """
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    def _obs2state(y0, rng):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.obs2state(y0, rng)
    # Transform the function into a pure one
    obs2state_pure =  hk.without_apply_rng(hk.transform(_obs2state))
    nn_params = obs2state_pure.init(rng_zero, yzero, rng_zero)
    return nn_params, obs2state_pure.apply

def create_diffusion_fn(params_model, sde_constr=ControlledSDE, seed=0,
                        **extra_args_sde_constr):
    """ Return a function for estimating the diffusion term given an observation of the system.
    The function is a wrapper around the SDE class

    Args:
        params_model (dict): The parameters of the model
        sde_constr (TYPE, optional): The SDE class to use
        seed (int, optional): The seed for the random number generator
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE class
        
    Returns:
        dict: A dictionary containing the initial parameter to compute the total diffusion and density diffusion
        function: The function that estimate the diffusion vector given an observation
                    The function takes as input the some hk model parameters, observation, control and a random key
                    and returns the estimated total diffusion term and the density diffusion term
                    diffusion_fn(params: dict, obs: ndarray, u: ndarray, rng: ndarray, extra_args) -> diffusion, density : ndarray, scalar
    """
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    uzero = jnp.zeros((params_model['n_u'],))

    def _diffusion(y0, u0, rng, net=False, extra_args=None):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        # TODO: Maybe having this function independent of the encoder structure
        # Because the density function is going to be depending on the neural network structure of the encoder
        # Which might be quite actually quite good as it imposes a structure on the encoder too
        x0 = m_model.obs2state(y0, rng)
        if not net:
            return m_model.diffusion(x0, u0, extra_args), m_model.density_function_v1(x0, u0)
        else:
            return m_model.diffusion(x0, u0, extra_args), m_model.density_function_v2(x0, u0)
    
    # Transform the function into a pure one
    diffusion_pure =  hk.without_apply_rng(hk.transform(_diffusion))
    nn_params = diffusion_pure.init(rng_zero, yzero, uzero, rng_zero)
    return nn_params, diffusion_pure.apply

def create_one_step_sampling(params_model, sde_constr= ControlledSDE, seed=0, 
                            num_samples=None, **extra_args_sde_constr):
    """Create a function that sample the next state

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): The seed for the random number generator
        num_samples (int, optional): The number of samples to generate
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        function: a function for one-step multi-sampling
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the next state or a number of particles of the next state
                    one_step_sampling(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: None or named args) -> next_x, next_y, u : ndarray
    """
    params_model = copy.deepcopy(params_model) # { k : v for k, v in params_model.items()}
    params_model['horizon'] = 1 # Set the horizon to 1

    # We remove num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss
    params_model.pop('num_short_dt', None)
    params_model.pop('short_step_dt', None)
    params_model.pop('long_step_dt', None)

    # Get the number of samples
    num_samples = params_model['num_particles'] if num_samples is None else num_samples

    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    uzero = jnp.zeros((params_model['n_u'],))

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        """ Sampling function """
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_sde(y, u, rng, extra_args)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    _ = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_samples)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        xevol, yevol, uevol =  vmap_sampling(_nn_params, y, u, m_rng, extra_args)
        return (xevol[0], yevol[0], uevol[0]) if num_samples == 1 else (xevol, yevol, uevol)

    return multi_sampling

def create_sampling_fn(params_model, sde_constr= ControlledSDE, 
                       seed=0, num_samples=None, **extra_args_sde_constr):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class. it specifies the SDE model
        seed (int, optional): The seed for the random number generator
        num_samples (int, optional): The number of samples to generate
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        dict: A dictionary containing the initial parameter models
        function: a function for multi-sampling on the posterior or prior
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the next state or a number of particles of the next state
                    sampling_fn(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: nor or named args) -> next_x, next_y, u : ndarray
    """
    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    # TODO: This function is almost the same as create_one_step_sampling. We should merge them
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    uzero = jnp.zeros((params_model['n_u'],))
    num_samples = params_model['num_particles'] if num_samples is None else num_samples

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        """ Sampling function """
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_sde(y, u, rng, extra_args)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_samples)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        return vmap_sampling(_nn_params, y, u, m_rng, extra_args)

    return nn_params, multi_sampling

def create_model_loss_fn(model_params, loss_params, sde_constr=ControlledSDE, verbose=True, 
                        **extra_args_sde_constr):
    """Create a loss function for evaluating the current model with respect to some
       pre-specified dataset

    Args:
        model_params (TYPE): The SDE model and solver parameters
        loss_params (TYPE): The pamaters used in the loss function computation. 
                            Typically penalty coefficient for the different losses.
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        dict : A dictionary containing the initial parameters for the loss computation
        function: A function for computing the loss
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the loss value and a dictionary of the different losses
                    loss_fn(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: nor or named args) -> loss : float, losses: dict
        function: A function for projecting the special nonnegative parameters of the model
                    The function takes as input the some hk model parameters and returns the projected parameters
                    project_fn(params: dict) -> params: dict
        function: A function for sampling the learned model
    """

    # Verbose print function
    vprint = print if verbose else lambda *a, **k: None

    # Deep copy params_model
    params_model = model_params

    # The number of samples is given by the loss dictionary -> If not present, use the default value from the params-model
    num_sample = loss_params.get('num_particles', params_model.get('num_particles', 1) )
    params_model['num_particles'] = num_sample

    # The step size is given by the loss dictionary -> If not present, use the default value from the params-model
    step_size = loss_params.get('stepsize', params_model['stepsize'] )
    params_model['stepsize'] = step_size

    # Now let's get the data stepsize from the loss dictionary -> If not present, use the default value is the step_size
    data_stepsize = loss_params.get('data_stepsize', step_size)
    # Let's check if the stepsize is a multiple of the data_stepsize
    if abs (step_size - data_stepsize) <= 1e-6:
        num_steps2data = 1
    else:
        assert abs(step_size % data_stepsize) <= 1e-6, 'The data stepsize must be a multiple of the stepsize'
        # Let's get the number of steps between data points
        num_steps2data = int((step_size/data_stepsize) +0.5) # Hack to avoid numerical issues
    # Let's get the horizon of the loss
    horizon = loss_params.get('horizon', params_model.get('horizon', 1))
    # Let's get the actual actual horizon of the loss
    data_horizon = horizon * num_steps2data
    # Let's set the horizon in the params_model
    params_model['horizon'] = horizon
    params_model['num_steps2data'] = num_steps2data
    loss_params['data_horizon'] = data_horizon
    # Print the number of particles used for the loss
    vprint('Using [ N = {} ] particles for the loss'.format(num_sample))
    # Print the horizon used for the loss
    vprint('Using [ T = {} ] horizon for the loss'.format(params_model['horizon']))
    # Print the stepsize used for the loss
    vprint('Using [ dt = {} ] stepsize for the loss'.format(params_model['stepsize']))
    # Print the number of steps between data points
    vprint('Using [ num_steps2data = {} ] steps between data points'.format(num_steps2data))

    # Extract the number of 
    if 'num_sample2consider' in loss_params:
        params_model['num_sample2consider'] = loss_params['num_sample2consider']
    
    if 'obs_weights' in loss_params:
        params_model['obs_weights'] = loss_params['obs_weights']
    
    if 'density_on_partial' in loss_params:
        params_model['density_on_partial'] = loss_params['density_on_partial']
    
    if 'u_sampling_strategy' in loss_params:
        params_model['u_sampling_strategy'] = loss_params['u_sampling_strategy']

    # We remove num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss
    params_model.pop('num_short_dt', None)
    params_model.pop('short_step_dt', None)
    params_model.pop('long_step_dt', None)
    vprint ('Removed num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss and sde training')

    # Now check if the diffusion is parameterized by a neural network
    if 'diffusion_density_nn' not in params_model or 'density_nn' not in params_model['diffusion_density_nn']:
        loss_params.pop('density_loss', None)

    # Now we insert the density loss parameters if they are not present
    if 'density_loss' in loss_params:
        params_model['density_loss'] = loss_params['density_loss']
        if not params_model['density_loss']['learn_mucoeff']:
            loss_params['pen_mu_coeff'] = 0.0
    else:
        loss_params['pen_mu_coeff'] = 0.0
        loss_params['pen_density_scvex'] = 0.0
        loss_params['pen_grad_density'] = 0.0

    # Define the transform for the sampling function
    def sample_loss(y, u, rng, extra_args=None):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_for_loss_computation(y, u, rng, extra_args)
    
    rng_zero = jax.random.PRNGKey(loss_params.get('seed', 0))
    yzero = jnp.zeros((data_horizon+1,params_model['n_y']))
    uzero = jnp.zeros((data_horizon,params_model['n_u'],))

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_loss))
    nn_params = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Let's get nominal parameters values
    nominal_params_val = loss_params.get('nominal_parameters_val', {})
    default_params_val = loss_params.get('default_parameters_val', 0.) # This value imposes that the parameters should be minimized to 0
    _nominal_params_val = set_values_all_leaves(nn_params, default_params_val)
    nominal_params = _nominal_params_val if len(nominal_params_val) == 0 else update_same_struct_dict(_nominal_params_val, nominal_params_val)
    special_params_val = loss_params.get('special_parameters_val', {})
    nominal_params = get_penalty_parameters(nominal_params, special_params_val, None)
    # Print the resulting penalty coefficients
    vprint('Nominal parameters values: \n {}'.format(nominal_params))

    # Let's get the penalty coefficients for regularization
    special_parameters = loss_params.get('special_parameters_pen', {})
    default_weights = loss_params.get('default_weights', 1.)
    penalty_coeffs = get_penalty_parameters(nn_params, special_parameters, default_weights)
    # Print the resulting penalty coefficients
    vprint('Penalty coefficients: \n {}'.format(penalty_coeffs))

    # Nonnegative parameters of the problem
    nonneg_params = get_non_negative_params(nn_params, {k : True for k in params_model.get('noneg_params', []) })
    # Print the resulting nonnegative parameters
    vprint('Nonnegative parameters: \n {}'.format(nonneg_params))

    # Define a projection function for the parameters
    def nonneg_projection(_params):
        return jax.tree_map(lambda x, nonp : jnp.maximum(x, 0.0) if nonp else x, _params, nonneg_params)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        return vmap_sampling(_nn_params, y, u, m_rng, extra_args)

    def loss_fn(_nn_params, y, u, rng, extra_args=None):
        # CHeck if rng is given as  a ingle key
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'

        # Split the rng key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling, in_axes=(None, 0, 0, 0, 0) if extra_args is not None else (None, 0, 0, 0, None))
        _error_data, grad_density, density_scvex, mu_coeff, extra_feat = batch_vmap(_nn_params, y, u, rng, extra_args)

        # COmpute the loss on fitting the trajectories
        loss_data = jnp.mean(jnp.mean(_error_data, axis=1))

        # Compute the loss on the gradient of the density
        loss_grad_density = jnp.mean(jnp.mean(grad_density, axis=1))

        # Compute the loss on the density local strong convexity
        loss_density_scvex = jnp.mean(jnp.mean(density_scvex, axis=1))

        # Compute the loss on the mu coefficient
        loss_mu_coeff = jnp.mean(jnp.mean(mu_coeff, axis=1)) # This is probably not needed

        # Weights penalization
        w_loss_arr = jnp.array( [jnp.sum(jnp.square(p - p_n)) * p_coeff \
                            for p, p_n, p_coeff in zip(jax.tree_util.tree_leaves(_nn_params), jax.tree_util.tree_leaves(nominal_params), jax.tree_util.tree_leaves(penalty_coeffs)) ]
                        )
        w_loss = jnp.sum(w_loss_arr)

        # Extra feature mean if there is any
        m_res = { k: jnp.mean(jnp.mean(v, axis=1)) for k, v in extra_feat.items()}

        # Compute the total sum
        total_sum = 0.0
        if loss_params.get('pen_data', 0) > 0:
            total_sum += loss_data * loss_params['pen_data']

        if loss_params.get('pen_grad_density', 0) > 0:
            p_mucoeff = loss_params['density_loss']['mu_coeff']
            eff_pen = loss_params['pen_grad_density'] * p_mucoeff * loss_params['density_loss'].get('grad_scaler', 1.0/p_mucoeff)
            total_sum += eff_pen * loss_grad_density 
        
        if loss_params.get('pen_density_scvex', 0) > 0:
            total_sum += loss_params['pen_density_scvex'] * loss_density_scvex
        
        if loss_params.get('pen_mu_coeff', 0) > 0:
            # We seek to maximize the mu coefficient
            total_sum -= loss_params['pen_mu_coeff'] * loss_mu_coeff

        if loss_params.get('pen_weights', 0) > 0:
            total_sum += loss_params['pen_weights'] * w_loss

        return total_sum, {'totalLoss' : total_sum, 'gradDensity' : loss_grad_density,
                            'densitySCVEX' : loss_density_scvex, 'weights' : w_loss, 'muCoeff' : loss_mu_coeff
                            , 'dataLoss' : loss_data, **m_res}
    

    # Now define a function that will be used for testing the current model parameters
    _, multi_sampling_sde = create_sampling_fn(params_model, sde_constr, loss_params.get('seed', 0), 
                                                loss_params['num_particles_test'], **extra_args_sde_constr)
    
    def test_fn(_nn_params, y, u, rng, extra_args=None):
        """ Given measurements y and control inputs u, as batched data, and a random number generator,
            Compute the error in predicting the trajectories and the variance of the prediction

        Args:
            _nn_params (TYPE): The parameters of the model
            y (TYPE): The measurements given as a batch of trajectories of the right horizon
            u (TYPE): The control inputs given as a batch of trajectories of the right horizon
            rng (TYPE): A random number generator
            extra_args (None, optional): Extra arguments to pass to the sampling function
        """
        # CHeck if rng is given as  a ingle key
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'

        # uval is size (horizon * num_steps2data, num_ctrl), lets reshape it to (horizon, num_steps2data, num_ctrl)
        u_values = u.reshape((u.shape[0], params_model['horizon'], num_steps2data, params_model['n_u']))
        # Let's get the actual y_values
        y_values = y[:,::num_steps2data,:]
        # How do we pick u_values? Different strategies stored in params['u_sampling_strategy']
        # By default we pick the first control value, i.e. u_values[:,0,:]
        # Another strategie is the mean of all the control values, i.e. u_values.mean(axis=1)
        # Another strategie is the median of all the control values, i.e. jnp.median(u_values, axis=1)
        # Another strategie is a random control value
        u_sampling_strategy = params_model.get('u_sampling_strategy', 'first')
        if u_sampling_strategy == 'first':
            u_values = u_values[:,:,0,:]
        elif u_sampling_strategy == 'mean':
            u_values = u_values.mean(axis=2)
        elif u_sampling_strategy == 'median':
            u_values = jnp.median(u_values, axis=2)
        elif u_sampling_strategy == 'random':
            rng, rng_u = jax.random.split(rng)
            rnd_indx = jax.random.randint(rng_u, shape=(params_model['horizon'],), minval=0, maxval=num_steps2data)
            # Now we have to do a fancy indexing, taking into account the batch size on first dimension
            u_values = u_values[:, jnp.arange(params_model['horizon']), rnd_indx, :]

        else:
            raise ValueError('Unknown u_sampling_strategy: {}. Choose from first, mean, median, random'.format(u_sampling_strategy))

        # Split the rng key first
        rng = jax.random.split(rng, y.shape[0])

        # Re-assign y and u
        y = y_values
        u = u_values
        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling_sde, in_axes=(None, 0, 0, 0, 0) if extra_args is not None else (None, 0, 0, 0, None))
        _, yevol, _ = batch_vmap(_nn_params, y[:, 0, :], u, rng, extra_args)

        # Compute the mean of the prediction
        yevol_mean = jnp.mean(yevol, axis=1)

        # Compute the error in the prediction wrt y
        error_data = jnp.mean(jnp.sum(jnp.sum(jnp.square((y - yevol_mean) / jnp.array(loss_params.get('obs_weights', 1.0))), axis=-1), axis=-1))

        # Compute the standard deviation of the prediction
        yevol_std = jnp.std(yevol, axis=1)
        std_val = jnp.mean(jnp.sum(jnp.sum(yevol_std, axis=-1), axis=-1))

        return error_data, {'TestErrorData' : error_data, 'TestStdData' : std_val}

    
    return nn_params, loss_fn, nonneg_projection, test_fn


def create_online_cost_sampling_fn(params_model,
                            params_mpc,
                            sde_constr= ControlledSDE,
                            seed=0,
                            **extra_args_sde_constr):
    """Create a function that integrate the dynamics as well as a cost function to minimize.
       Typically, the cost function is the objective used in the underlying MPC problem

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        cost_fn (TYPE): The cost function to optimize on the fly in a stochastic MPC manner
        terminal_cost (None, optional): The terminal cost, if available
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: Description
    """
    # Random ky for initialization
    rng_zero = jax.random.PRNGKey(seed)

    # Number of control inputs
    n_u = params_model['n_u']

    # Initialization of the observation and uzero
    yzero = jnp.zeros((params_model['n_y'],))

    # A zero-cost function for initializating the cost sampling function
    _cost_fn_zero = lambda _x, _u, _extra_args : jnp.array(0.)

    num_sample = params_mpc.get('num_particles', params_model.get('num_particles', 1) )
    params_model['num_particles'] = num_sample
    # Print the number of particles used for the loss
    print('Using [ N = {} ] particles for the loss'.format(num_sample))

    # Get the horizon from params_loss
    params_model['horizon'] = params_mpc.get('horizon', params_model.get('horizon', 1))
    # Print the horizon used for the loss
    print('Using [ T = {} ] horizon for the loss'.format(params_model['horizon']))

    # Let set up the number of short time steps
    params_model['num_short_dt'] = params_mpc['num_short_dt']
    params_model['short_step_dt'] = params_mpc['short_step_dt']
    params_model['long_step_dt'] = params_mpc['long_step_dt']

    # Define the time steps used during integration
    time_steps = compute_timesteps(params_model)
    # params_model['time_steps'] = time_steps

    (has_ubound, input_lb, input_ub),\
        (has_xbound, slack_proximal, state_idx, penalty_coeff, state_lb, state_ub, weight_constr, slack_scaling) = \
            initialize_problem_constraints(params_model['n_x'], params_model['n_u'], params_mpc)

    # Transform the penalty coefficient into an array
    penalty_coeff = jnp.array(penalty_coeff) if penalty_coeff is not None else penalty_coeff

    # Define the cost of penalizing the constraints using a nonsmooth function and penalty method
    def constr_cost_noprox(x_true, slack_x=None):
        """ Penalty method with nonsmooth cost fuction.
            This should be avoided when doing nonlinear MPC using accelerated gradient descent
        """
        x = x_true[state_idx]
        # diff_x should always be less than 0
        diff_x = jnp.concatenate((x - state_ub, state_lb - x))
        _penalty_coeff = jnp.concatenate((penalty_coeff, penalty_coeff))
        #[TODO Franck] Maybe mean/sum the error instead of doing a mean over states
        return jnp.sum(jnp.where( diff_x > 0, 1., 0.) * jnp.square(diff_x) * _penalty_coeff) * weight_constr

    # The cost of penalizing the constraints using a smooth function and proximal method
    def constr_cost_withprox(x_true, slack_x):
        """ With proximal constraint on the slack variable -> smooth norm 2 regularization
        """
        diff_state = x_true[state_idx] - slack_x * slack_scaling
        return jnp.sum(jnp.square(diff_state) * penalty_coeff) * weight_constr

    # A function to constraint a vector between a given minimum and maximum values
    constr_vect = lambda a, a_lb, a_ub:  jnp.minimum(jnp.maximum(a, a_lb), a_ub)

    # Now ready to define the constraint cost as well as the proximal operator if needed
    constr_cost = None
    proximal_fn = None

    # Check if the problem has a state as slack variables
    has_slack = has_xbound and slack_proximal
    opt_params_size = n_u  + ( len(state_idx) if has_slack else 0)

    # In case bound on u are given but no bound on x -> No constraint cost
    # We known that the parameters of the opt is only based on control
    # So we constraint such parameters as lowered by input_lb and uppered by input_ub
    if has_ubound and (not has_xbound):
        proximal_fn = lambda u_slack: constr_vect(u_slack, input_lb, input_ub)

    # No bound on u are given but x is constrained and we are using proximal operator
    # To constrain the slack variable associated with these states
    if (not has_ubound) and has_xbound and slack_proximal:
        constr_cost = constr_cost_withprox
        proximal_fn = lambda u_slack: jnp.concatenate((u_slack[:n_u], constr_vect(u_slack[n_u:], state_lb, state_ub)))

    # No bound on u is given but x is constrained. However, no slack variables
    # for proximal computation is given -> Nonsmooth penalization
    # IN this case, there is not proximal operator
    if (not has_ubound) and has_xbound and (not slack_proximal):
        constr_cost = constr_cost_noprox

    # We have a bound on u and bounds on x, but the bounds on x are enforced
    # as nonsmooth soft penalty cost
    if has_ubound and has_xbound and (not slack_proximal):
        constr_cost = constr_cost_noprox
        proximal_fn = lambda u_slack: constr_vect(u_slack, input_lb, input_ub)

    if has_ubound and has_xbound and slack_proximal:
        constr_cost = constr_cost_withprox
        u_slack_lb = jnp.concatenate((input_lb, state_lb))
        u_slack_ub = jnp.concatenate((input_ub, state_ub))
        proximal_fn = lambda u_slack: constr_vect(u_slack, u_slack_lb, u_slack_ub)

    # Define the augmented cost with the penalization term
    def aug_cost_fn(_x, _u, _slack, _cost_fn, extra_args=None):
        # Compute the actual cost
        actual_cost = _cost_fn(_x, _u, extra_args)
        if constr_cost is None:
            return actual_cost
        # Compute the constraints cost
        pen_cost = constr_cost(_x, _slack)
        return actual_cost + pen_cost

    # Define the function to integrate the cost function and the dynamics
    def sample_sde(y, opt_params, rng, _cost_fn, _terminal_cost=None, extra_dyn_args=None, extra_cost_args=None):

        # Do some check
        assert opt_params.ndim == 2, 'The parameters must be a two dimension array'
        if has_slack:
            assert opt_params.shape[1] == opt_params_size, 'Shape of the opt params do not match'
        else:
            assert opt_params.shape[1] == n_u, 'Shape of the opt params do not match'

        # Build the SDE solver
        m_model = sde_constr(params_model, **extra_args_sde_constr)

        # Compute the evolution of the state
        _aug_cost_fn = lambda x, u, slack, _extra_args: aug_cost_fn(x, u, slack, _cost_fn, _extra_args)
        x_evol = m_model.sample_dynamics_with_cost(y, opt_params, rng, _aug_cost_fn, state_idx,
                            extra_dyn_args=extra_dyn_args, extra_cost_args=extra_cost_args)

        # Evaluate the cost_to_go function
        # [TODO Franck] probably a bug here when using ts.shape-1 optimization variable
        # When has_slack should have ts.shape opt variable so that the slack
        # matches the end state/final state
        end_cost = _terminal_cost(x_evol[-1,:-1]) if _terminal_cost is not None else jnp.array(0.)
        if constr_cost is not  None:
            pen_cost = constr_cost(x_evol[-1,:-1], opt_params[0, n_u:] if has_slack else None)
        else:
            pen_cost = jnp.array(0.)
        # Compute the total cost by adding the terminal cost
        total_cost = pen_cost*time_steps[-1] + params_mpc['discount']*end_cost + x_evol[-1,-1]

        # Modified the cost to add the penalty with respect to constraints of the first state
        # x0_constr = 0. if not has_xbound else constr_cost_noprox(x_evol[0,:-1]) * time_steps[0]
        if has_xbound:
            # Initial constraint cost -> This should be constant as the constr cost is only a function of the state
            x_evol = x_evol.at[1:,-1].add(constr_cost_noprox(x_evol[0,:-1]) * time_steps[0])

        return total_cost, x_evol

    # Initialize an optimization parameters given a sequence of STATE x and u
    # [TODO Fanck] Define it in terms of observation and use the state to observation transformation to match the output
    # However the call of this with an observation will only happen once...
    def _construct_opt_params(u=None, x=None):
        # This ensures that if there is a slack variable
        # the constraint are also enforced on the terminal state
        # num_var = params_model['horizon'] + (0 if has_slack else 1)
        # TODO: Make a more general initialization that avoid infeasible control values
        num_var = params_model['horizon']
        # Non-zero initialization for gradient descent to work properly for the control inputs
        zero_u = jnp.ones((num_var, n_u)) * 1e-4
        # Non-zero initialization for gradient descent to work properly for the slack variables
        zero_x = jnp.ones((num_var, len(state_idx))) * 1e-4 if has_slack else None
        slack_scaling_vect = jnp.ones((num_var, len(state_idx))) * (slack_scaling if slack_scaling.ndim < 1 else slack_scaling[None] ) if has_slack else None

        if u is not None and u.ndim == 1:
            u = jnp.array([ u for _ in range(num_var)]) + 1e-4 # THis is just so that the parameters are not zero
        if x is not None and x.ndim == 1:
            x = jnp.array([ x for _ in range(num_var+1)])
        if x is not None: # We replace the first component of the state with the last component
            x = x.at[0,:].set(x[-1,:])

        if u is None and not has_slack:
            return zero_u # jnp.zeros((num_var, n_u))

        if u is None and x is None: # slack is true
            return jnp.concatenate((zero_u, zero_x), axis=1)# jnp.zeros((num_var, opt_params_size))

        if u is None and x is not None: # slack is true
            assert x.ndim == 2 and x.shape[0] == num_var+1
            return jnp.concatenate((zero_u, x[:-1,state_idx]/slack_scaling_vect), axis=1) + 1e-4

        if u is not None and not has_slack:
            assert u.ndim == 2 and u.shape[0] == num_var
            return u + 1e-4

        if u is not None and x is None: # has slack is true
            assert u.ndim == 2 and u.shape[0] == num_var
            return jnp.concatenate((u, zero_x), axis=1) + 1e-4

        if u is not None and x is not None: # has slack is true
            assert u.ndim == 2 and x.ndim == 2 and u.shape[0]+1 == x.shape[0]
            return jnp.concatenate((u, x[:-1,state_idx]/slack_scaling_vect), axis=1) + 1e-4

        assert False, 'This case is not handle...'

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, yzero, _construct_opt_params(), rng_zero, _cost_fn_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, opt_params, rng, _cost_fn, _terminal_cost=None, extra_dyn_args=None, extra_cost_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None, None, None, None))
        total_loss, xtraj = vmap_sampling(_nn_params, y, opt_params, m_rng, _cost_fn, _terminal_cost, extra_dyn_args, extra_cost_args)
        return jnp.mean(total_loss), xtraj

    # Vmapped the function such that it works
    # Properly define the proximal_function
    vmapped_prox = None if proximal_fn is None else jax.vmap(proximal_fn)

    if vmapped_prox is not None:
        construct_opt_params = lambda u=None, x=None: vmapped_prox(_construct_opt_params(u,x))
    else:
        construct_opt_params = _construct_opt_params

    return nn_params, multi_sampling, vmapped_prox, constr_cost, construct_opt_params


# Utility class for value and policy functions learning
class ValuePolicy(hk.Module):
    """ Define functions for value and policy parameterization and evaluation
    """
    def __init__(self, params, name=None):
        super().__init__(name)
        # Store the parameters
        self.params = params

        # Initialize the value
        #  function
        self.value_nn_init()

        # Initialize the policy function
        self.policy_nn_init()

        # Check if functions for the value and policy evaluation are defined
        assert hasattr(self, 'value_fn'), 'Value function not defined'
        assert hasattr(self, 'policy_fn'), 'Policy function not defined'

def create_value_n_policy_fn(model_params, value_policy_constr, seed=0):
    """ Create the value and policy functions and initialize its parameter dictionary
    """
    # Initialization values for ValuePolicy
    _x0 = jnp.zeros((model_params['n_x'],))

    # Initialze rng key
    rng_key = jax.random.PRNGKey(seed)

    # Transform the function into a pure one
    val_pure = hk.without_apply_rng(hk.transform(lambda x : value_policy_constr(model_params).value_fn(x)))
    pol_pure = hk.without_apply_rng(hk.transform(lambda x : value_policy_constr(model_params).policy_fn(x)))

    # Initialize the parameters
    val_params = val_pure.init(rng_key, _x0)
    pol_params = pol_pure.init(rng_key, _x0)

    # Return the functions and their params
    return (val_params, val_pure.apply), (pol_params, pol_pure.apply)


def create_valueNpolicy_loss_fn(model_params, loss_params, value_policy_constr, seed=0):
    """ Create a function to compute the loss for the value and policy functions
    """
    (val_params, val_pure_fn), (pol_params, pol_pure_fn) = create_value_n_policy_fn(model_params, value_policy_constr, seed)

    # Merged parameters
    valpol_params = {'value': val_params, 'policy': pol_params}
    valpol_pure = lambda _params, x : (val_pure_fn(_params['value'], x), pol_pure_fn(_params['policy'], x))
    
    # Define the loss function
    def loss_fn(_params, x, u, target_value, opt_params, cost_opt=jnp.array(0.0)):
        """ Compute the loss function for the value and policy functions
        """
        # Compute the value function
        Vx, Pix = jax.vmap(valpol_pure, in_axes=(None, 0))(_params, x)

        # Compute the gradient of Vx
        gradVx = jax.vmap(jax.grad(lambda _x : val_pure_fn(_params['value'], _x)), in_axes=0)(x)
        gradVx_norm = jnp.mean(jnp.square(gradVx))

        # Error on the value function
        ErrVx = jnp.mean(jnp.square(Vx - target_value))
        # print(u.shape, Pix.shape, x.shape, Vx.shape, target_value.shape)

        # Error on the policy function
        ErrPix = jnp.mean(jnp.square(Pix - u))

        # Regularization parameters
        w_loss_policy = jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(_params['policy'])]))
        w_loss_value = jnp.sum(jnp.array([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(_params['value'])]))

        # Regularization with respect to the optimal parameters
        # w_loss_opt = jnp.array( [jnp.sum(jnp.square(p-q)) \
        #                     for p, q in zip(jax.tree_util.tree_leaves(_params), jax.tree_util.tree_leaves(opt_params)) ]
        #             )
        w_loss_opt_val = jnp.array( [jnp.sum(jnp.square(p-q)) \
                            for p, q in zip(jax.tree_util.tree_leaves(_params['value']), jax.tree_util.tree_leaves(opt_params['value'])) ]
                    )
        w_opt_val = jnp.sum(w_loss_opt_val) # jnp.sum(w_loss_opt)

        w_loss_opt_pol = jnp.array( [jnp.sum(jnp.square(p-q)) \
                            for p, q in zip(jax.tree_util.tree_leaves(_params['policy']), jax.tree_util.tree_leaves(opt_params['policy'])) ]
                    )
        w_opt_pol = jnp.sum(w_loss_opt_pol) # jnp.sum(w_loss_opt)
        
        # Compute the total loss
        total_error = loss_params['value_loss'] * ErrVx + loss_params['policy_loss'] * ErrPix 

        # + loss_params['policy_reg'] * w_loss_policy + loss_params['value_reg'] * w_loss_value
        if 'policy_reg' in loss_params:
            total_error += loss_params['policy_reg'] * w_loss_policy
        
        if 'value_reg' in loss_params:
            total_error += loss_params['value_reg'] * w_loss_value

        if 'gradVx_reg' in loss_params:
            total_error += loss_params['gradVx_reg'] * gradVx_norm
        
        if 'opt_val_dev' in loss_params:
            total_error += loss_params['opt_val_dev'] * w_opt_val * cost_opt
        
        if 'opt_pol_dev' in loss_params:
            total_error += loss_params['opt_pol_dev'] * w_opt_pol * cost_opt

        # if weight_opt is not None:
        #     weight_opt_val, weight_opt_pol = weight_opt
        #     total_error += weight_opt_val * w_opt_val
        #     total_error += weight_opt_pol * w_opt_pol

        return total_error, {'totalLoss': total_error, 'Vx' : jnp.mean(Vx), 'Target' : jnp.mean(target_value), 'valueLoss': ErrVx, 'policyLoss': ErrPix, 
            'policyReg': w_loss_policy, 'valueReg': w_loss_value, 'w_opt_val' : w_opt_val, 'w_opt_pol' : w_opt_pol, 'GradVx' : gradVx_norm}

    return valpol_params, loss_fn, (pol_pure_fn, val_pure_fn)