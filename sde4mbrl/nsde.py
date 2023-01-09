import jax
import jax.numpy as jnp
import haiku as hk

from sde4mbrl.sde_solver import sde_solver_name
from sde4mbrl.utils import initialize_problem_constraints

from sde4mbrl.utils import set_values_all_leaves, update_same_struct_dict, get_penalty_parameters, get_non_negative_params
import copy

# [TODO] Batching over t for loss computationwith fixed time step
# [TODO] Initialization when observation dimension different of state dimension


def compute_timesteps(params):
    horizon = params['horizon']
    # The stepsize of the numerical integration scheme
    stepsize = params['stepsize']
    # Get the short and long time steps for control purpose or long term integration
    num_short_dt = params.get('num_short_dt', horizon)
    assert num_short_dt <= horizon, 'The number of short dt is greater than horizon'
    num_long_dt = horizon - num_short_dt
    short_step_dt = params.get('short_step_dt', stepsize)
    long_step_dt = params.get('long_step_dt', stepsize)
    return jnp.array([short_step_dt] * num_short_dt + [long_step_dt] * num_long_dt)

class ControlledSDE(hk.Module):
    """Define an SDE object (stochastic dynamical system) with latent variable
       and which is controlled via a control input.
       Typically \dot{x} = (f(t,x) + G(t,x) u) dt + sigma(t,x) dW, where the noise
       is considered as a stratonovich noise (HJB theory doesn't change much)

       This class implements several functions to facilitate (a) Train an SDE to fit data
       given a prior as another SDE; (b) Train a cost-to-go/Value function as well as the sde
       for online control in real-time.

       A user should create a class that inherits the properties of this class
       while redefining the functions below:
            - prior_diffusion : Define the prior diffusion function (can be parameterized by NN)
            - prior_drift : Define the prior drift function (can be parameterized by NN)
            - posterior_drift : Define the posterior drift function (can be parameterized)
            - (or prior_error and posterior_is_prior_plus_error) if the posterior is prior + an error function
            - init_encoder : Initialize the encoder function -> Provide a way to go from observation to state and its log probability
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

        # Save some specific function parameters
        self.n_x, self.n_u, self.n_y = params['n_x'], params['n_u'], params['n_y']

        # The prediction horizon
        self.horizon = params.get('horizon', 1)
        self.time_step = compute_timesteps(params)

        # Initialize the encode
        self.init_encoder()

        # Initialize the SDE solver
        self.sde_solve = sde_solver_name[params.get('sde_solver', 'stratonovich_milstein')]

        # Initalize the way the posterior drift is computed with respect to prior
        self.init_error_prior_drift()

        if self.posterior_is_prior_plus_error:
            self.posterior_drift = lambda t, x, u: self.prior_error(t,x,u) + self.prior_drift(t,x,u)

        assert (self.posterior_is_prior_plus_error and hasattr(self, 'prior_error')) or \
            (not self.posterior_is_prior_plus_error and not hasattr(self, 'prior_error')),\
            'posterior_is_prior_plus_error should match an existing function prior_error'


    def init_error_prior_drift(self):
        """ Essentially set the variable posterior_is_prior_plus_error
            if that variable is True, then a function named prior_error must be
            defined, which returns a value that add up to the prior to obtain posterior drif
            self.prior_error  = lambda t,x,u ->
        """
        self.posterior_is_prior_plus_error = False

    def prior_diffusion(self, x, extra_args=None):
        """Save the prior diffusion function with the attribute name 'prior_diffusion'
        Access to the prior functions is via self.prior_diffusion, which is a function
        of the time and latent state of the system

        Args:
            t (TYPE): The current time
            x (TYPE): The current state of the system (can be latent state)
        """
        pass

    def prior_drift(self, x, u, extra_args=None):
        """Save the prior drift function with the attribute name 'prior_drift'
        Access to the prior functions is via self.prior_drift, which is a function
        of the time and latent state of the system

        Args:
            t (TYPE): The current time
            x (TYPE): The current state of the system (can be latent state)
            u (TYPE): The current control signal applied to the system
        """
        pass

    def posterior_drift(self, x, u, extra_args=None):
        """Save the posterior drift function with the attribute name 'posterior_drift'
        Access to the posterior drift function is via self.posterior_drift, which is a function
        of the time and latent state of the system

        Args:
            t (TYPE): The current time
            x (TYPE): The current state of the system (can be latent state)
            u (TYPE): The current control signal applied to the system
        """
        pass

    def init_encoder(self):
        """ Define a function to encode from the current observation to the
            latent space. This function should be accessed by a call to
            self.obs2state. Besides, self.obs2state is a function with the
            following prototype
            self.obs2state = lambda obs, randval: latent var
            self.logprob(obs| z) = lambda obs, z: real value
        """
        self.obs2state = self.identity_obs2state

        # Initialize the log probability function
        self.logprob = self.identity_logprob

    def normal_logprob(self, y, x, rng, mean_fn, noise_fn):
        """Logprobability function of the Gaussian distribution

        Args:
            y (TYPE): The observation of the system
            x (TYPE): The current state of the system
            rng (TYPE): A random number generator key
            mean_fn (TYPE): A function to compute the mean of a Gaussian process
            noise_fn (TYPE): A function to compute the var of a Gaussian process

        Returns:
            TYPE: The logprobability of observing y knowning x (or inverse)
        """
        x_est, _, x_var = self.normal_dist_compute(y, rng, mean_fn, noise_fn)
        return -jnp.sum(jnp.log(x_var)) - 0.5 * jnp.sum(jnp.square((x_est-x)/x_var))

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

    def normal_dist_compute(self, y, rng, mean_fn, noise_fn):
        """ Compute a Gaussian estimate given a mean and standard deviation function
        """
        x_mean = mean_fn(y)
        x_sigma = noise_fn(y)
        z_dist = jax.random.normal(rng, x_mean.shape)
        return x_mean + x_sigma * z_dist, x_mean, x_sigma

    def identity_logprob(self, y, x, rng=None):
        """ Logprobability function for deterministic transformation
            between the states and the observation
        """
        # TODO: Add a discount factor for accounting for future
        # Maximize the log probablity
        return -jnp.sum(jnp.square(y-x))

    def identity_obs2state(self, y, rng=None):
        """ Return the identity fuction when the observation is exactly the
            state of the system
        """
        return y


    def sample_prior(self, y0, uVal, rng, extra_scan_args=None):
        """Sample trajectory from the prior distribution

        Args:
            ts (TYPE): The time indexes at which to evaluate the solution of the SDE
            y0 (TYPE): The observation of the system at time ts[0]
            uVal (TYPE): An array of the control signal to applied between ts[0] and ts[-1]
                         If a 1D array is given, we apply a constant control during the
                         integration horizon. if a 2D array is give, the first axis should
                         match the dimension of the time index ts. Then, a linear interpolation
                         is done to provide time-varying control signal during the integration horizon
            rng (TYPE): A random key generator
            params_solver (None, optional): SDE solver parameters as requested by
                                            differentiable_sde_solver
        """
        rng_obs2state, rng_brownian = jax.random.split(rng)
        x0 =  self.obs2state(y0, rng_obs2state)
        return self.sample_general(self.prior_drift, self.prior_diffusion,
                                    x0, uVal, rng_brownian, extra_scan_args)


    def sample_posterior(self, y0, uVal, rng, extra_scan_args=None):
        """Sample trajectory from the posterior distribution

        Args:
            ts (TYPE): The time indexes at which to evaluate the solution of the SDE
            y0 (TYPE): The observation of the system at time ts[0]
            uVal (TYPE): An array of the control signal to applied between ts[0] and ts[-1]
                         If a 1D array is given, we apply a constant control during the
                         integration horizon. if a 2D array is give, the first axis should
                         match the dimension of the time index ts. Then, a linear interpolation
                         is done to provide time-varying control signal during the integration horizon
            rng (TYPE): A random key generator
            params_solver (None, optional): SDE solver parameters as requested by
                                            differentiable_sde_solver
        """
        rng_obs2state, rng_brownian = jax.random.split(rng)
        x0 =  self.obs2state(y0, rng_obs2state)
        return self.sample_general(self.posterior_drift, self.prior_diffusion,
                                    x0, uVal, rng_brownian, extra_scan_args)


    def sample_general(self, drift_term, diff_term, x0, uVal, rng_brownian, extra_scan_args=None):
        """A general function for sampling from a drift fn and a diffusion fn
        given times at which to sample solutions and control values to be applied

        Args:
            drift_term (TYPE): A function representing the drift
                               drift_term : t, s, u : array_like(s)
            diff_term (TYPE): A function representing the diffusion (WeakDiagonal setting)
                               diff_term : t, s : array_like(s)
            ts (TYPE): The time indexes at which to evaluate the solution of the SDE
            x0 (TYPE): The latent state of the system at time ts[0]
            uVal (TYPE): An array of the control signal to applied between ts[0] and ts[-1]
                         If a 1D array is given, we apply a constant control during the
                         integration horizon. if a 2D array is give, the first axis should
                         match the dimension of the time index ts. Then, a linear interpolation
                         is done to provide time-varying control signal during the integration horizon
            rng_brownian (TYPE): A random key generator
            params_solver (None, optional): SDE solver parameters as requested by
                                            differentiable_sde_solver

        Returns:
            TYPE: The solution of the SDE at time indexes ts
        """

        # Finally solve the stochastic differential equation
        if hk.running_init():
            # Dummy return in this case -> This case is just to initialize NNs
            # Initialize the drift and diffusion parameters
            _ = drift_term(x0, uVal if uVal.ndim ==1 else uVal[0], extra_scan_args)
            _ = diff_term(x0, extra_scan_args)

            #[TODO Franck] Maybe make the return type to be the same after sde_solve
            #Fine if the returned types are different at initilization
            return jnp.zeros_like(x0)[None] # 2D array to be similar with sde_solve
        else:
            # Solve the sde and return its output (latent space)
            return self.sde_solve(self.time_step, x0, uVal, 
                        rng_brownian, drift_term, diff_term, 
                        projection_fn= self.projection_fn if hasattr(self, 'projection_fn') else None,
                        extra_scan_args=extra_scan_args
                    )

    def sample_for_loss_computation(self, meas_y, uVal, rng, extra_scan_args=None):
        """Compute the log-probability of the measured path at time indexes and
            given the control inputs uVal. This also estimates the KL-divergence
            in order to properly fit the SDE via the ELBO bound.

            This function is typically used for fitting data to SDEs

        Args:
            ts (TYPE): The time indexes for integration
            meas_y (TYPE): The measured observations at the different time indexes
            uVal (TYPE): The control inputs applied at the different time indexes
            rng (TYPE): The random key generator for brownian noise

        Returns:
            TYPE: logprob, kl_divergence, some extra computation as a dictionary
        """
        # Check if the trajectory horizon matches
        assert meas_y.shape[0] == uVal.shape[0]+1, 'Trajectory horizon must match'

        # Define the augmented dynamics or the ELBO bound
        def augmented_drift(_aug_x, _u, extra_args):
            """ Define the augmented drift function """

            # This is state + elbo -> so we keep only the state information
            _x = _aug_x[:-1]
            # Compute the prior drift term
            drift_pri_x = self.prior_drift(_x, _u, extra_args)

            # COmpute the error between prior and posterior
            if self.posterior_is_prior_plus_error:
                error_prior = self.prior_error(_x, _u, extra_args)
                drift_pos_x = drift_pri_x + error_prior
            else:
                drift_pos_x = self.posterior_drift(_x, _u, extra_args)
                error_prior = drift_pos_x - drift_pri_x
            
            # Compute the diffusion term
            diff_x = self.prior_diffusion(_x, extra_args) # A vector

            # Check if the diffusion is zero, then this is an ODE and we remove the KL div term
            if self.params['diffusion_type'] == 'zero' or self.params['diffusion_type'] == 'nonoise':
                return jnp.concatenate((drift_pos_x, jnp.array([0.0])))

            # Now let's check for the zeros in the diffusion -> These are given as a config by user
            if 'ignore_diff_indx' in self.params:
                indx = jnp.array(self.params['ignore_diff_indx'])
                cTerm = 0.5 * jnp.sum(jnp.square((error_prior[indx])/diff_x[indx]))
            else:
                cTerm = 0.5 * jnp.sum(jnp.square((error_prior)/diff_x))

            return jnp.concatenate((drift_pos_x, jnp.array([cTerm])))

        def augmented_diffusion(_aug_x, extra_args):
            """ Define the augmented diffusion function """
            _x = _aug_x[:-1]
            diff_x = self.prior_diffusion(_x, extra_args)
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Now convert the observation to a state and then integrate it
        rng_logprob, rng_obs2state, rng_brownian = jax.random.split(rng, 3)
        x0 =  self.obs2state(meas_y[0], rng_obs2state)

        # Solve the augmented integration problem
        aug_x0 = jnp.concatenate((x0, jnp.array([0.])) )
        est_x = self.sample_general(augmented_drift, augmented_diffusion, aug_x0, uVal, rng_brownian, extra_scan_args)

        # [TODO Franck] Average the KL divergence over each point of the traj
        # might not be a good idea _> To investigate !!! Maye the sum !!!!
        xnext, kl_div = est_x[:, :-1], jnp.sum(est_x[:, -1])

        # Compute the logprob(obs | state) for the estimated state
        # as a function of the observation
        # The state with no diffusion terms are not used for computing the logprob since this assumes the model is correct
        # [TODO Franck] Maybe we should use the state with no diffusion terms for computing the logprob
        if 'ignore_diff_indx' in self.params:
            indx = jnp.array(self.params['ignore_diff_indx'])
            plogVal = self.logprob(meas_y[1:,indx], xnext[1:,indx], rng_logprob)
        else:
            plogVal = self.logprob(meas_y[1:], xnext[1:], rng_logprob)

        # Extra values to print or to penalize the loss function on
        extra = {}

        return plogVal, kl_div, extra

    def sample_dynamics_with_cost(self, y, u, rng, cost_fn, slack_index=None, extra_dyn_args=None, 
                extra_cost_args=None, prior_sampling=False):
        """Generate a function that integrate the sde dynamics augmented with a cost
           function evolution (which is also integrated along the dynamics).
           The cost function is generally the cost we want to minimize
           in a typically Nonlinear MPC problem

        Args:
            ts (TYPE): The time indexes at which the integration happens
            y (TYPE): The current observation of the system
            u (TYPE): The sequence of control to applied at each time step (can be merged with slack)
            rng (TYPE): A random key generator for Brownian noise
            cost_fn (TYPE): The cost function to integrate
                            cost_fn : (_x, _u, _slack)

        Returns:
            TYPE: The state's evolution as well as cost evolution along the trajectory
        """

        # Define the augmented dynamics
        def cost_aug_drift(_aug_x, _u, extra_args=None):
            _x = _aug_x[:-1]
            actual_u = _u[:self.n_u]
            drift_fn = self.prior_drift if prior_sampling else self.posterior_drift
            drift_pos_x = drift_fn(_x, actual_u, None if extra_dyn_args is None else extra_args[0])
            slack_t = _u[self.n_u:] if _u.shape[0] > self.n_u else None
            # [TODO Franck] Maybe add time dependency ?
            cost_term = cost_fn(_x, actual_u, slack_t, None if extra_cost_args is None else extra_args[-1])
            return jnp.concatenate((drift_pos_x, jnp.array([cost_term])))

        def cost_aug_diff(_aug_x, extra_args=None):
            _x = _aug_x[:-1]
            diff_x = self.prior_diffusion(_x, None if extra_dyn_args is None else extra_args[0])
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Now convert the observation to a state and then integrate it
        x =  self.obs2state(y, rng)
        rng, rng_brownian = jax.random.split(rng)

        # Solve the augmented integration problem
        aug_x = jnp.concatenate((x, jnp.array([0.])) )

        # The slack variable should be initialized to the corresponding state values
        # The slack variable are shifted -> The slack corresponding to the first time step is actually the last
        # Because control time step and state time step are shifted
        if slack_index is not None and u.shape[1] > self.n_u:
            u = u.at[0,self.n_u:].set(x[slack_index])
        
        extra_scan_args = None if (extra_dyn_args is None and extra_cost_args is None) else \
                            ( (extra_dyn_args, extra_cost_args) if extra_dyn_args is not None and extra_cost_args is not None else \
                                    ( (extra_dyn_args,) if extra_dyn_args is not None else (extra_cost_args,)
                                    )
                            )
        return self.sample_general(cost_aug_drift, cost_aug_diff, aug_x, u, rng_brownian, extra_scan_args)

def create_obs2state_fn(params_model, sde_constr=ControlledSDE, seed=0,
                        **extra_args_sde_constr):
    """ Return a function for estimating the state given an observation of the system
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
                    one_step_sampling(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: None or named args) -> next_states : ndarray
    """
    params_model = copy.deepcopy(params_model) # { k : v for k, v in params_model.items()}
    params_model['horizon'] = 1
    # params_model['num_short_dt'] = 1
    # params_model['short_step_dt'] = params_model['stepsize']

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
        return m_model.sample_posterior(y, u, rng, extra_args)[-1]

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    _ = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_samples)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        res_val =  vmap_sampling(_nn_params, y, u, m_rng, extra_args)
        return res_val[0] if num_samples == 1 else res_val

    return multi_sampling

def create_sampling_fn(params_model, sde_constr= ControlledSDE, 
                        prior_sampling=True, seed=0, num_samples=None,
                        **extra_args_sde_constr):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class. it specifies the SDE model
        prior_sampling (bool, optional): If True, the sampling is done from the prior distribution. If False, the sampling is done from the posterior distribution
        seed (int, optional): The seed for the random number generator
        num_samples (int, optional): The number of samples to generate
        **extra_args_sde_constr: Extra arguments for the constructor of the SDE solver

    Returns:
        dict: A dictionary containing the initial parameter models
        function: a function for multi-sampling on the posterior or prior
                    The function takes as input the some hk model parameters, observation, control and a random key, and possibly extra arguments for drift and diffusion
                    and returns the next state or a number of particles of the next state
                    sampling_fn(params: dict, y: ndarray, u: ndarray, rng: ndarray, extra_args: nor or named args) -> next_states : ndarray
    """
    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((params_model['n_y'],))
    uzero = jnp.zeros((params_model['n_u'],))
    num_samples = params_model['num_particles'] if num_samples is None else num_samples

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        """ Sampling function """
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        m_sampling = m_model.sample_prior if prior_sampling else m_model.sample_posterior
        return m_sampling(y, u, rng, extra_args)

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

def create_model_loss_fn(model_params, loss_params, sde_constr=ControlledSDE, seed=0, 
                        **extra_args_sde_constr):
    """Create a loss function for evaluating the current model with respect to some
       pre-specified dataset

    Args:
        model_params (TYPE): The SDE model and solver parameters
        loss_params (TYPE): The pamaters used in the loss function computation. 
                            Typically penalty coefficient for the different losses.
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): A value to initialize the parameters of the model
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
    """
    # Deep copy params_model
    params_model = model_params

    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.zeros((2,params_model['n_y']))
    uzero = jnp.zeros((1,params_model['n_u'],))

    # The number of samples is given by the loss dictionary -> If not present, use the default value from the params-model
    num_sample = loss_params.get('num_particles', params_model.get('num_particles', 1) )
    params_model['num_particles'] = num_sample

    # Print the number of particles used for the loss
    print('Using [ N = {} ] particles for the loss'.format(num_sample))

    # Get the horizon from params_loss -> if not present, use the default value from params_model
    params_model['horizon'] = loss_params.get('horizon', params_model.get('horizon', 1))

    # Print the horizon used for the loss
    print('Using [ T = {} ] horizon for the loss'.format(params_model['horizon']))

    # We remove num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss
    params_model.pop('num_short_dt', None)
    params_model.pop('short_step_dt', None)
    params_model.pop('long_step_dt', None)
    print ('Removed num_short_dt, short_step_dt, and long_step_dt from the model as they are not used in the loss and sde training')

    # Define the transform for the sampling function
    def sample_sde(y, u, rng, extra_args=None):
        m_model = sde_constr(params_model, **extra_args_sde_constr)
        return m_model.sample_for_loss_computation(y, u, rng, extra_args)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, yzero, uzero, rng_zero)

    # Let's get nominal parameters values
    nominal_params_val = loss_params.get('nominal_parameters_val', {})
    default_params_val = loss_params.get('default_parameters_val', 0.) # This value imposes that the parameters should be minimized to 0
    _nominal_params_val = set_values_all_leaves(nn_params, default_params_val)
    nominal_params = _nominal_params_val if nominal_params_val is None else update_same_struct_dict(_nominal_params_val, nominal_params_val)
    special_params_val = loss_params.get('special_parameters_val', {})
    nominal_params = get_penalty_parameters(nominal_params, special_params_val, None)

    # Print the resulting penalty coefficients
    print('Nominal parameters values: \n {}'.format(nominal_params))

    # Let's get the penalty coefficients for regularization
    special_parameters = loss_params.get('special_parameters_pen', {})
    default_weights = loss_params.get('default_weights', 0.)
    penalty_coeffs = get_penalty_parameters(nn_params, special_parameters, default_weights)

    # Print the resulting penalty coefficients
    print('Penalty coefficients: \n {}'.format(penalty_coeffs))

    # Nonnegative parameters of the problem
    nonneg_params = get_non_negative_params(nn_params, {k : True for k in params_model.get('noneg_params', []) })

    # Print the resulting nonnegative parameters
    print('Nonnegative parameters: \n {}'.format(nonneg_params))

    # Define a projection function for the parameters
    def nonneg_projection(_params):
        return jax.tree_map(lambda x, nonp : jnp.maximum(x, 0.) if nonp else x, _params, nonneg_params)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, y, u, rng, extra_args=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, 0, None))
        return vmap_sampling(_nn_params, y, u, m_rng, extra_args)

    def loss_fn(_nn_params, y, u, rng, extra_args=None):
        # CHeck if rng is given as  a ingle key
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'

        # Split the key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling, in_axes=(None, 0, 0, 0, 0) if extra_args is not None else (None, 0, 0, 0, None))
        logprob, kl_div, extra_feat = batch_vmap(_nn_params, y, u, rng, extra_args)

        # We change the sign for the cost function that is suitable for minimization
        # Compute the mean of the logprob estimation
        loss_logprob = -jnp.mean(jnp.mean(logprob, axis=1))

        # Cmpute the mean of the kl_divergence
        loss_kl_div = jnp.mean(jnp.mean(kl_div, axis=1))

        # Extra feature mean if there is any
        m_res = { k: jnp.mean(jnp.mean(v, axis=1)) for k, v in extra_feat.items()}

        # Compute the total sum
        total_sum = loss_logprob * loss_params.get('logprob', 1.)
        total_sum += loss_kl_div * loss_params.get('kl', 1.)

        # W loss
        w_loss_arr = jnp.array( [jnp.sum(jnp.square(p - p_n)) * p_coeff \
                            for p, p_n, p_coeff in zip(jax.tree_util.tree_leaves(_nn_params), jax.tree_util.tree_leaves(nominal_params), jax.tree_util.tree_leaves(penalty_coeffs)) ]
                        )
        w_loss = jnp.sum(w_loss_arr)
        # total_sum += params_loss['weights'] * w_loss
        total_sum += w_loss * loss_params.get('pen_params', 1.)

        return total_sum, {'totalLoss' : total_sum, 'logprob' : loss_logprob,
                            'kl' : loss_kl_div, 'weights' : w_loss, **m_res}

    return nn_params, loss_fn, nonneg_projection


def create_online_cost_sampling_fn(params_model,
                            params_mpc,
                            sde_constr= ControlledSDE,
                            seed=0,
                            prior_sampling=False,
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
        (has_xbound, slack_proximal, state_idx, penalty_coeff, state_lb, state_ub ) = \
            initialize_problem_constraints(params_model['n_x'], params_model['n_u'], params_mpc)

    # Transform the penalty coefficient into an array
    penalty_coeff = jnp.array(penalty_coeff) if penalty_coeff is not None else penalty_coeff

    def constr_cost_noprox(x_true, slack_x=None):
        """ Penalty method with nonsmooth cost fuction.
            This should be avoided when doing nonlinear MPC using accelerated gradient descent
        """
        x = x_true[state_idx]
        # diff_x should always be less than 0
        diff_x = jnp.concatenate((x - state_ub, state_lb - x))
        #[TODO Franck] Maybe mean/sum the error instead of doing a mean over states
        return jnp.sum(jnp.where( diff_x > 0, 1., 0.) * jnp.square(diff_x) * penalty_coeff)

    def constr_cost_withprox(x_true, slack_x):
        """ With proximal constraint on the slack variable -> smooth norm 2 regularization
        """
        return jnp.sum(jnp.square(x_true[state_idx]-slack_x) * penalty_coeff)

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
                            extra_dyn_args=extra_dyn_args, extra_cost_args=extra_cost_args, prior_sampling=prior_sampling)

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
        # TODO: Make sure that this makes sense for other nodes
        # return total_cost, x_evol[:,:-1]

        # Modified the cost to add the penalty with respect to constraints of the first state
        # x0_constr = 0. if not has_xbound else constr_cost_noprox(x_evol[0,:-1]) * time_steps[0]
        if has_xbound:
            x_evol = x_evol.at[1:,-1].add(constr_cost_noprox(x_evol[0,:-1]) * time_steps[0]) # Only add it to the first time step -> The other are likely useless
        # x_evol[1,-1] += x0_constr # Only add it to the first time step -> The other are likely useless

        return total_cost, x_evol

    # Initialize an optimization parameters given a sequence of STATE x and u
    # [TODO Fanck] Define it in terms of observation and use the state to observation transformation to match the output
    # However the call of this with an observation will only happen once...
    def _construct_opt_params(u=None, x=None):
        # This ensures that if there is a slack variable
        # the constraint are also enforced on the terminal state
        # num_var = params_model['horizon'] + (0 if has_slack else 1)
        num_var = params_model['horizon']
        zero_u = jnp.ones((num_var, n_u)) * 1e-4
        zero_x = jnp.ones((num_var, len(state_idx))) * 1e-4 if has_slack else None

        if u is not None and u.ndim == 1:
            u = jnp.array([ u for _ in range(num_var)])

        if u is None and not has_slack:
            return zero_u # jnp.zeros((num_var, n_u))

        if u is None and x is None: # slack is true
            return jnp.concatenate((zero_u, zero_x), axis=1)# jnp.zeros((num_var, opt_params_size))

        if u is None and x is not None: # slack is true
            assert x.ndim == 2 and x.shape[0] == num_var+1
            return jnp.concatenate((zero_u, x[:-1,state_idx]), axis=1)

        if u is not None and not has_slack:
            assert u.ndim == 2 and u.shape[0] == num_var
            return u

        if u is not None and x is None: # has slack is true
            assert u.ndim == 2 and u.shape[0] == num_var
            return jnp.concatenate((u, zero_x), axis=1)

        if u is not None and x is not None: # has slack is true
            assert u.ndim == 2 and x.ndim == 2 and u.shape[0]+1 == x.shape[0]
            return jnp.concatenate((u, x[:-1,state_idx]), axis=1)

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