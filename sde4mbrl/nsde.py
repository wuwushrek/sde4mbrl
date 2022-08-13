import jax
import jax.numpy as jnp
import haiku as hk

from sde4mbrl.sde_solver import sde_solver_name
from sde4mbrl.utils import initialize_problem_constraints

# [TODO] Batching over t for loss computationwith fixed time step

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

        # Some check to ensure the right parameters are given
        assert 'n_y' in params and 'n_x' in params and 'n_u' in params,\
                "The number of observations, control inputs, and hidden variables (states) should be given"

        # Save the parameters
        self.params = params

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

    def prior_diffusion(self, t, x):
        """Save the prior diffusion function with the attribute name 'prior_diffusion'
        Access to the prior functions is via self.prior_diffusion, which is a function
        of the time and latent state of the system

        Args:
            t (TYPE): The current time
            x (TYPE): The current state of the system (can be latent state)
        """
        pass

    def prior_drift(self, t, x, u):
        """Save the prior drift function with the attribute name 'prior_drift'
        Access to the prior functions is via self.prior_drift, which is a function
        of the time and latent state of the system

        Args:
            t (TYPE): The current time
            x (TYPE): The current state of the system (can be latent state)
            u (TYPE): The current control signal applied to the system
        """
        pass

    def posterior_drift(self, t, x, u):
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

    def identity_logprob(self, y, x, rng):
        """ Logprobability function for deterministic transformation
            between the states and the observation
        """
        # Maximize the log probablity
        return -jnp.sum(jnp.square(y-x))

    def identity_obs2state(self, y, rng):
        """ Return the identity fuction when the observation is exactly the
            state of the system
        """
        return y


    def sample_prior(self, ts, y0, uVal, rng, params_solver=None):
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
                                ts, x0, uVal, rng_brownian, params_solver)



    def sample_posterior(self, ts, y0, uVal, rng, params_solver=None):
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
                                ts, x0, uVal, rng_brownian, params_solver)


    def sample_general(self, drift_term, diff_term, ts, x0, uVal,
                        rng_brownian, params_solver=None):
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
        # Load the sde solver or define a new solver
        if params_solver is not None:
            sde_solve = sde_solver_name[params_solver.get('sde_solver', 'stratonovich_milstein')]
        else:
            sde_solve = self.sde_solve

        # Finally solve the stochastic differential equation
        if hk.running_init():
            # Dummy return in this case -> This case is just to initialize NNs
            # Initialize the drift and diffusion parameters
            drift_val = drift_term(ts[0], x0, uVal if uVal.ndim ==1 else uVal[0])
            diff_val = diff_term(ts[0], x0)

            #[TODO Franck] Maybe make the return type to be the same after sde_solve
            #Fine if the returned types are different at initilization
            return jnp.zeros_like(x0)[None] # 2D array to be similar with sde_solve
        else:
            # Solve the sde and return its output (latent space)
            return sde_solve(ts, x0, uVal, rng_brownian, drift_term, diff_term)

    def sample_for_loss_computation(self, ts, meas_y, uVal, rng):
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
        assert ts.shape[0] == meas_y.shape[0] and \
                (meas_y.shape[0] == uVal.shape[0]+1 or meas_y.shape[0] == uVal.shape[0]),\
                'Trajectory horizon must match'

        # Define the augmented dynamics or the ELBO bound
        def augmented_drift(_t, _aug_x, _u):
            """ Define the augmented drift function """
            _x = _aug_x[:-1]
            drift_pri_x = self.prior_drift(_t, _x, _u)
            if self.posterior_is_prior_plus_error:
                error_prior = self.prior_error(_t, _x, _u)
                drift_pos_x = drift_pri_x + error_prior
            else:
                drift_pos_x = self.posterior_drift(_t, _x, _u)
                error_prior = drift_pos_x - drift_pri_x
            diff_x = self.prior_diffusion(_t, _x) # A vector
            cTerm = 0.5 * jnp.sum(jnp.square((error_prior)/diff_x))
            return jnp.concatenate((drift_pos_x, jnp.array([cTerm])))

        def augmented_diffusion(_t, _aug_x):
            """ Define the augmented diffusion function """
            _x = _aug_x[:-1]
            diff_x = self.prior_diffusion(_t, _x)
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Now convert the observation to a state and then integrate it
        rng_logprob, rng_obs2state, rng_brownian = jax.random.split(rng, 3)
        x0 =  self.obs2state(meas_y[0], rng_obs2state)

        # Solve the augmented integration problem
        aug_x0 = jnp.concatenate((x0, jnp.array([0.])) )
        est_x = self.sample_general(augmented_drift, augmented_diffusion, ts, aug_x0, uVal, rng_brownian)

        # [TODO Franck] Average the KL divergence over each point of the traj
        # might not be a good idea _> To investigate !!! Maye the sum !!!!
        xnext, kl_div = est_x[:, :-1], jnp.sum(est_x[:, -1])

        # Compute the logprob(obs | state) for the estimated state
        # as a function of the observation
        plogVal = self.logprob(meas_y[1:], xnext[1:], rng_logprob)

        # Extra values to print or to penalize the loss function on
        extra = {}

        return plogVal, kl_div, extra

    def sample_dynamics_with_cost(self, ts, y, u, rng, cost_fn):
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
        def cost_aug_drift(_t, _aug_x, _u):
            _x = _aug_x[:-1]
            actual_u = _u[:self.params['n_u']]
            drift_pos_x = self.posterior_drift(_t, _x, actual_u)
            slack_t = _u[self.params['n_u']:] if _u.shape[0] > self.params['n_u'] else None
            # [TODO Franck] Maybe add time dependency ?
            cost_term = cost_fn(_x, actual_u, slack_t)
            return jnp.concatenate((drift_pos_x, jnp.array([cost_term])))

        def cost_aug_diff(_t, _aug_x):
            _x = _aug_x[:-1]
            diff_x = self.prior_diffusion(_t, _x)
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Now convert the observation to a state and then integrate it
        rng_obs2state, rng_brownian = jax.random.split(rng)
        x =  self.obs2state(y, rng_obs2state)

        # Solve the augmented integration problem
        aug_x = jnp.concatenate((x, jnp.array([0.])) )
        return self.sample_general(cost_aug_drift, cost_aug_diff, ts, aug_x, u, rng_brownian)



def create_sampling_fn(params_model, sde_constr= ControlledSDE, prior_sampling=True, seed=0):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        prior_sampling (bool, optional): Specify if the returned function samples from prior or posterior
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: The parameters of the model, a function for multi-sampling
    """
    # Some dummy initialization scheme
    #[TODO Franck] Maybe something more general in case these inputs are not valid
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.ones((params_model['n_y'],))
    uzero = jnp.ones((params_model['n_u'],))
    tzero = jnp.array([0., 0.001])
    num_sample = params_model['num_particles']

    # Define the transform for the sampling function
    def sample_sde(t, y, u, rng):
        """ Sampling function """
        m_model = sde_constr(params_model)
        m_sampling = m_model.sample_prior if prior_sampling else m_model.sample_posterior
        return m_sampling(t, y, u, rng)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, tzero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, t, y, u, rng):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, None, 0))
        return vmap_sampling(_nn_params, t, y, u, m_rng)

    return nn_params, multi_sampling



def create_model_loss_fn(params_model, params_loss, sde_constr=ControlledSDE, seed=0):
    """Create a loss function for evaluating the current model with respect to some
       pre-specified dataset

    Args:
        params_model (TYPE): The SDE model and solver parameters
        params_loss (TYPE): The pamaters used in the loss function. Typically penalty coefficient
                            for the different losses.
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: The parameters of the model, a function to compute a loss with respect to a dataset
    """
    rng_zero = jax.random.PRNGKey(seed)
    tzero = jnp.zeros((2,))
    yzero = jnp.ones((2,params_model['n_y']))
    uzero = jnp.ones((1,params_model['n_u'],))

    num_sample = params_model['num_particles_learning_sde']

    # Define the transform for the sampling function
    def sample_sde(t, y, u, rng):
        m_model = sde_constr(params_model)
        return m_model.sample_for_loss_computation(t, y, u, rng)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, tzero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, t, y, u, rng):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, None, 0))
        return vmap_sampling(_nn_params, t, y, u, m_rng)

    def loss_fn(_nn_params, y, u, rng, t=None):
        # CHeck if rng is given as  a ingle key
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'

        # Split the key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling, in_axes=(None, 0, 0, 0, 0))
        logprob, kl_div, extra_feat = batch_vmap(_nn_params, t, y, u, rng)

        # [TODO Franck] Remove this check as it might be useless
        # Check the dimension of the problem
        assert logprob.shape == (y.shape[0], num_sample) and \
                kl_div.shape == (y.shape[0], num_sample), "Dimension does not match"

        # We change the sign for the cost function that is suitable for minimization
        # Compute the mean of the logprob estimation
        loss_logprob = -jnp.mean(jnp.mean(logprob, axis=1))

        # Cmpute the mean of the kl_divergence
        loss_kl_div = jnp.mean(jnp.mean(kl_div, axis=1))

        # Extra feature mean if there is any
        m_res = { k: jnp.mean(jnp.mean(v, axis=1)) for k, v in extra_feat.items()}

        # Compute the total sum
        total_sum = loss_logprob * params_loss['logprob']
        total_sum += loss_kl_div * params_loss['kl']

        return total_sum, {'Loss' : total_sum, 'logprob' : loss_logprob, 'kl' : loss_kl_div, **m_res}

    return nn_params, loss_fn


def create_valuefun_loss_fn(params_model, params_loss, cost_fn, R_inv, sde_constr= ControlledSDE, seed=0):
    """Create a function that integrates the dynamics as well as a cost function to minimize.
       Typically, the cost function is the objective used in the underlying MPC problem.
       This function returns a loss function to train the underlying value function of the system
       while including the sde dynamics in the loss function
       The idea follows the iterative improvement from generalized HJB equations

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        params_loss (TYPE): The pamaters used in the loss function. Typically penalty coefficient
                            for the different losses.
        cost_fn (TYPE): The cost function to optimize on the fly in a stochastic MPC manner
        R_inv (TYPE): The cost function is assumed to be c(t,x) +uR(t,x)u. In this case,
                      R_inv returns the inverse of R(t,x)
        sde_constr (TYPE): A class constructor that is child of ControlledSDE class
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: Description
    """
    # Random ky for initialization
    rng_zero = jax.random.PRNGKey(seed)

    # Initialization of the observation and uzero
    yzero = jnp.ones((params_model['n_y'],))
    xzero = jnp.ones((params_model['n_x'],))
    uzero = jnp.ones((1,params_model['n_u']))
    tzero = jnp.array([0., 0.001])

    # Number of control inputs
    n_u = params_model['n_u']

    # Save the number of samples
    num_sample = params_model['num_particles_valuefun_learning']

    (has_ubound, input_lb, input_ub),\
        (has_xbound, _, state_idx, penalty_coeff, state_lb, state_ub ) = \
            initialize_problem_constraints(params_model)

    def constr_cost_noprox(x_true, slack_x=None):
        """ Penalty method with nonsmooth cost fuction
        """
        x = x_true[state_idx]
        # diff_x should always be less than 0
        diff_x = jnp.concatenate((x - state_ub, state_lb - x))
        #[TODO Franck] Maybe sum/mean the error instead of doing a mean over states
        return jnp.sum(jnp.where( diff_x > 0, 1., 0.) * jnp.square(diff_x))

    # A function to constraint a vector between a given minimum and maximum values
    constr_vect = lambda a, a_lb, a_ub:  jnp.minimum(jnp.maximum(a, a_lb), a_ub)

    # Now ready to define the constraint cost as well as the proximal operator if needed
    constr_cost = None
    proximal_fn = None

    # No bound on u are given but x is constrained and we are using proximal operator
    # To constrain the slack variable associated with these states
    if has_xbound:
        constr_cost = constr_cost_noprox

    if has_ubound and params_model.get('enforce_ubound', False):
        proximal_fn = lambda _u: constr_vect(_u, input_lb, input_ub)

    # Define the augmented cost with the penalization term
    def aug_cost_fn(_x, _u, _slack=None, final_cost=None):
        # Compute the actual cost
        actual_cost = final_cost(_x) if final_cost is not None else cost_fn(_x, _u)
        if constr_cost is None:
            return actual_cost
        # Compute the constraints cost
        pen_cost = constr_cost(_x, _slack)
        return actual_cost + penalty_coeff * pen_cost

    # Define the function to integrate the cost function and the dynamics
    def sample_cost_evol(t, y, _u, rng):
        # Build the SDE solver
        m_model = sde_constr(params_model)
        # Compute the evolution of the state -> u is a function so second return argument are the function values
        x_evol, _ = m_model.sample_dynamics_with_cost(t, y, _u, rng, aug_cost_fn)
        # Estimate the value function at the startign state
        value_xt = m_model.value_fn(x_evol[0,:-1])
        # Estimate the value function at the ending state -> If terminal cost at an end time are given
        value_xf = m_model.value_fn(x_evol[-1,:-1])
        # final integrated cost, final state, est value at initial time, est value at end time
        return x_evol[-1,-1], x_evol[-1,:-1], value_xt, value_xf

    # Define the control function to apply for value function update
    def control_affine_opt(t, x):
        """ Return optimal control given value function and control affine dynamics
        """
        # Obatin the model
        m_model = sde_constr(params_model)

        # Some check up
        assert hasattr(m_model, 'G_fn'), \
            "The class should contain a function to evaluate the control affine coefficient of the dynamics"
        assert hasattr(m_model, 'value_fn'), \
            "The class should contain a function value_fn to estimate the cost to go/ value function"

        # COmpute the control matrix coefficient in f(t,x) + G(t,x) @ u
        G_val = m_model.G_fn(t, x)

        if hk.running_init():
            # Trick so that we can call jax.grad inside in haiku module
            Vval = m_model.value_fn(x)
            grad_value_fn = jnp.full(x.shape, Vval)
        else:
            grad_value_fn = jax.grad(m_model.value_fn)(x)
        # Compute the optimal control input given the value function
        u_opt = - R_inv(t, x) @ (G_val.T @ grad_value_fn)

        #[TODO Franck] Maybe not a good idea looking at the theory
        # But this can be deactivated using the 'enforce_ubound' parameter input
        if proximal_fn is not None:
            # Constrain the control if required
            u_opt = proximal_fn(u_opt)

        return u_opt

    # Define the control function to apply for value function update
    def value_function(x):
        """ Return the value function/cost-to-go for a given state
        """
        m_model = sde_constr(params_model)
        assert hasattr(m_model, 'value_fn'), \
            "The class should contain a function value_fn to estimate the cost to go/ value function"
        return m_model.value_fn(x)

    # Now transform all functions to pure functions
    control_affine_pure = hk.without_apply_rng(hk.transform(control_affine_opt))
    _params_control_affine = control_affine_pure.init(rng_zero, tzero[0], xzero)

    value_function_pure = hk.without_apply_rng(hk.transform(value_function))
    _params_value_function = value_function_pure.init(rng_zero,xzero)

    sample_cost_evol_pure = hk.without_apply_rng(hk.transform(sample_cost_evol))
    _params_sample_cost_evol = sample_cost_evol_pure.init(rng_zero, tzero, yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling_value_fn(_nn_params, t, y, rng, _nn_params_terminal, terminal_info=None):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        u_fun = lambda _t, _x: control_affine_pure.apply(_nn_params_terminal, _t, _x)
        cost_to_go = lambda _x: value_function_pure.apply(_nn_params_terminal, _x)

        def cost_and_costogo(_t, _y, _rng):
            total_cost, xfinal, value_xt, value_xf = sample_cost_evol_pure.apply(_nn_params, _t, _y, u_fun, _rng)
            end_cost = aug_cost_fn(xfinal, None, None, final_cost = cost_to_go)
            if terminal_info is not None:
                T, terminal_fn = terminal_info
                diff_terminal = jnp.where(_t >= T, terminal_fn(xfinal)-value_xf, jnp.array(0.))
            else:
                diff_terminal = jnp.array(0.)
            diff_initial = total_cost + end_cost - value_xt
            return diff_initial, diff_terminal

        m_rng = jax.random.split(rng, num_sample)
        diff_initial, diff_terminal = jax.vmap(cost_and_costogo)(t, y, m_rng)
        return jnp.square(jnp.mean(diff_initial)), jnp.square(jnp.mean(diff_terminal))

    # Define the loss function to learn value function and optimize it
    def loss_fn(_nn_params, t, y, rng, _nn_params_terminal, terminal_info=None):
        assert rng.ndim == 1, 'The rng key is splitted inside the loss function computation'

        # Split the key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        batch_vmap = jax.vmap(multi_sampling_value_fn, in_axes=(None, 0, 0, 0, None, None))
        diff_value_xt, diff_value_xend = batch_vmap(_nn_params, t, y, rng, _nn_params_terminal, terminal_info)

        diff_value_xt = jnp.mean(diff_value_xt)
        diff_value_xend = jnp.mean(diff_value_xend)
        total_sum = params_loss['Vt'] * diff_value_xt + params_loss['Vend'] * diff_value_xend

        # Estimated value function
        est_value = value_function_pure.apply(_nn_params, )
        return total_sum, {'ErrorValue': total_sum, 'Vt' : diff_value_xt, 'Vend' : diff_value_xend}

    return (_params_sample_cost_evol, _params_control_affine, _params_value_function), control_affine_pure.apply, loss_fn



def create_online_cost_sampling_fn(params_model, cost_fn,
                            terminal_cost=None,
                            sde_constr= ControlledSDE,
                            seed=0):
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
    yzero = jnp.ones((params_model['n_y'],))
    uzero = jnp.ones((1, n_u))
    tzero = jnp.array([0., 0.001])

    # Save the number of samples
    num_sample = params_model['num_particles_online_control']

    (has_ubound, input_lb, input_ub),\
        (has_xbound, _, state_idx, penalty_coeff, state_lb, state_ub ) = \
            initialize_problem_constraints(params_model)

    def constr_cost_noprox(x_true, slack_x=None):
        """ Penalty method with nonsmooth cost fuction.
            This should be avoided when doing nonlinear MPC usin accelerated gradient descent
        """
        x = x_true[state_idx]
        # diff_x should always be less than 0
        diff_x = jnp.concatenate((x - state_ub, state_lb - x))
        #[TODO Franck] Maybe mean/sum the error instead of doing a mean over states
        return jnp.sum(jnp.where( diff_x > 0, 1., 0.) * jnp.square(diff_x))

    def constr_cost_withprox(x_true, slack_x):
        """ With proximal constraint on the slack variable -> smooth norm 2 regularization
        """
        return jnp.sum(jnp.square(x_true[state_idx]-slack_x))

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
    def aug_cost_fn(_x, _u, _slack, final_cost=False):
        # Compute the actual cost
        if final_cost:
            actual_cost = 0. if terminal_cost is None else terminal_cost(_x)
        else:
            actual_cost = cost_fn(_x, _u)
        if constr_cost is None:
            return actual_cost
        # Compute the constraints cost
        pen_cost = constr_cost(_x, _slack)
        return actual_cost + penalty_coeff * pen_cost

    # Define the function to integrate the cost function and the dynamics
    def sample_sde(t, y, opt_params, rng):

        # Do some check
        assert opt_params.ndim == 2, 'The parameters must be a two dimension array'
        if has_slack:
            assert opt_params.shape[1] == opt_params_size, 'Shape of the opt params do not match'
        else:
            assert opt_params.shape[1] == n_u, 'Shape of the opt params do not match'
        assert (t.shape[0] == opt_params.shape[0]) or (t.shape[0] == opt_params.shape[0]+1),\
                 'Time step should match the parameter size'

        # Build the SDE solver
        m_model = sde_constr(params_model)

        # Compute the evolution of the state
        x_evol = m_model.sample_dynamics_with_cost(t, y, opt_params, rng, aug_cost_fn)

        # Evaluate the cost_to_go function
        # [TODO Franck] probably a bug here when using ts.shape-1 optimization variable
        # When has_slack should have ts.shape opt variable so that the slack
        # matches the end state/final state
        end_cost = aug_cost_fn(x_evol[-1,:-1], None,
                                opt_params[-1, n_u:] if has_slack else None,
                                final_cost=True)

        # Compute the total cost by adding the terminal cost
        total_cost = end_cost + x_evol[-1,-1]
        return total_cost, x_evol[:,:-1]

    # Initialize an optimization parameters given a sequence of STATE x and u
    # [TODO Fanck] Define it in terms of observation and use the state to observation transformation to match the output
    # However the call of this with an observation will only happen once...
    def construct_opt_params(x, u):
        """ u  must be the same shape as the time step """
        assert u.ndim == 2, 'The control must be a 2D over the time horizon'
        if not has_slack:
            return u
        zero_x = jnp.zeros((u.shape[0],len(state_idx),))
        return jnp.concatenate((u, x[:,state_idx]), axis=1) if x is not None \
                else jnp.concatenate((u, zero_x), axis=1)


    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, tzero, yzero, construct_opt_params(None, uzero), rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, t, y, opt_params, rng):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, None, 0))
        total_loss, xtraj = vmap_sampling(_nn_params, t, y, opt_params, m_rng)
        return jnp.mean(total_loss), xtraj

    # Properly define the proximal_function
    vmapped_prox = None if proximal_fn is None else jax.vmap(proximal_fn)
    return nn_params, multi_sampling, vmapped_prox, constr_cost, construct_opt_params
