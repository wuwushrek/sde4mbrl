import jax
import jax.numpy as jnp
import haiku as hk

from jax.experimental.host_callback import id_print

import diffrax
import pickle

# [TODO Franck] Incorporate piecewise constant for MuJoCo/Brax Env
# [TODO Franck] Fixed ts should be improved for memory and computational efficiency
# [TODO Franck] When loading from file upgrade models params from another dict arguments
# [TODO Franck] Check problem dimension

class LatentSDE(hk.Module):
    """Define a latent SDE object with latent variables representing the states
       of an SDE. In addition, this class should implement functions to estimate the
       the latent space from observation either in a probabilistic or deterministic
       manner.

       This class implements several function to train an SDE to fit data
       given a prior as another SDE. A user should create a class that inherits
       properties of this class while redefining the functions below:
            - prior_diffusion : Define the prior diffusion function (can be parameterized by NN)
            - prior_drift : Define the prior drift function (can be parameterized by NN)
            - posterior_drift : Define the posterior drift function (can be parameterized)
            - init_encoder : Initialize the encoder function -> Provide a way to go from observation to state and its log probability
    """
    def __init__(self, params={}, name=None):
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

        # Initialize the encode
        self.init_encoder()

        # Initialize the SDE solver
        self.sde_solve = differentiable_sde_solver(params.get('sde_solver', {}))


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
        """ Cmpute a Gaussian estimate given a mean and standard deviation function
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
            sde_solve = differentiable_sde_solver(params_solver)
        else:
            sde_solve = self.sde_solve

        # This test if when uVal is given as a function of t and x
        # rather than an array of float values
        if not hasattr(uVal, 'ndim'):
            u_fun = uVal
        elif uVal.ndim == 1:
            u_fun = lambda t, x : uVal
        else:
            # Check the control input dimension and broadcast it if needed
            assert uVal.ndim ==2 and uVal.shape[0] == ts.shape[0],\
                'Control u should be a 2D array and should match ts size'
            # Now define the control as a linear interpolation of time
            u_fun = lambda t, x : linear_interpolation(t, ts, uVal)

        # Modified the vector field to implement a time-varying control
        drift_fn = lambda t, x, args=None : drift_term(t, x, u_fun(t, x))
        diff_fn = lambda t, x, args=None : diff_term(t, x)

        # Finally solve the stochastic differential equation
        if hk.running_init():
            # Dummy return in this case -> This case is just to initialize NNs
            # Initialize the drift and diffusion parameters
            drift_val = drift_fn(ts[0], x0)
            diff_val = diff_fn(ts[0], x0)

            #[TODO Franck] Maybe make the return type to be the same after sde_solve
            #Fine if the returned types are different at initilization
            return jnp.zeros_like(x0)[None] # 2D array to be similar with sde_solve
        else:
            # Solve the sde and return its output (latent space)
            return sde_solve(ts, x0, rng_brownian, drift_fn, diff_fn)

    def sample_for_loss_computation(self, ts, meas_y, uVal, rng):
        """Summary

        Args:
            ts (TYPE): Description
            meas_y (TYPE): Description
            uVal (TYPE): Description
            rng (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Check if the trajectory horizon matches
        assert ts.shape[0] == meas_y.shape[0] and meas_y.shape[0] == uVal.shape[0],\
                'Trajectory horizon must match'

        # Define the augmented dynamics or the ELBO bound
        def augmented_drift(_t, _aug_x, _u):
            """ Define the augmented drift function """
            _x = _aug_x[:-1]
            drift_pos_x = self.posterior_drift(_t, _x, _u)
            drift_pri_x = self.prior_drift(_t, _x, _u)
            diff_x = self.prior_diffusion(_t, _x) # A vector
            cTerm = 0.5 * jnp.sum(jnp.square((drift_pos_x - drift_pri_x)/diff_x))
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

        # Extra values to print
        extra = {'noise' : jnp.mean(self.prior_diffusion(ts[0], aug_x0))}

        return plogVal, kl_div, extra

    def sample_dynamics_with_cost(self, ts, y, u, rng, cost_fn, slack=None):
        """Generate a function that integrate the sde dynamics augmented with a cost
           function evolution (which is also integrated along the dynamics)

        Args:
            ts (TYPE): The time indexes at which the integration happens
            y (TYPE): The current observation of the system
            u (TYPE): The sequence of control to applied at each time step
            rng (TYPE): A random key generator for Brownian noise
            cost_fn (TYPE): The cost function to integrate
                            cost_fn : (_x, _u, _slack)
            slack (None, optional): Slack variable that are used in the cost function
                                    They are extrapolated similarly to the control inputs
                                    before integration

        Raises:
            The state at the last time Ts and
        """
        # If slack variables are given extrapolate it over the time horizon
        assert slack is None or (slack.ndim == 2 and slack.shape[0] == ts.shape[0]), \
                'Slack dimension and the time horizon should match for interpolation'
        slack_fun = lambda t : None if slack is None else linear_interpolation(t, ts, slack)

        # Define the augmented dynamics
        def cost_aug_drift(_t, _aug_x, _u):
            _x = _aug_x[:-1]
            drift_pos_x = self.posterior_drift(_t, _x, _u)
            slack_t = slack_fun(_t)
            # [TODO Franck] Maybe add time dependency ?
            cost_term = cost_fn(_x, _u, slack_t)
            return jnp.concatenate((drift_pos_x, jnp.array([cost_term])))

        def cost_aug_diff(_t, _aug_x):
            _x = _aug_x[:-1]
            diff_x = self.prior_diffusion(_t, _x)
            return jnp.concatenate((diff_x, jnp.array([0.])))

        # Now convert the observation to a state and then integrate it
        rng_obs2state, rng_brownian = jax.random.split(rng, 2)
        x =  self.obs2state(y, rng_obs2state)

        # Solve the augmented integration problem
        aug_x = jnp.concatenate((x, jnp.array([0.])) )
        est_x = self.sample_general(cost_aug_drift, cost_aug_diff, ts, aug_x, u, rng_brownian)
        return est_x


def create_sampling_fn(params_model, sde_constr= LatentSDE, prior_sampling=True, seed=0):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of LatentSDE class
        prior_sampling (bool, optional): Specify if the returned function samples from prior or posterior
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: Description
    """
    rng_zero = jax.random.PRNGKey(seed)
    yzero = jnp.ones((params_model['n_y'],))
    uzero = jnp.ones((params_model['n_u'],))
    num_sample = params_model['num_particles']
    # Define the transform for the sampling function
    def sample_sde(t, y, u, rng):
        m_model = sde_constr(params_model)
        m_sampling = m_model.sample_prior if prior_sampling else m_model.sample_posterior
        return m_sampling(t, y, u, rng)

    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, jnp.array([0., 0.001]), yzero, uzero, rng_zero)

    # Now define the n_sampling method
    def multi_sampling(_nn_params, t, y, u, rng):
        assert rng.ndim == 1, 'RNG must be a single key for vmapping'
        m_rng = jax.random.split(rng, num_sample)
        vmap_sampling = jax.vmap(sampling_pure.apply, in_axes=(None, None, None, None, 0))
        return vmap_sampling(_nn_params, t, y, u, m_rng)

    return nn_params, multi_sampling

def create_loss_fn(params_model, params_loss, sde_constr=LatentSDE, seed=0):
    """Create a sampling function for prior or posterior distribution

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        params_loss (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of LatentSDE class
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: Description
    """
    rng_zero = jax.random.PRNGKey(seed)
    tzero = jnp.zeros((2,))
    yzero = jnp.ones((2,params_model['n_y']))
    uzero = jnp.ones((2,params_model['n_u'],))

    num_sample = params_model['num_particles']
    ts = None if not params_model['fixed_ts'] else \
            np.array([i * params_model['sde_solver']['init_step'] \
                        for i in range(params_loss['horizon']) ])

    # Define the transform for the sampling function
    # ts, meas_y, uVal, rng
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
        assert rng.ndim == 1, 'THe rng key is splitted inside the loss function computation'
        # Split the key first
        rng = jax.random.split(rng, y.shape[0])

        # Do multiple step prediction of state and compute the logprob and KL divergence
        if ts is not None:
            batch_vmap = jax.vmap(multi_sampling, in_axes=(None, None, 0, 0, 0))
            t = ts
        else:
            assert t is not None, "Both t and ts can not be None"
            batch_vmap = jax.vmap(multi_sampling, in_axes=(None, 0, 0, 0, 0))
        logprob, kl_div, extra_feat = batch_vmap(_nn_params, t, y, u, rng)

        # [TODO Franck] Remove this check as it might be useless
        # Check the dimension of the problem
        assert logprob.shape == (y.shape[0], num_sample) and \
                kl_div.shape == (y.shape[0], num_sample), "Dimension does not match"

        # We change the sign for a cost function to minimize
        # Compute the mean of the logprob estimation
        loss_logprob = -jnp.mean(jnp.mean(logprob, axis=1))

        # Cmpute the mean of the logprob estimation
        loss_kl_div = jnp.mean(jnp.mean(kl_div, axis=1))

        # Extra feature mean
        m_res = { k: jnp.mean(jnp.mean(v, axis=1)) for k, v in extra_feat.items()}

        # Compute the total sum
        total_sum = loss_logprob * params_loss['logprob']
        total_sum += loss_kl_div * params_loss['kl']
        total_sum += - m_res['noise'] * params_loss['noise'] # Maximimze the noise
        return total_sum, {'logprob' : loss_logprob, 'kl' : loss_kl_div, **m_res}

    return nn_params, loss_fn

def create_cost_sampling_fn(params_model, cost_fn,
                            terminal_cost=None,
                            sde_constr= LatentSDE,
                            seed=0):
    """Create a sampling function for the posterior combined with a loss function

    Args:
        params_model (TYPE): The SDE solver parameters and model parameters
        sde_constr (TYPE): A class constructor that is child of LatentSDE class
        seed (int, optional): A value to initialize the parameters of the model

    Returns:
        TYPE: Description
    """
    # Random ky for initialization
    rng_zero = jax.random.PRNGKey(seed)

    # Initialization of the observation and uzero
    yzero = jnp.ones((params_model['n_y'],))
    uzero = jnp.ones((2,params_model['n_u']))
    tzero = jnp.array([0., 0.001])

    # Number of control inputs
    n_u = params_model['n_u']

    # Save the number of samples
    num_sample = params_model['num_particles']

    # Check if some bounds on u are present
    has_ubound = 'input_constr' in params_model
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
    # By default we impose bounds constraint on the states using nonsmooth penalization
    slack_proximal = False
    if has_xbound:
        print('Found states bound constraints...\n')
        slack_dict = params_model['state_constr']
        assert len(slack_dict['state_id']) <= params_model['n_x'] and \
                len(slack_dict['state_id']) == len(slack_dict['state_bound']),\
                'The number of the constrained states identifier does not match the number of bounds'
        state_idx = jnp.array(slack_dict['state_id'])
        slack_proximal = slack_dict['slack_proximal']
        penalty_coeff = slack_dict['state_penalty']
        state_lb = jnp.array([x[0] for x in slack_dict['state_bound']])
        state_ub = jnp.array([x[1] for x in slack_dict['state_bound']])

    def constr_cost_noprox(x_true, slack_x=None):
        """ Penalty method with nonsmooth cost fuction
        """
        x = x_true[state_idx]
        # diff_x should always be less than 0
        diff_x = jnp.concatenate((x - state_ub, state_lb - x))
        #[TODO Franck] Maybe sum the error instead of doing a mean over states
        return jnp.sum(jnp.where( diff_x > 0, 1., 0.) * jnp.square(diff_x))

    def constr_cost_withprox(x_true, slack_x):
        """ With proximal constraint on the slack variable -> smooth norm 2 regularization
        """
        return jnp.sum(jnp.square(x_true[state_idx]-slack_x))

    # A function to constraint a vector between a given minimum and maximum values
    constr_vect = lambda a, a_lb, a_ub:  jnp.maximum(jnp.minimum(a, a_lb), a_ub)

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
        print(opt_params_size, opt_params)
        if has_slack:
            assert opt_params.shape[1] == opt_params_size, 'Shape of the opt params do not match'
        else:
            assert opt_params.shape[1] == n_u, 'Shape of the opt params do not match'
        assert t.shape[0] == opt_params.shape[0], 'Time step should match the parameter size'
        # Separate the parameters into slack variable and control variable
        u_val, slack = (opt_params[:, :n_u], opt_params[:,n_u:]) if has_slack else (opt_params, None)
        # Build the SDE solver
        m_model = sde_constr(params_model)
        # Compute the evolution of the state
        x_evol = m_model.sample_dynamics_with_cost(t, y, u_val, rng, aug_cost_fn, slack)
        # Evaluate the cost_to_go function
        end_cost = aug_cost_fn(x_evol[-1,:-1], None,
                                slack[-1] if slack is not None else slack,
                                final_cost=True)
        # Compute the total cost by adding the terminal cost
        total_cost = end_cost + x_evol[-1,-1]
        return total_cost, x_evol[:,:-1]

    # Initialize an optimization parameters given a sequence of STATE x and u
    # [TODO Fanck] Define it in terms of observation and use the state to observation
    # transformation to match the output
    # However the call of this with an observation will only happen once
    def construct_opt_params(x, u):
        if not has_slack:
            return u
        zero_x = jnp.zeros((u.shape[0],len(state_idx),))
        return jnp.concatenate((u, x[:,state_idx]), axis=1) if x is not None \
                else jnp.concatenate((u, zero_x), axis=1)


    # Transform the function into a pure one
    sampling_pure =  hk.without_apply_rng(hk.transform(sample_sde))
    nn_params = sampling_pure.init(rng_zero, tzero, yzero,
                                    construct_opt_params(None, uzero),
                                    rng_zero)

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


def load_model_from_file(file_dir, sde_constr, seed=0, num_particles=None):
    """ Load and return the learned weight parameters, and
        a function to estimate/predict trajectories of the system

    Args:
        file_dir (str): Directory of the file containing models parameters+opt weight
    """
    # Load the file containing the trajectory
    mFile = open(file_dir, 'rb')
    mData = pickle.load(mFile)
    mFile.close()

    m_params = mData["best_params"]
    params_model = mData['training_parameters']['params']['model']
    if num_particles is not None:
        params_model['num_particles'] = num_particles

    prior_params, _prior_fn = create_sampling_fn(params_model, sde_constr=sde_constr,
                                    prior_sampling=True, seed=seed)
    posterior_params, _posterior_fn = create_sampling_fn(params_model, sde_constr=sde_constr,
                                    prior_sampling=False, seed=seed)
    prior_fn = lambda t, y, u, rng: _prior_fn(prior_params, t, y, u, rng)
    posterior_fn = lambda t, y, u, rng: _posterior_fn(m_params, t, y, u, rng)
    return prior_fn, posterior_fn, \
        {'prior_params':prior_params, 'posterior_params': posterior_params,
         'model' : params_model, 'best_params' : m_params}


def differentiable_sde_solver(params_solver={}):
    """Construct and return a function that solves sdes and that is
        differentiable in forward and reverse mode

    Args:
        params_solver (dict): A dictionary containing solver specific parameters.
            The key and value of the dictionary are described below:
          - max_steps=4096 : The maximum steps when discretizing the SDE

          - init_step=0.1 : The initial time step of the numerical integrator

          - stepsize_controller=diffrax.ConstantStepSize(.)
                    or for adaptive scheme diffrax.PIDController(rtol,atol, ...)

          - adjoint=diffrax.RecursiveCheckpointAdjoint() or diffrax.NoAdjoint()

          - solver=diffrax.ReversibleHeun() : Define the solver to use for
                                              numerical integration

          - specific_sde=False : Boolean specifying if solver is specific
                                (see Diffrax documentation)

          - brownian_tol = init_step/2 : Specify the threshold in the virtual
                                         brownian tree when using fixed/adaptive time step
                                         integrator. For fixed time step, a value slightly
                                         lower than the time step is enough. For adaptive
                                         time step, it should be small but the complexity
                                         (computation) increases too.
                                         https://docs.kidger.site/diffrax/api/brownian/#diffrax.VirtualBrownianTree

          - ts = jnp.array([t0, t1, .., tn]) : Time indexes to save the solution
                                       of the integration
    """
    # Extract the parameters to construct the numerical differentiator
    specific_sde = params_solver.get('specific_sde', False)

    # Extract the maximum number of steps to solve the problem
    max_steps = params_solver.get('max_steps', 4096)

    # Initial time step
    dt0 = params_solver.get('init_step', 0.001)

    # Pick the step size controller
    if 'stepsize_controller' not in params_solver:
        stepsize_controller = diffrax.ConstantStepSize()
    else:
        stepsize_fn = params_solver['stepsize_controller'].get('name', 'ConstantStepSize')
        stepsize_args = params_solver['stepsize_controller'].get('params', {})
        stepsize_controller = getattr(diffrax, stepsize_fn)(**stepsize_args)

    # Specify if adjoint method is enable for backpropagtion
    if 'adjoint' not in params_solver:
        adjoint = diffrax.RecursiveCheckpointAdjoint()
    else:
        adjoint_fn = params_solver['adjoint'].get('name', 'RecursiveCheckpointAdjoint')
        adjoint_args = params_solver['adjoint'].get('params', {})
        adjoint = getattr(diffrax, adjoint_fn)(**adjoint_args)

    # Construct the SDE solver
    if 'solver' not in params_solver:
        solver = diffrax.ReversibleHeun()
    else:
        solver_fn = params_solver['solver'].get('name', 'ReversibleHeun')
        solver_args = params_solver['solver'].get('params', {})
        solver = getattr(diffrax, solver_fn)(**solver_args)

    # Level of tolerance of the brownian motion
    b_tol = params_solver.get('brownian_tol', dt0-1e-6)

    # Define the integration scheme given the drift and diffusion terms
    def sde_solve(ts, z0, rng_brownian, drift_fn, diffusion_fn):
        """Solve a stochastic differential equation given the drift
           diffusion functions

        Args:
            ts (jnp.array): The integration time of the SDEs
            z0 (jnp.array): The initial state of the system
            rng_brownian (jnp.array): A seed to generate brownian motion
            drift_fn (function): The drift function of the SDE
            diffusion_fn (function): The diffusion function of the SDEs

        Returns:
            function: A function to solve the SDEs
        """
        # Create the ODE term corresponding to the drift function
        ode_term = diffrax.ODETerm(drift_fn)

        # Use virtual brownian tree if backprop is enabled
        brownian_motion = diffrax.VirtualBrownianTree(
                            t0=ts[0], t1=ts[-1], tol=b_tol, shape=(z0.shape[-1],),
                            key=rng_brownian
                            )

        # [TODO] better than diagonal diffusion term
        # Create the diffusion term  -> Assume diagonal diffusion function
        control_term = diffrax.WeaklyDiagonalControlTerm(
                                diffusion_fn, brownian_motion)

        # From diffrax, we need to differentiate sde-specific solvers
        solv_term = (ode_term, control_term) if specific_sde else \
                        diffrax.MultiTerm(ode_term,control_term)

        # Point at which to save the solution
        # The time at which the solutions are saved
        saveat = diffrax.SaveAt(ts=ts)

        # Solve the SDE given the time orizon and initial state
        return diffrax.diffeqsolve(solv_term, solver, t0=ts[0], t1=ts[-1],
                    dt0=dt0, y0=z0, saveat=saveat,
                    stepsize_controller=stepsize_controller,
                    adjoint=adjoint, max_steps=max_steps).ys

    return sde_solve


def linear_interpolation(x, xp, fp):
    """Return a function to perform linear interpolation for a given value x.
       THIS FUNCTION ASSUMES THAT xp IS ALREADY SORTED
    Args:
        xp (array): 1D array of float
        fp (array): a function taken at x -> size(xp) x M
    """
    # This function assumes that xp is already sorted
    i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, xp.shape[0]-1)
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    delta = x - xp[i - 1]
    f = jnp.where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)

    #[TODO Franck]
    # Check ranges ---> Probably is useless to check this
    f = jnp.where(x < xp[0], fp[0], f)
    f = jnp.where(x > xp[-1], fp[-1], f)

    return f

def heun_strat_solver(ts, z0, rng_brownian, drift_fn, diffusion_fn):
    # Build the brownian motion for this integration
    # [TODO Franck] Maybe create it in one go instead of every call
    dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, z0.shape[0]))
    # Define the body loop
    def heun_step(cur_y, dnoise):
        _t, _y = cur_y
        _next_t, _dw = dnoise
        _dt = _next_t - _t
        drift_t = drift_fn(_t, _y)
        diff_t = diffusion_fn(_t, _y)
        sqr_dt = jnp.sqrt(_dt) * _dw
        _ybar = _y + drift_t * _dt + diff_t * sqr_dt
        _drift_t = drift_fn(_next_t, _ybar)
        _diff_t = diffusion_fn(_next_t, _ybar)
        _ynext = _y + 0.5 * (drift_t + _drift_t) * _dt + 0.5 * (diff_t + _diff_t) * sqr_dt
        return (_next_t, _ynext), _ynext
    carry_init = (ts[0], z0)
    xs = (ts[1:], dw)
    _, yevol = jax.lax.scan(heun_step, carry_init, xs)
    return jnp.concatenate((z0[None], yevol))
