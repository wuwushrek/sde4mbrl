import jax
import jax.numpy as jnp

def heun_strat_solver_uarr(ts, x0, us, rng_brownian, drift_fn, diffusion_fn):
    """Implement Heun method for Stratonovich SDEs

    Args:
        ts (TYPE): The time indexes at which the integration is one
        z0 (TYPE): The initial state of the solver
        us (TYPE): The control input given as a 2D array similar to ts
        rng_brownian (TYPE): A random key generator for the brownian noise
        drift_fn (TYPE): The drift function of the dynamics
        diffusion_fn (TYPE): The diffusion function of the dynamics

    Returns:
        TYPE: The evolution of x at each time indexes
    """
    # Check the dimension of the problem
    assert (ts.shape[0] == us.shape[0] or ts.shape[0] == us.shape[0]+1) \
            and x0.ndim == 1 and rng_brownian.ndim == 1,\
            "Dimension mismatch on the input of the sde solver"

    # Case the control given is not a two dimensional array
    if us.ndim == 1:
        us = us[None]

    # Build the brownian motion for this integration
    # [TODO Franck] Maybe create it in one go instead of every call
    dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, x0.shape[0]))

    # Define the body loop
    def heun_step(cur_x, extra):
        """ One step heun method for Stratonovich integrals
        """
        _t, _x = cur_x
        _next_t, _dw, _u = extra
        _dt = _next_t - _t
        drift_t = drift_fn(_t, _x, _u)
        diff_t = diffusion_fn(_t, _x)
        sqr_dt = jnp.sqrt(_dt) * _dw
        _xbar = _x + drift_t * _dt + diff_t * sqr_dt
        _drift_t = drift_fn(_next_t, _xbar, _u)
        _diff_t = diffusion_fn(_next_t, _xbar)
        _xnext = _x + 0.5 * (drift_t + _drift_t) * _dt + 0.5 * (diff_t + _diff_t) * sqr_dt
        return (_next_t, _xnext), _xnext

    carry_init = (ts[0], x0)
    xs = (ts[1:], dw, us[:dw.shape[0]])
    _, yevol = jax.lax.scan(heun_step, carry_init, xs)

    return jnp.concatenate((x0[None], yevol))

def heun_strat_solver_ufun(ts, x0, us, rng_brownian, drift_fn, diffusion_fn):
    """Implement Heun method for Stratonovich SDEs -> u is a function of time and x here
        and not an ndarray

    Args:
        ts (TYPE): The time indexes at which the integration is one
        z0 (TYPE): The initial state of the solver
        us (TYPE): The control input given as a 2D array similar to ts
        rng_brownian (TYPE): A random key generator for the brownian noise
        drift_fn (TYPE): The drift function of the dynamics
        diffusion_fn (TYPE): The diffusion function of the dynamics

    Returns:
        TYPE: The evolution of x at each time indexes
    """
    assert not hasattr(us, 'ndim'), 'us should be a function of time here!'

    # Build the brownian motion for this integration
    # [TODO Franck] Maybe create it in one go instead of every call
    dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, x0.shape[0]))

    # Define the body loop
    def heun_step(cur_x, extra):
        """ One step heun method for Stratonovich integrals
        """
        _t, _x = cur_x
        _next_t, _dw = extra
        _u = us(_t, _x)
        _dt = _next_t - _t
        drift_t = drift_fn(_t, _x, _u)
        diff_t = diffusion_fn(_t, _x)
        sqr_dt = jnp.sqrt(_dt) * _dw
        _xbar = _x + drift_t * _dt + diff_t * sqr_dt
        # _unext = us(_next_t, _xbar)
        _drift_t = drift_fn(_next_t, _xbar, _u)
        _diff_t = diffusion_fn(_next_t, _xbar)
        _xnext = _x + 0.5 * (drift_t + _drift_t) * _dt + 0.5 * (diff_t + _diff_t) * sqr_dt
        # _unext = us(_next_t, _xnext)
        return (_next_t, _xnext), (_xnext, _u)

    carry_init = (ts[0], x0)
    xs = (ts[1:], dw)
    _, (yevol, uevol) = jax.lax.scan(heun_step, carry_init, xs)

    return jnp.concatenate((x0[None], yevol)), uevol

def stratonovich_heun(ts, x0, us, rng_brownian, drift_fn, diffusion_f):
    """ A wrapper function for both the case where us is a function or
        a set of control inputs
    """
    solfn = heun_strat_solver_uarr if hasattr(us, 'ndim') else heun_strat_solver_ufun
    return solfn(ts, x0, us, rng_brownian, drift_fn, diffusion_f)

def milstein_strat_solver_uarr(ts, x0, us, rng_brownian, drift_fn, diffusion_fn):
    """Implement Milstein method for Stratonovich SDEs

    Args:
        ts (TYPE): The time indexes at which the integration is one
        z0 (TYPE): The initial state of the solver
        us (TYPE): The control input given as a 2D array similar to ts
        rng_brownian (TYPE): A random key generator for the brownian noise
        drift_fn (TYPE): The drift function of the dynamics
        diffusion_fn (TYPE): The diffusion function of the dynamics

    Returns:
        TYPE: The evolution of x at each time indexes
    """
    # Check teh dimension of the problem
    assert (ts.shape[0] == us.shape[0] or ts.shape[0] == us.shape[0]+1) and x0.ndim == 1 and rng_brownian.ndim == 1,\
            "DImension mismatch on the input of the sde solver {}, {}, {}, {}".format(ts.shape, us.shape, x0.ndim, rng_brownian.ndim)

    # Case the control given is not a two dimensional array
    if us.ndim == 1:
        us = us[None]

    # Build the brownian motion for this integration
    # [TODO Franck] Maybe create it in one go instead of every call
    dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, x0.shape[0]))

    # Define the body loop
    def milstein_step(cur_x, extra):
        """ One step heun method for Stratonovich integrals
        """
        _t, _x, = cur_x
        _next_t, _dw, _u = extra
        _dt = _next_t - _t
        # Drift and diffusion at current time and current state
        drift_t = drift_fn(_t, _x, _u)
        diff_t = diffusion_fn(_t, _x)
        # Store sqrt (dt)
        sqr_v = jnp.sqrt(_dt)
        # Store sqrt(_dt) * nooise
        sqr_dt = sqr_v * _dw
        # COmpute _xbar by storing the first two terms which are used later
        _xbar_temp = _x + drift_t * _dt
        _xbar =  _xbar_temp + diff_t * sqr_v
        # DIffusion at _xbar
        _diff_t = diffusion_fn(_t, _xbar)
        # Now the next state can be computed
        _xnext = _xbar_temp + diff_t * sqr_dt + (0.5/sqr_v) * (_diff_t - diff_t) * jnp.square(_dw)
        return (_next_t, _xnext), _xnext

    carry_init = (ts[0], x0)
    xs = (ts[1:], dw, us[:dw.shape[0]])
    _, yevol = jax.lax.scan(milstein_step, carry_init, xs)

    return jnp.concatenate((x0[None], yevol))

def milstein_strat_solver_ufun(ts, x0, us, rng_brownian, drift_fn, diffusion_fn):
    """Implement Milstein method for Stratonovich SDEs -> u is given as a function

    Args:
        ts (TYPE): The time indexes at which the integration is one
        z0 (TYPE): The initial state of the solver
        us (TYPE): The control input given as a function of t and x
        rng_brownian (TYPE): A random key generator for the brownian noise
        drift_fn (TYPE): The drift function of the dynamics
        diffusion_fn (TYPE): The diffusion function of the dynamics

    Returns:
        TYPE: The evolution of x at each time indexes
    """
    assert not hasattr(us, 'ndim'), 'us should be a function of time here!'

    # Build the brownian motion for this integration
    # [TODO Franck] Maybe create it in one go instead of every call
    dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, x0.shape[0]))

    # Define the body loop
    def milstein_step(cur_x, extra):
        """ One step heun method for Stratonovich integrals
        """
        _t, _x, = cur_x
        _next_t, _dw = extra
        _u = us(_t, _x)
        _dt = _next_t - _t
        # Drift and diffusion at current time and current state
        drift_t = drift_fn(_t, _x, _u)
        diff_t = diffusion_fn(_t, _x)
        # Store sqrt (dt)
        sqr_v = jnp.sqrt(_dt)
        # Store sqrt(_dt) * nooise
        sqr_dt = sqr_v * _dw
        # COmpute _xbar by storing the first two terms which are used later
        _xbar_temp = _x + drift_t * _dt
        _xbar =  _xbar_temp + diff_t * sqr_v
        # DIffusion at _xbar
        _diff_t = diffusion_fn(_t, _xbar)
        # Now the next state can be computed
        _xnext = _xbar_temp + diff_t * sqr_dt + (0.5/sqr_v) * (_diff_t - diff_t) * jnp.square(_dw)
        return (_next_t, _xnext), (_xnext, _u)

    carry_init = (ts[0], x0)
    xs = (ts[1:], dw)
    _, (yevol, uevol) = jax.lax.scan(milstein_step, carry_init, xs)

    return jnp.concatenate((x0[None], yevol)), uevol

def stratonovich_milstein(ts, x0, us, rng_brownian, drift_fn, diffusion_f):
    """ A wrapper function for both the case where us is a function or
        a set of control inputs
    """
    solfn = milstein_strat_solver_uarr if hasattr(us, 'ndim') else milstein_strat_solver_ufun
    return solfn(ts, x0, us, rng_brownian, drift_fn, diffusion_f)

# A dictionary to map string keys to sde solver functions
sde_solver_name ={
    'stratonovich_heun': stratonovich_heun,
    'stratonovich_milstein': stratonovich_milstein
}



# def differentiable_sde_solver(params_solver={}):
#     """Construct and return a function that solves sdes and that is
#         differentiable in forward and reverse mode

#     Args:
#         params_solver (dict): A dictionary containing solver specific parameters.
#             The key and value of the dictionary are described below:
#           - max_steps=4096 : The maximum steps when discretizing the SDE

#           - init_step=0.1 : The initial time step of the numerical integrator

#           - stepsize_controller=diffrax.ConstantStepSize(.)
#                     or for adaptive scheme diffrax.PIDController(rtol,atol, ...)

#           - adjoint=diffrax.RecursiveCheckpointAdjoint() or diffrax.NoAdjoint()

#           - solver=diffrax.ReversibleHeun() : Define the solver to use for
#                                               numerical integration

#           - specific_sde=False : Boolean specifying if solver is specific
#                                 (see Diffrax documentation)

#           - brownian_tol = init_step/2 : Specify the threshold in the virtual
#                                          brownian tree when using fixed/adaptive time step
#                                          integrator. For fixed time step, a value slightly
#                                          lower than the time step is enough. For adaptive
#                                          time step, it should be small but the complexity
#                                          (computation) increases too.
#                                          https://docs.kidger.site/diffrax/api/brownian/#diffrax.VirtualBrownianTree

#           - ts = jnp.array([t0, t1, .., tn]) : Time indexes to save the solution
#                                        of the integration
#     """
#     # Extract the parameters to construct the numerical differentiator
#     specific_sde = params_solver.get('specific_sde', False)

#     # Extract the maximum number of steps to solve the problem
#     max_steps = params_solver.get('max_steps', 4096)

#     # Initial time step
#     dt0 = params_solver.get('init_step', 0.001)

#     # Pick the step size controller
#     if 'stepsize_controller' not in params_solver:
#         stepsize_controller = diffrax.ConstantStepSize()
#     else:
#         stepsize_fn = params_solver['stepsize_controller'].get('name', 'ConstantStepSize')
#         stepsize_args = params_solver['stepsize_controller'].get('params', {})
#         stepsize_controller = getattr(diffrax, stepsize_fn)(**stepsize_args)

#     # Specify if adjoint method is enable for backpropagtion
#     if 'adjoint' not in params_solver:
#         adjoint = diffrax.RecursiveCheckpointAdjoint()
#     else:
#         adjoint_fn = params_solver['adjoint'].get('name', 'RecursiveCheckpointAdjoint')
#         adjoint_args = params_solver['adjoint'].get('params', {})
#         adjoint = getattr(diffrax, adjoint_fn)(**adjoint_args)

#     # Construct the SDE solver
#     if 'solver' not in params_solver:
#         solver = diffrax.ReversibleHeun()
#     else:
#         solver_fn = params_solver['solver'].get('name', 'ReversibleHeun')
#         solver_args = params_solver['solver'].get('params', {})
#         solver = getattr(diffrax, solver_fn)(**solver_args)

#     # Level of tolerance of the brownian motion
#     b_tol = params_solver.get('brownian_tol', dt0-1e-6)

#     # Define the integration scheme given the drift and diffusion terms
#     def sde_solve(ts, z0, rng_brownian, drift_fn, diffusion_fn):
#         """Solve a stochastic differential equation given the drift
#            diffusion functions

#         Args:
#             ts (jnp.array): The integration time of the SDEs
#             z0 (jnp.array): The initial state of the system
#             rng_brownian (jnp.array): A seed to generate brownian motion
#             drift_fn (function): The drift function of the SDE
#             diffusion_fn (function): The diffusion function of the SDEs

#         Returns:
#             function: A function to solve the SDEs
#         """
#         # Create the ODE term corresponding to the drift function
#         ode_term = diffrax.ODETerm(drift_fn)

#         # Use virtual brownian tree if backprop is enabled
#         brownian_motion = diffrax.VirtualBrownianTree(
#                             t0=ts[0], t1=ts[-1], tol=b_tol, shape=(z0.shape[-1],),
#                             key=rng_brownian
#                             )

#         # [TODO] better than diagonal diffusion term
#         # Create the diffusion term  -> Assume diagonal diffusion function
#         control_term = diffrax.WeaklyDiagonalControlTerm(
#                                 diffusion_fn, brownian_motion)

#         # From diffrax, we need to differentiate sde-specific solvers
#         solv_term = (ode_term, control_term) if specific_sde else \
#                         diffrax.MultiTerm(ode_term,control_term)

#         # Point at which to save the solution
#         # The time at which the solutions are saved
#         saveat = diffrax.SaveAt(ts=ts)

#         # Solve the SDE given the time orizon and initial state
#         return diffrax.diffeqsolve(solv_term, solver, t0=ts[0], t1=ts[-1],
#                     dt0=dt0, y0=z0, saveat=saveat,
#                     stepsize_controller=stepsize_controller,
#                     adjoint=adjoint, max_steps=max_steps).ys

#     return sde_solve


# def linear_interpolation(x, xp, fp):
#     """Return a function to perform linear interpolation for a given value x.
#        THIS FUNCTION ASSUMES THAT xp IS ALREADY SORTED
#     Args:
#         xp (array): 1D array of float
#         fp (array): a function taken at x -> size(xp) x M
#     """
#     # This function assumes that xp is already sorted
#     i = jnp.clip(jnp.searchsorted(xp, x, side='right'), 1, xp.shape[0]-1)
#     df = fp[i] - fp[i - 1]
#     dx = xp[i] - xp[i - 1]
#     delta = x - xp[i - 1]
#     f = jnp.where((dx == 0), fp[i], fp[i - 1] + (delta / dx) * df)

#     #[TODO Franck]
#     # Check ranges ---> Probably is useless to check this
#     f = jnp.where(x < xp[0], fp[0], f)
#     f = jnp.where(x > xp[-1], fp[-1], f)

#     return f

# def heun_strat_solver(ts, z0, rng_brownian, drift_fn, diffusion_fn):
#     # Build the brownian motion for this integration
#     # [TODO Franck] Maybe create it in one go instead of every call
#     dw = jax.random.normal(key=rng_brownian, shape=(ts.shape[0]-1, z0.shape[0]))
#     # Define the body loop
#     def heun_step(cur_y, dnoise):
#         _t, _y = cur_y
#         _next_t, _dw = dnoise
#         _dt = _next_t - _t
#         drift_t = drift_fn(_t, _y)
#         diff_t = diffusion_fn(_t, _y)
#         sqr_dt = jnp.sqrt(_dt) * _dw
#         _ybar = _y + drift_t * _dt + diff_t * sqr_dt
#         _drift_t = drift_fn(_next_t, _ybar)
#         _diff_t = diffusion_fn(_next_t, _ybar)
#         _ynext = _y + 0.5 * (drift_t + _drift_t) * _dt + 0.5 * (diff_t + _diff_t) * sqr_dt
#         return (_next_t, _ynext), _ynext
#     carry_init = (ts[0], z0)
#     xs = (ts[1:], dw)
#     _, yevol = jax.lax.scan(heun_step, carry_init, xs)
#     return jnp.concatenate((z0[None], yevol))
