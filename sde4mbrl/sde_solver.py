import jax
import jax.numpy as jnp

# All of functions here do not include time dependencies in the SDE
# A way to incorporate time dependency is by augmenting the state with a time
# factor. The choice is rather simple as in most dynamical systems of interest,
# the time does not appear in the dynamics.

# def heun_strat_solver_uarr(time_step, x0, us, rng_brownian, drift_fn, diffusion_fn, 
#                             projection_fn=None, extra_scan_args=None):
#     """Implement Heun method for Stratonovich SDEs

#     Args:
#         ts (TYPE): The time indexes at which the integration is one
#         z0 (TYPE): The initial state of the solver
#         us (TYPE): The control input given as a 2D array similar to ts
#         rng_brownian (TYPE): A random key generator for the brownian noise
#         drift_fn (TYPE): The drift function of the dynamics
#         diffusion_fn (TYPE): The diffusion function of the dynamics

#     Returns:
#         TYPE: The evolution of x at each time indexes
#     """
#     # Case the control given is not a two dimensional array
#     if us.ndim == 1:
#         us = us[None]

#     # Check the dimension properties
#     assert time_step.shape[0] == us.shape[0]  and \
#             x0.ndim == 1 and rng_brownian.ndim == 1,\
#             "Dimension mismatch on the input of the sde solver"

#     num_step = time_step.shape[0]

#     # Build the brownian motion for this integration
#     # [TODO Franck] Maybe create it in one go instead of every call
#     dw = jnp.sqrt(time_step[:,None]) * jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

#     # Define the body loop
#     def heun_step(_x, extra):
#         """ One step heun method for Stratonovich integrals
#         """
#         _dw, _u, dt, _e_args = extra
#         drift_t = drift_fn(_x, _u, extra_args=_e_args)
#         diff_t = diffusion_fn(_x, extra_args=_e_args)
#         _xbar = _x + drift_t * dt + diff_t * _dw
#         _drift_t = drift_fn(_xbar, _u, extra_args=_e_args)
#         _diff_t = diffusion_fn(_xbar, extra_args=_e_args)
#         _xnext = _x + 0.5 * (drift_t + _drift_t) * dt + 0.5 * (diff_t + _diff_t) * _dw
#         _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext
#         return _xnext, _xnext

#     carry_init = x0
#     xs = (dw, us, time_step, time_step) if extra_scan_args is None else (dw, us, time_step, extra_scan_args)
#     _, yevol = jax.lax.scan(heun_step, carry_init, xs)

#     return jnp.concatenate((x0[None], yevol))

# def heun_strat_solver_ufun(time_step, x0, us, rng_brownian, drift_fn, diffusion_fn, 
#                             projection_fn=None, extra_scan_args=None):
#     """Implement Heun method for Stratonovich SDEs -> u is a function of time and x here
#         and not an ndarray

#     Args:
#         ts (TYPE): The time indexes at which the integration is one
#         z0 (TYPE): The initial state of the solver
#         us (TYPE): The control input given as a 2D array similar to ts
#         rng_brownian (TYPE): A random key generator for the brownian noise
#         drift_fn (TYPE): The drift function of the dynamics
#         diffusion_fn (TYPE): The diffusion function of the dynamics

#     Returns:
#         TYPE: The evolution of x at each time indexes
#     """
#     # Check the dimension properties
#     assert x0.ndim == 1 and rng_brownian.ndim == 1,\
#             "Dimension mismatch on the input of the sde solver"

#     num_step = time_step.shape[0]

#     # Build the brownian motion for this integration
#     # [TODO Franck] Maybe create it in one go instead of every call
#     dw = jnp.sqrt(time_step[:,None]) * jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

#     # Define the body loop
#     def heun_step(_x, extra):
#         """ One step heun method for Stratonovich integrals
#         """
#         _dw, dt, _e_extra = extra
#         _u = us(_x)
#         drift_t = drift_fn(_x, _u, extra_args=_e_extra)
#         diff_t = diffusion_fn(_x, extra_args=_e_extra)
#         _xbar = _x + drift_t * dt + diff_t * _dw
#         # _unext = us(_next_t, _xbar)
#         _drift_t = drift_fn(_xbar, _u, extra_args=_e_extra)
#         _diff_t = diffusion_fn(_xbar, extra_args=_e_extra)
#         _xnext = _x + 0.5 * (drift_t + _drift_t) * dt + 0.5 * (diff_t + _diff_t) * _dw
#         _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext
#         return _xnext, (_xnext, _u)

#     carry_init = x0
#     xs = (dw, time_step, time_step) if extra_scan_args is None else (dw, time_step, extra_scan_args)
#     _, (yevol, uevol) = jax.lax.scan(heun_step, carry_init, xs)

#     return jnp.concatenate((x0[None], yevol)), uevol

# def stratonovich_heun(time_step, x0, us, rng_brownian, drift_fn, diffusion_f, projection_fn=None, extra_scan_args=None):
#     """ A wrapper function for both the case where us is a function or
#         a set of control inputs
#     """
#     solfn = heun_strat_solver_uarr if hasattr(us, 'ndim') else heun_strat_solver_ufun
#     return solfn(time_step, x0, us, rng_brownian, drift_fn, diffusion_f, projection_fn, extra_scan_args)

# def milstein_strat_solver_uarr(time_step, x0, us, rng_brownian, drift_fn, diffusion_fn, 
#                             projection_fn=None, extra_scan_args=None):
#     """Implement Milstein method for Stratonovich SDEs

#     Args:
#         ts (TYPE): The time indexes at which the integration is one
#         z0 (TYPE): The initial state of the solver
#         us (TYPE): The control input given as a 2D array similar to ts
#         rng_brownian (TYPE): A random key generator for the brownian noise
#         drift_fn (TYPE): The drift function of the dynamics
#         diffusion_fn (TYPE): The diffusion function of the dynamics

#     Returns:
#         TYPE: The evolution of x at each time indexes
#     """
#     # Case the control given is not a two dimensional array
#     if us.ndim == 1:
#         us = us[None]

#     # Check the dimension properties
#     assert time_step.shape[0] == us.shape[0] and \
#             x0.ndim == 1 and rng_brownian.ndim == 1,\
#             "Dimension mismatch on the input of the sde solver"

#     num_step = time_step.shape[0]

#     # # Reduce the control input size if necessary
#     # if us.shape[0] == num_step + 1:
#     #     us = us[:-1]

#     # Build the brownian motion for this integration
#     # [TODO Franck] Maybe create it in one go instead of every call
#     dw = jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))
#     # dw = jnp.zeros((num_step, x0.shape[0]))

#     # Store sqrt (dt)
#     _sqrt_dt = jnp.sqrt(time_step)

#     # Define the body loop
#     def milstein_step(_x, extra):
#         """ One step heun method for Stratonovich integrals
#         """
#         _dw, _u, dt, sqrt_dt, _e_args = extra
#         # Drift and diffusion at current time and current state
#         drift_t = drift_fn(_x, _u, extra_args=_e_args)
#         diff_t = diffusion_fn(_x, extra_args=_e_args)
#         # Store sqrt(_dt) * nooise
#         sqrdt_dw = sqrt_dt * _dw
#         # COmpute _xbar by storing the first two terms which are used later
#         _xbar_temp = _x + drift_t * dt
#         _xbar =  _xbar_temp + diff_t * sqrt_dt
#         # DIffusion at _xbar
#         _diff_t = diffusion_fn(_xbar, extra_args=_e_args)
#         # Now the next state can be computed
#         _xnext = _xbar_temp + diff_t * sqrdt_dw + (0.5/sqrt_dt) * (_diff_t - diff_t) * jnp.square(_dw)
#         _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext
#         return _xnext, _xnext

#     carry_init = x0
#     # A hack here to use _sqrt dt as the content. Maybe use None instead
#     xs = (dw, us, time_step,_sqrt_dt,_sqrt_dt) if extra_scan_args is None else (dw, us, time_step,_sqrt_dt, extra_scan_args)
#     _, yevol = jax.lax.scan(milstein_step, carry_init, xs)

#     return jnp.concatenate((x0[None], yevol))

# def milstein_strat_solver_ufun(time_step, x0, us, rng_brownian, drift_fn, diffusion_fn, 
#                                 projection_fn=None, extra_scan_args=None):
#     """Implement Milstein method for Stratonovich SDEs -> u is given as a function

#     Args:
#         ts (TYPE): The time indexes at which the integration is one
#         z0 (TYPE): The initial state of the solver
#         us (TYPE): The control input given as a function of t and x
#         rng_brownian (TYPE): A random key generator for the brownian noise
#         drift_fn (TYPE): The drift function of the dynamics
#         diffusion_fn (TYPE): The diffusion function of the dynamics

#     Returns:
#         TYPE: The evolution of x at each time indexes
#     """
#     # Check the dimension properties
#     assert x0.ndim == 1 and rng_brownian.ndim == 1,\
#             "Dimension mismatch on the input of the sde solver"

#     num_step = time_step.shape[0]

#     # Build the brownian motion for this integration
#     # [TODO Franck] Maybe create it in one go instead of every call
#     dw = jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

#     # Store sqrt (dt)
#     _sqrt_dt = jnp.sqrt(time_step)

#     # Define the body loop
#     def milstein_step(_x, extra):
#         """ One step heun method for Stratonovich integrals
#         """
#         _dw, dt, sqrt_dt, _e_args = extra
#         _u = us(_x)
#         # Drift and diffusion at current time and current state
#         drift_t = drift_fn(_x, _u, extra_args=_e_args)
#         diff_t = diffusion_fn(_x, extra_args=_e_args)
#         # Store sqrt(_dt) * nooise
#         sqrdt_dw = sqrt_dt * _dw
#         # COmpute _xbar by storing the first two terms which are used later
#         _xbar_temp = _x + drift_t * dt
#         _xbar =  _xbar_temp + diff_t * sqrt_dt
#         # DIffusion at _xbar
#         _diff_t = diffusion_fn(_xbar, extra_args=_e_args)
#         # Now the next state can be computed
#         _xnext = _xbar_temp + diff_t * sqrdt_dw + (0.5/sqrt_dt) * (_diff_t - diff_t) * jnp.square(_dw)
#         _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext
#         return _xnext, (_xnext, _u)

#     carry_init = x0
#     xs = (dw, time_step, _sqrt_dt, _sqrt_dt) if extra_scan_args is None else (dw, time_step, _sqrt_dt, extra_scan_args)

#     _, (yevol, uevol) = jax.lax.scan(milstein_step, carry_init, xs)

#     return jnp.concatenate((x0[None], yevol)), uevol

# def stratonovich_milstein(time_step, x0, us, rng_brownian, drift_fn, diffusion_f, 
#                             projection_fn=None, extra_scan_args=None):
#     """ A wrapper function for both the case where us is a function or
#         a set of control inputs
#     """
#     solfn = milstein_strat_solver_uarr if hasattr(us, 'ndim') else milstein_strat_solver_ufun
#     return solfn(time_step, x0, us, rng_brownian, drift_fn, diffusion_f, projection_fn, extra_scan_args)

def euler_maruyama(obs2state_fn, time_step, y0, us, rng_brownian, drift_fn, diffusion_fn, 
                                projection_fn=None, extra_scan_args=None):
    """Implement Euler-Maruyama method for Ito SDEs -> us can be a function or a sequence of control inputs
        Args:
            obs2state_fn (tuple): A tuple of functions to convert the observation to the state and vice versa.
                                  The first function is the conversion from observation to state and the second
                                    function is the conversion from state to observation
                                    obs2state_fn[0] : obs, rng -> state
                                    obs2state_fn[1] : state, rng -> obs
            time_step (TYPE): The time indexes at which the integration is done
            y0 (TYPE): The initial observation of the solver
            us (TYPE): The control input given as a function of y or as an array
            rng_brownian (TYPE): A random key generator for the brownian noise
            drift_fn (TYPE): The drift function of the dynamics
            diffusion_fn (TYPE): The diffusion function of the dynamics
            projection_fn (TYPE): A projection function to project the state back or into a desired manifold
            extra_scan_args (TYPE): Extra arguments to pass to the scan function
    """
    # Check of us is a function or an array
    if hasattr(us, 'ndim'):
        # If us is of dimension 1 then add an axis
        if us.ndim == 1:
            us = us[None]
        # Check the dimension properties
        assert time_step.shape[0] == us.shape[0], "Dimension mismatch on the input of the sde solver"
    
    # Check the initial observation dimensions
    assert y0.ndim == 1 and rng_brownian.ndim == 1, "Not the right dimension for the initial observation and the random key"

    # The number of steps
    num_step = time_step.shape[0]

    # Split the random key
    rng_brownian, rng_state = jax.random.split(rng_brownian)
    rng_obs = jax.random.split(rng_state, num_step+1)

    # Convert the initial observation to the initial state
    x0 = obs2state_fn[0](y0, rng_obs[0])

    # Build the brownian motion for this integration
    dw  = jax.random.normal(key=rng_brownian, shape=(num_step, x0.shape[0]))

    def euler_step(_stateObs, extra):
        """ One step euler method for Ito integrals
        """
        _rng_obs, _dw, dt, _maybe_u, _e_args = extra
        _x, _y = _stateObs
        
        # Extract the control input
        _u = us(_y) if _maybe_u is None else _maybe_u

        # Drift and diffusion at current time and current state
        drift_t = drift_fn(_x, _u, extra_args=_e_args)
        diff_t = diffusion_fn(_x, _u, extra_args=_e_args)
        
        # Now the next state can be computed
        # TODO [Franck] This is not correct. _dw needs to be multiplied by sqrt(dt). Keep this for reproducibility of the paper results that used euler_maruyama
        # But the impact of no dt can be removed by increasing or decreasing the diffusion coefficient by the adequate factor
        _xnext = _x + drift_t * dt + diff_t * _dw

        # Do the projection
        _xnext = projection_fn(_xnext) if projection_fn is not None else _xnext

        # Compute the next observation
        _ynext = obs2state_fn[1](_xnext, _rng_obs)

        return (_xnext, _ynext), (_xnext, _ynext, _u)
    
    # Define the carry and the scan arguments
    carry_init = (x0, y0)
    xs = (rng_obs[1:], dw, time_step, None if not hasattr(us, 'ndim') else us, None if extra_scan_args is None else extra_scan_args)

    # Do the scan
    _, (xevol, yevol, uevol) = jax.lax.scan(euler_step, carry_init, xs)
    return jnp.concatenate((x0[None], xevol)), jnp.concatenate((y0[None], yevol)), uevol


# A dictionary to map string keys to sde solver functions
sde_solver_name ={
    # 'stratonovich_heun': stratonovich_heun,
    # 'stratonovich_milstein': stratonovich_milstein,
    'euler_maruyama': euler_maruyama
}