import jax
import jax.numpy as jnp

from typing import NamedTuple

# Implementation inspired from and Armijo line search paper
# Convergence Analysis of Proximal Gradient with Momentum for Nonconvex
# Optimization

# A Function to estimate the suboptimality error
opt_stop_criteria = lambda grad_val: jnp.sum(jnp.square(grad_val))


def init_opt_state(opt_var, optimizer_params):
    """Create an  instance of APGState given

    Args:
        opt_var (TYPE): Description
        optimizer_params (TYPE): Description
    """
    # Extract the initial stepsize
    init_step_size = jnp.array(optimizer_params['linesearch']['init_stepsize']\
                        if 'linesearch' in optimizer_params \
                        else optimizer_params['stepsize'])

    # The momentum, stepsize, and avg_stepsize
    momentum = jnp.array(optimizer_params.get('beta_init', 0.25))

    # Dummy initialization given opt_var
    # [TODO] Better to have infinity for initial cost I guess
    opt_state_init = APGState(num_steps = jnp.array(1),
                    num_iter_no_improv = jnp.array(0),
                    momentum = momentum,
                    avg_momentum = momentum,
                    # num_linesearch = jnp.array(1),
                    avg_linesearch = jnp.array(1.),
                    stepsize = init_step_size,
                    # This can be used to warm-up online next mpc calls
                    avg_stepsize = init_step_size,
                    opt_cost = jnp.array(0.),
                    x_opt = opt_var,
                    curr_cost = jnp.array(0.),
                    init_cost = jnp.array(0.),
                    grad_sqr = jnp.array(0.),
                    not_done = jnp.array(True),
                    xk = opt_var,
                    yk = opt_var,
                    grad_yk = jnp.zeros_like(opt_var))
    return opt_state_init


# [TODO] Properly initialize the slack variables based on the
def init_apg(x0, cost_fn, optimizer_params, proximal_fn=None,
             momentum=None, stepsize=None):
    """Initialize the APG algorithm
    This function will evaluate the cost function at x0 and its gradient
    It is better to use it in a function that is going to be jit-compiled

    Args:
        x0 (TYPE): The initial parameters of the optimization problem
        cost_fn (TYPE): The cost function to minimize
        optimizer_params (TYPE): The parameters for the optimizer
        proximal_fn (None, optional): A function to compute the proximal or projection
                    when the opt variables are subject to contraints. This code
                    assumes only projection on a box constraint for now

    Returns:
        TYPE: an APGState

    """
    # Extract the initial stepsize
    init_step_size = jnp.array(optimizer_params['linesearch']['init_stepsize']\
                        if 'linesearch' in optimizer_params \
                        else optimizer_params['stepsize'])

    # [TODO Franck] Better to project also x0 if proximal function is given
    # This help making sure that we start in the right constrained space
    if proximal_fn is not None:
        x0 = proximal_fn(x0, stepsize=init_step_size)

    # Evaluate the function and its gradient at the current opt variable
    cost_x0, grad_x0 = jax.value_and_grad(cost_fn)(x0)
    # cost_x0, grad_x0 = cost_fn(x0), x0
    grad_x0_nsqr = opt_stop_criteria(grad_x0)

    if momentum is None:
        momentum = jnp.array(optimizer_params.get('beta_init', 0.25))

    if stepsize is None:
        stepsize = init_step_size

    avg_stepsize = stepsize

    opt_state_init = APGState(num_steps = jnp.array(1),
                    num_iter_no_improv = jnp.array(0),
                    momentum = momentum,
                    avg_momentum = momentum,
                    # num_linesearch = jnp.array(1),
                    avg_linesearch = jnp.array(1.),
                    stepsize = stepsize,
                    avg_stepsize = avg_stepsize, # This can be used to warm-up online next mpc calls
                    # init_cost = jnp.abs(cost_x0), # This is used as a scale for terminaison criterion
                    opt_cost = cost_x0,
                    x_opt = x0,
                    curr_cost = cost_x0,
                    init_cost = cost_x0,
                    grad_sqr = grad_x0_nsqr,
                    not_done = jnp.array(True),
                    xk = x0,
                    yk = x0,
                    grad_yk = grad_x0)

    return opt_state_init

def apg(optim_state, cost_fn, optimizer_params, proximal_fn=None):
    """Compute a one step Accelerated proximal gradient descent given a cost function
        and the optimization parameters such as learning rate, momentum, etc..

    Args:
        cost_fn (TYPE): The cost function to optimize. As
        optim_state (TYPE): The initial state of the optimizer (an instance)
        optimizer_params (TYPE): The parameters of the optimization problems
            - tol: The terminaison criteria based on the norm of gradient
            - max_iter: The maximum of gradient calculations
            - stepsize: The learning rate. if lr <=0 line search is used based on max_stepsize
            - linesearch: A dictionary with the keys:
                - max_stepsize: upper bound on the learning rate
                - coef: ``1-agressiveness`` used for checking the wolfe or improvement condition
                    coef < 0.5 provides a guarantess with Lipschitz constant. low coef -> higher step size
                - decrease_factor: factor to decrease the learning rate.
                - increase_factor: factor to increase the learning rate.
                - maxls: maximum number of steps when doing the backtracking line search
                - reset_option: 'conservative' or increase. The reset strategy at each iteration
                    "conservative": re-use previous stepsize, producing a non increasing
                    sequence of stepsizes. Slow convergence.
                    "increase": attempt to re-use previous stepsize multiplied by
                    increase_factor. Cheap and efficient heuristic.
            - moment_scale: An adaptive coefficient to scale the momentum (it might be unused)
                 if moment_scale <= 0, standard momentum is used. moment_scale in (0,1)
            - beta_init: The initial momentum coeffiicent value -> default is 0.25
        proximal_fn (None, optional): A function to compute the proximal or projection
                    when the opt variables are subject to contraints. This code
                    assumes only projection on a box constraint for now
    """
    assert isinstance(optim_state, APGState), "The input optim_state should be an instance of APG"

    # # Define the loop condition or stopping criteria
    # no_stop = lambda grad_val_crit: jnp.array(grad_val_crit > optimizer_params['tol'])

    # Only iterate at least once -> Reason to enforce the second argument to be true
    #[TODO Franck] Maybe improve it to check a criteria with respect to te current opt_state
    # init_states = optim_state

    # _while_loop(cond_fun, body_fun, init_val, max_iter)
    ret = _while_loop(cond_fun = lambda t: t.not_done,  # check boolean violated
                        body_fun=lambda _opt_s : one_step_apg(_opt_s, cost_fn, optimizer_params, proximal_fn),
                        init_val=optim_state,
                        max_iter=optimizer_params['max_iter']
                    )

    return ret


def one_step_apg(state_ops, cost_fn, optimizer_params, proximal_fn):
    """ Define the body function implementing each iteration of APG
    """
    # Extract the variables and current state of the optimizer
    opt_state = state_ops
    xpast, ycurr, cost_ycurr, grad_ycurr = opt_state.xk, opt_state.yk, \
                        opt_state.curr_cost, opt_state.grad_yk

    if 'linesearch' in optimizer_params:
        # Now perform a line search if requested and do one step improvement on ycurr
        _stepsize = opt_state.stepsize
        # # Extract the number of linesearch during the last iteration
        # last_num_search = opt_state.num_linesearch

        # Reset the step size from the previous iteration
        if optimizer_params['linesearch']['reset_option'] == 'increase':
            # Do not increase if the last search required more than one iteration
            # mult_increase = jnp.where(last_num_search > 1, 1., optimizer_params['linesearch']['increase_factor'])
            mult_increase = optimizer_params['linesearch']['increase_factor']
            _stepsize = _stepsize * mult_increase
            _stepsize = jnp.minimum(_stepsize, optimizer_params['linesearch']['max_stepsize'])

        # Compute the new stepsize, current variable and associated cost
        # We augment the decrease factor with a coefficient that depends
        # on the number of linesearch done in the past iterations
        stepsize, xcurr, cost_xcurr, linesearch_numstep = \
            armijo_line_search(cost_fn, ycurr, cost_ycurr, grad_ycurr, opt_state.grad_sqr,
                                    _stepsize, optimizer_params['linesearch'])
    else:
        # If constant learning rate is given
        stepsize = opt_state.stepsize

        # Compute a gradient step given ycurr
        xcurr = ycurr - stepsize * grad_ycurr

        # cost at xcurr is going to be computed if proximal_fn is given
        # so there is no need to re-compute it here
        # We compute it only when proximal_fn is None
        if proximal_fn is None:
            cost_xcurr = cost_fn(xcurr)

        # Only one iteration was enough to find the step size
        linesearch_numstep = 1

    # Apply proximal projection if it is given and compute the resulting cost function
    if proximal_fn is not None:
        xcurr = proximal_fn(xcurr, stepsize=stepsize)
        cost_xcurr = cost_fn(xcurr)

    # Another auxiliary variable to implement to momentum dynamics
    # vcurr = xcurr + opt_state.momentum * (xcurr - xpast)
    vcurr = xpast + opt_state.momentum * (xcurr - xpast)
    cost_vcurr = cost_fn(vcurr)

    # Check the non-monotony condition and obtain ynext
    xcurr_less_vcurr = cost_xcurr <= cost_vcurr
    ynext = jnp.where(xcurr_less_vcurr, xcurr, vcurr)

    # Compute ynext grad and cost associated to ynext
    cost_ynext = jnp.where(xcurr_less_vcurr, cost_xcurr, cost_vcurr)
    grad_ynext = jax.grad(cost_fn)(ynext)
    grad_ynext_nsqr = opt_stop_criteria(grad_ynext)

    # Update the states of the optimization problem
    if optimizer_params.get('moment_scale', None) is None: # We don't apply scale
        moment_next = opt_state.num_steps / (opt_state.num_steps + 3.)
    else:
        moment_next = jnp.where(xcurr_less_vcurr,
                        optimizer_params['moment_scale']*opt_state.momentum,
                        jnp.minimum(opt_state.momentum/optimizer_params['moment_scale'], 1.)
                    )
    # moment_next = jnp.clip(moment_next, 0., 1.)
    # Check if we are done
    # stop_not_sat_prim = jnp.abs(cost_ynext-cost_ycurr) > optimizer_params['atol'] + optimizer_params['rtol']*jnp.abs(cost_ycurr)
    stop_not_sat = jnp.abs(cost_ynext-opt_state.opt_cost) > optimizer_params['atol'] + optimizer_params['rtol']*jnp.abs(opt_state.opt_cost)
    # stop_not_sat = jnp.logical_or(stop_not_sat, stop_not_sat_prim)

    # Did we improve over the optimal cost
    cost_improv = opt_state.opt_cost > cost_ynext
    new_opt_cost = jnp.where(cost_improv, cost_ynext, opt_state.opt_cost)
    new_x_opt = jnp.where(cost_improv, ynext, opt_state.x_opt)
    new_noimprov = jnp.where(cost_improv, jnp.array(0), opt_state.num_iter_no_improv+1)

    # Update stopping criteria
    stop_not_sat = jnp.logical_and(stop_not_sat, new_noimprov < optimizer_params['max_no_improvement_iter'])

    # Construct the new states of the optimizer
    num_total_step = opt_state.num_steps + 1
    avg_linesearch = (opt_state.num_steps * opt_state.avg_linesearch + linesearch_numstep)/num_total_step
    avg_stepsize = (opt_state.num_steps * opt_state.avg_stepsize + stepsize)/num_total_step
    avg_momentum = (opt_state.num_steps * opt_state.avg_momentum + moment_next)/num_total_step
    opt_state_next = APGState(num_steps = num_total_step,
                                num_iter_no_improv = new_noimprov,
                                momentum = moment_next,
                                avg_momentum = avg_momentum,
                                avg_linesearch = avg_linesearch,
                                # num_linesearch = linesearch_numstep,
                                stepsize = stepsize,
                                avg_stepsize = avg_stepsize,
                                opt_cost = new_opt_cost,
                                curr_cost = cost_ynext,
                                x_opt = new_x_opt,
                                init_cost = opt_state.init_cost,
                                # init_cost = opt_state.init_cost,
                                grad_sqr = grad_ynext_nsqr,
                                not_done = stop_not_sat,
                                xk = xcurr,
                                yk = ynext,
                                grad_yk = grad_ynext
                                )
    return opt_state_next # no_stop(grad_ynext_nsqr)


class APGState(NamedTuple):
    """Named tuple containing state information.
    Attributes:
    num_steps: iteration number
    error: residuals (as gradient) of current estimate
    opt_cost: current value of the cost function
    stepsize: current stepsize
    linesearch_steps: The average number of steps to compute stepsize
    momentum: The momentum coefficient
    """
    num_iter_no_improv: int
    num_steps: int
    momentum: float
    avg_momentum: float
    avg_linesearch: float
    # num_linesearch: int
    avg_stepsize: float
    stepsize: float
    opt_cost: float
    x_opt: jnp.ndarray
    curr_cost: float
    init_cost: float
    # init_cost: float
    grad_sqr: float
    not_done: bool
    xk: jnp.ndarray
    yk: jnp.ndarray
    grad_yk: jnp.ndarray



""" Define more utilities for the Armijo search.
    This set of functions are mainly to have consistent functions to use when compiling with
    tensorflow or tensorflow lite
"""
def update_search(_stepsize, xcurr, grad, cost_fn, decrease_factor, max_stepsize):
    """ Essentialy does a step in the direction of decreasing function value.
        This is done under some limits on how the stepsize varies and the max stepsize allowable
        -----
        Multiply stepsize per factor, return new opt variables and new fun evaluations.
    """
    _stepsize = jnp.minimum(_stepsize * decrease_factor, max_stepsize)
    _xnext = xcurr - _stepsize * grad
    _f_next = cost_fn(_xnext)
    return _stepsize, _xnext, _f_next


def wolfe_cond_violated(stepsize, coef, f_cur, f_next, grad_sqnorm):
    """ Check if wolfe condition is violated or not.
        This condition characterizes sufficient evolution of teh function value
    """
    eps = jnp.finfo(f_next.dtype).eps
    return stepsize * coef * grad_sqnorm > f_cur - f_next + eps


def armijo_line_search(cost_fn, xcurr, f_cur, grad, grad_sqnorm, stepsize, user_params):
    """Perform Armijo (backtracking) Line search from starting opt problem variables,
        learning and gradient at the starting opt variables.

    Args:
        cost_fn: The cost function to optimize
        xcurr: current opt variable to optimize.
        f_cur: value of the cost function at xcurr
        grad: gradient at xcurr.
        stepsize: initial guess for the learning rate.
        user_params: Dictionary
            maxls: maximum number of steps when doing the backtracking line search
            coef: ``1-agressiveness`` used for checking the wolfe condition
            decrease_factor: factor to decrease the learning rate.
            max_stepsize: upper bound on the learning rate

    Returns:
        stepsize: stepsize Armijo line search conditions
        next_params: params after gradient step
        f_next: loss after gradient step
    """
    # Extract the parameters for this function
    coef, decrease_factor, max_stepsize = \
        (user_params[v] for v in ('coef', 'decrease_factor', 'max_stepsize'))

    # compute xnext and the function evaluated at xnext
    xnext = xcurr - stepsize * grad
    f_next = cost_fn(xnext)
    update_stepsize =  lambda _stepsize : update_search(_stepsize, xcurr, grad, cost_fn, decrease_factor, max_stepsize)

    def body_fun(t):
        """ Body function at each line search iteration
        """
        _stepsize, _xnext, _f_next, num_iter, _ = t

        violated = wolfe_cond_violated(_stepsize, coef, f_cur, _f_next, grad_sqnorm)
        _stepsize, _xnext, _f_next = jax.lax.cond(violated, update_stepsize,
                                                lambda _: (_stepsize, _xnext, _f_next),
                                                operand=_stepsize
                                            )
        num_iter += violated
        return _stepsize, _xnext, _f_next, num_iter, violated

    init_val = stepsize, xnext, f_next, jnp.array(1), jnp.array(True)
    ret = _while_loop(cond_fun=lambda t: t[-1],  # check boolean violated
                        body_fun=body_fun,
                        init_val=init_val, max_iter=user_params['maxls'])
    return ret[:-1] # remove boolean

# def armijo_line_search_wlopp(cost_fn, xcurr, f_cur, grad, grad_sqnorm, stepsize, user_params):
#     """ Perform Armijo (backtracking) Line search but instead of using jax cond, scan, or while_loop, we use Python
#         native functions. 
#     """
#     # Extract the parameters for this function
#     coef, decrease_factor, max_stepsize = \
#         (user_params[v] for v in ('coef', 'decrease_factor', 'max_stepsize'))

#     # compute xnext and the function evaluated at xnext
#     xnext = xcurr - stepsize * grad
#     f_next = cost_fn(xnext)
#     update_stepsize =  lambda _stepsize : update_search(_stepsize, xcurr, grad, cost_fn, decrease_factor, max_stepsize)


# def armijo_line_search(cost_fn, xcurr, f_cur, grad, grad_sqnorm, stepsize, user_params):
#     """Perform Armijo (backtracking) Line search from starting opt problem variables,
#         learning and gradient at the starting opt variables.

#     Args:
#         cost_fn: The cost function to optimize
#         xcurr: current opt variable to optimize.
#         f_cur: value of the cost function at xcurr
#         grad: gradient at xcurr.
#         stepsize: initial guess for the learning rate.
#         user_params: Dictionary
#             maxls: maximum number of steps when doing the backtracking line search
#             coef: ``1-agressiveness`` used for checking the wolfe condition
#             decrease_factor: factor to decrease the learning rate.
#             max_stepsize: upper bound on the learning rate

#     Returns:
#         stepsize: stepsize Armijo line search conditions
#         next_params: params after gradient step
#         f_next: loss after gradient step
#     """
#     # Extract the parameters for this function
#     coef, decrease_factor, max_stepsize = \
#         (user_params[v] for v in ('coef', 'decrease_factor', 'max_stepsize'))

#     # compute xnext and the function evaluated at xnext
#     xnext = xcurr - stepsize * grad
#     f_next = cost_fn(xnext)

#     def update_stepsize(_stepsize):
#         """Multiply stepsize per factor, return new opt variables and new fun evaluations."""
#         _stepsize = jnp.minimum(_stepsize * decrease_factor, max_stepsize)
#         _xnext = xcurr - _stepsize * grad
#         _f_next = cost_fn(_xnext)
#         return _stepsize, _xnext, _f_next

#     def body_fun(t):
#         """ Body function at each line search iteration
#         """
#         _stepsize, _xnext, _f_next, num_iter, _ = t

#         violated = wolfe_cond_violated(_stepsize, coef, f_cur, _f_next, grad_sqnorm)
#         _stepsize, _xnext, _f_next = jax.lax.cond(violated, update_stepsize,
#                                                 lambda _: (_stepsize, _xnext, _f_next),
#                                                 operand=_stepsize
#                                             )
#         num_iter += violated
#         return _stepsize, _xnext, _f_next, num_iter, violated

#     init_val = stepsize, xnext, f_next, jnp.array(1), jnp.array(True)
#     ret = _while_loop(cond_fun=lambda t: t[-1],  # check boolean violated
#                         body_fun=body_fun,
#                         init_val=init_val, max_iter=user_params['maxls'])
#     return ret[:-1] # remove boolean


def _while_loop(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]
