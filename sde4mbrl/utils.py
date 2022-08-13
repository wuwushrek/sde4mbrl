import jax
import jax.numpy as jnp
import collections.abc
import copy

def initialize_problem_constraints(params_model):
    """Check if there are any constraints involved on the hidden states or the inputs
       of the sde model. The constraints are going to be enforced via nonsmooth optimization
       or a change of variable with smooth optimization and proximal projection

    Args:
        params_model (TYPE): The parameters of the model

    Returns:
        TYPE: Description
    """

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
    # By default we impose bounds constraint on the states using nonsmooth penalization
    slack_proximal, state_idx, penalty_coeff, state_lb, state_ub = False, None, None, None, None
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

    return (has_ubound, input_lb, input_ub), \
            (has_xbound, slack_proximal, state_idx, penalty_coeff, state_lb, state_ub )


def update_params(d, u):
    """Update a dictionary with multiple levels

    Args:
        d (TYPE): The dictionary to update
        u (TYPE): The dictionary that contains keys/values to add to d

    Returns:
        TYPE: The modified dictionary d
    """
    d = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
