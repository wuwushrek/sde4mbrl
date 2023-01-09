import jax
import jax.numpy as jnp
import collections.abc
import copy
import os

def load_yaml(config_path):
    """Load the yaml file

    Args:
        config_path (str): The path to the yaml file

    Returns:
        dict: The dictionary containing the yaml file
    """
    import yaml
    yml_file = open(os.path.expanduser(config_path))
    yml_byte = yml_file.read()
    cfg_train = yaml.load(yml_byte, yaml.SafeLoader)
    yml_file.close()
    return cfg_train

def apply_fn_to_allleaf(fn_to_apply, types_change, dict_val):
    """Apply a function to all the leaf of a dictionary
    """
    res_dict = {}
    for k, v in dict_val.items():
        # if the value is a dictionary, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = apply_fn_to_allleaf(fn_to_apply, types_change, v)
        elif isinstance(v, types_change):
            res_dict[k] = fn_to_apply(v)
        else:
            res_dict[k] = v
    return res_dict
    
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
    # By default we impose bounds constraint on the states using nonsmooth penalization
    slack_proximal, state_idx, penalty_coeff, state_lb, state_ub = False, None, None, None, None
    if has_xbound:
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
            d[k] = update_params(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def set_values_all_leaves(d, v):
    """Set the same value to all the leaves of a dictionary

    Args:
        d (TYPE): Description
        v (TYPE): Description

    Returns:
        TYPE: Description
    """
    d = copy.deepcopy(d)
    for k, _v in d.items():
        if isinstance(_v, collections.abc.Mapping):
            d[k] = set_values_all_leaves(_v, v)
        else:
            d[k] = v
    return d

def update_same_struct_dict(params, sub_params):
    """Update a dictionary params whose structure is a superset sub_params
    """
    params = copy.deepcopy(params)
    for k, v in sub_params.items():
        if isinstance(v, collections.abc.Mapping):
            if k in params:
                params[k] = update_same_struct_dict(params[k], v)
        else:
            params[k] = v
    return params

def set_values_matching_keys(d, keys_d):
    """ Set the values of the keys of d that matches any key in keys
    """
    key_changed = set()
    d = copy.deepcopy(d)
    for k, val in keys_d.items():
        for kd in d.keys():
            if k in kd:
                d[kd] = val if not isinstance(d[kd], collections.abc.Mapping) else set_values_all_leaves(d[kd], val)
                key_changed.add(kd)
    return key_changed, d


def get_penalty_parameters(dict_params, dict_penalty, default_value):
    """Get the penalty parameters for the loss penalization
        This function assumes that dict_penalty is a flat dictionary

    Args:
        dict_params (TYPE): The dictionary of the parameters
        dict_penlaty (TYPE): A flat dictionary of the penalty parameters

    Returns:
        TYPE: Description
    """
    # penalty_params = {}
    # If some of the key matches --> directly set them

    key_changed, _dict_params = set_values_matching_keys(dict_params, dict_penalty)

    for k, v in _dict_params.items():
        if k in key_changed:
            continue
        if isinstance(v, collections.abc.Mapping):
            _dict_params[k] = get_penalty_parameters(v, dict_penalty, default_value)
        else:
            if default_value is not None:
                _dict_params[k] = default_value
    return _dict_params

def get_non_negative_params(dict_params, enforced_nonneg):
    """Get the parameters that are non negative

    Args:
        dict_params (TYPE): The dictionary of the parameters
        enforced_nonneg (TYPE): The list of the parameters that are non negative

    Returns:
        TYPE: Description
    """
    key_changed, non_neg_params = set_values_matching_keys(dict_params, enforced_nonneg)
    for k, v in dict_params.items():
        if k in key_changed:
            continue
        if isinstance(v, collections.abc.Mapping):
            non_neg_params[k] = get_non_negative_params(v, enforced_nonneg)
        else:
            non_neg_params[k] = False
    return non_neg_params

# def get_penalty_parameters(dict_params, dict_penalty, default_value):
#     """Get the penalty parameters for the loss penalization
#         This function assumes that dict_penalty is a flat dictionary

#     Args:
#         dict_params (TYPE): The dictionary of the parameters
#         dict_penlaty (TYPE): A flat dictionary of the penalty parameters

#     Returns:
#         TYPE: Description
#     """
#     # penalty_params = {}
#     # If some of the key matches --> directly set them

#     key_changed, _dict_params = set_values_matching_keys(dict_params, dict_penalty.keys(), default_value)

#     for k, v in _dict_params.items():
#         if k in key_changed:
#             continue
#         if isinstance(v, collections.abc.Mapping):
#             penalty_params[k] = get_penalty_parameters(v, dict_penalty, default_value)
#         elif k in dict_penalty:
#             penalty_params[k] = dict_penalty[k]
#         else:
#             if default_value is not None:
#                 penalty_params[k] = default_value

#     return penalty_params

# def get_default_parameters(dict_params, dict_nominal, default_value):
#     """Get the penalty parameters for the loss penalization

#     Args:
#         dict_params (TYPE): The dictionary of the parameters
#         dict_penlaty (TYPE): The dictionary of the penalty parameters

#     Returns:
#         TYPE: Description
#     """
#     penalty_params = {}
#     for k, v in dict_params.items():
#         if isinstance(v, collections.abc.Mapping):
#             penalty_params[k] = get_penalty_parameters(v, dict_nominal, default_value)
#         elif k in dict_nominal:
#             penalty_params[k] = dict_nominal[k]
#         else:
#             penalty_params[k] = default_value
#     return penalty_params

# def get_non_negative_params(dict_params, enforced_nonneg):
#     """Get the parameters that are non negative

#     Args:
#         dict_params (TYPE): The dictionary of the parameters
#         enforced_nonneg (TYPE): The list of the parameters that are non negative

#     Returns:
#         TYPE: Description
#     """
#     non_neg_params = {}
#     for k, v in dict_params.items():
#         if isinstance(v, collections.abc.Mapping):
#             non_neg_params[k] = get_non_negative_params(v, enforced_nonneg)
#         elif k in enforced_nonneg:
#             non_neg_params[k] = True
#         else:
#             non_neg_params[k] = False
#     return non_neg_params

def get_value_from_dict(kval, dict_val):
    """ Get the value of the associated key in the dictionary
        The dictionary is allowed to have multilple depht and we need
        to iterate through all of them.
        This function assumes the unicity of the key in the dictioanry
        evem if it's embedded inside another dictionary. The function returns
        the first value it finds in the dictionary.

        Args:
            kval: The key to look for
    """
    for k, v in dict_val.items():
        if isinstance(v, collections.abc.Mapping):
            fv =  get_value_from_dict(kval, v)
            if fv is None:
                continue
            else:
                return fv
        elif k == kval:
            return v
    return None
        
        
