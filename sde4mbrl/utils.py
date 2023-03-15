import collections.abc
import copy
import os

def load_yaml(config_path):
    """Load a yaml file from a given path

    Args:
        config_path (str): The path to the yaml file

    Returns:
        dict: The dictionary containing the configuration in the yaml file
    """
    import yaml
    # Expand the path to the user home directory if need
    yml_file = open(os.path.expanduser(config_path))
    # Load the yaml file
    yml_byte = yml_file.read()
    cfg_train = yaml.load(yml_byte, yaml.SafeLoader)
    yml_file.close()
    return cfg_train

def apply_fn_to_allleaf(fn_to_apply, types_change, dict_val):
    """Apply a given function to all the leaves of a dictionary of type types_change
        Args:
            fn_to_apply (function): The function to apply to the leaves
            types_change (type): The type of the leaves to apply the function
            dict_val (dict): The dictionary to apply the function to
        Returns:
            dict: The new dictionary with the function applied to the leaves
    """
    res_dict = {}
    for k, v in dict_val.items():
        # if the value is a dictionary or a collection in general, convert it recursively
        if isinstance(v, dict):
            res_dict[k] = apply_fn_to_allleaf(fn_to_apply, types_change, v)
        elif isinstance(v, types_change):
            res_dict[k] = fn_to_apply(v)
        else:
            res_dict[k] = v
    return res_dict
    

def update_params(d, u):
    """Update a dictionary with multiple levels

    Args:
        d (TYPE): The dictionary to update
        u (TYPE): The dictionary that contains keys/values to add to d

    Returns:
        TYPE: A new dictionary d
    """
    d = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_params(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def set_values_all_leaves(d, v):
    """Set the same value (v) to all the leaves of a dictionary

    Args:
        d (TYPE): A dictionary
        v (TYPE): The value to set to all the leaves

    Returns:
        TYPE: The dictionary with the same value set to all the leaves
    """
    d = copy.deepcopy(d)
    for k, _v in d.items():
        if isinstance(_v, collections.abc.Mapping):
            d[k] = set_values_all_leaves(_v, v)
        else:
            d[k] = v
    return d

def update_same_struct_dict(params, sub_params):
    """Update a dictionary params whose structure is a superset of the dictionary sub_params

    Args:
        params (TYPE): The dictionary to update
        sub_params (TYPE): The dictionary that contains keys (subset of params)/values to add to params

    Returns:
        TYPE: A new dictionary params
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
    """ Set the values of the keys of d that matches any key in keys_d

    Args:
        d (TYPE): The dictionary to update
        keys_d (TYPE): A dictionary of keys/values to set to d
    
    Returns:
        TYPE: A tuple of the keys that have been changed and the updated dictionary
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
    """ Given a multi-level dictionary dict_params, this function
    returns a new multi-level dictionary with real values and with the same structure as dict_params.
    What changes is that the values of the leaves are set to the value of the corresponding key in dict_penalty if it exists.
    If the key does not exist, the value is set to default_value if it is not None else it is not changed.

    Args:
        dict_params (TYPE): The dictionary of the parameters
        dict_penalty (TYPE): The dictionary of the penalty parameters
        default_value (TYPE): The default value to set to the parameters that are not in dict_penalty

    Returns:
        TYPE: The dictionary of the parameters with the values set to the values of dict_penalty if they exist
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
    """ Given a multi-level dictionary dict_params, this function
    returns a new multi-level dictionary with boolean values and with the same structure as dict_params.
    What changes is that the values of the leaves are set to True if the corresponding key in enforced_nonneg exists.
    If the key does not exist, the value is set to False.

    Args:
        dict_params (TYPE): The dictionary of the parameters
        enforced_nonneg (TYPE): The dictionary of the non-negative parameters

    Returns:
        TYPE: The dictionary of the parameters with the values set to True if the corresponding key in enforced_nonneg exists

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

def get_value_from_dict(kval, dict_val):
    """ Get the value of a key in a multi-level dictionary
    This function may return a dictionary if the key is not a leaf or none if the key is not found.

    Args:
        kval (TYPE): The key to search
        dict_val (TYPE): The dictionary to search

    Returns:
        TYPE: The value of the key or None if the key is not found

    """
    for _key in dict_val.keys():
        if kval in _key:
            return dict_val[_key]

    for k, v in dict_val.items():
        if isinstance(v, collections.abc.Mapping):
            fv = get_value_from_dict(kval, v)
            if fv is None:
                continue
            else:
                return fv
    return None

    # for k, v in dict_val.items():
    #     if isinstance(v, collections.abc.Mapping):
    #         fv =  get_value_from_dict(kval, v)
    #         if fv is None:
    #             continue
    #         else:
    #             return fv
    #     elif k == kval:
    #         return v
    # return None