import numpy as np

# import copy
# import collections
import os

################################ Qauternion and Conversion Utilities ################################
q_ENU_to_NED = np.array([0, np.sqrt(0.5), np.sqrt(0.5), 0])
q_FLU_to_FRD = np.array([0, 1., 0, 0])

def quatmult( a, b, array_lib=np):
    """Multiply two quaternions.
    """
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return array_lib.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2]
                )

def quatinv( a, array_lib=np):
    """Inverse a quaternion.
    """
    w, x, y, z = a
    return array_lib.array([w, -x, -y, -z])

def quat_rotatevector(q, v, array_lib=np):
    """ Rotate a vector (numpy) by a quaternion (numpy) """
    v = array_lib.array([0., v[0], v[1], v[2]])
    return quatmult(quatmult(q, v, array_lib), quatinv(q, array_lib), array_lib)[1:]

def quat_rotatevectorinv(q, v, array_lib=np):
    """Rotate a vector (numpy) by a quaternion (numpy) in the opposite direction.
    """
    v = array_lib.array([0., v[0], v[1], v[2]])
    return quatmult(quatinv(q, array_lib), quatmult(v, q, array_lib), array_lib)[1:]

def quat_from_euler(roll, pitch, yaw, array_lib=np):
    """Convert Euler angles to quaternion.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        list: Quaternion [w, x, y, z].
    """
    cy = array_lib.cos(yaw * 0.5)
    sy = array_lib.sin(yaw * 0.5)
    cr = array_lib.cos(roll * 0.5)
    sr = array_lib.sin(roll * 0.5)
    cp = array_lib.cos(pitch * 0.5)
    sp = array_lib.sin(pitch * 0.5)

    return array_lib.array([cy * cr * cp + sy * sr * sp,
                    cy * sr * cp - sy * cr * sp,
                    cy * cr * sp + sy * sr * cp,
                    sy * cr * cp - cy * sr * sp])

def quat_to_euler(q, array_lib=np):
    """Convert quaternion to Euler angles.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Euler angles [roll, pitch, yaw].
    """
    w, x, y, z = q
    roll = array_lib.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    t2 = 2 * (w * y - z * x)
    t2 = array_lib.clip(t2, -1., 1.)
    pitch = array_lib.arcsin(t2)
    yaw = array_lib.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw

def quat_get_yaw(q, array_lib=np):
    """Get yaw angle from quaternion.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        float: Yaw angle in radians.
    """
    w, x, y, z = q
    return array_lib.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def enu_to_ned_orientation(q, array_lib=np):
    """Convert orientation from ENU (FLU) to NED.

    Args:
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Quaternion [w, x, y, z].
    """
    q_FLU_to_NED = quatmult(q_ENU_to_NED, q, array_lib)
    return quatmult(q_FLU_to_NED, quatinv(q_FLU_to_FRD,array_lib), array_lib)

def flu_to_frd_conversion(v, array_lib=np):
    """Convert orientation from FLU to FRD.
       Typically use for converting body frame enu to ned or inverse
    Args:
        v (list): A vector [x, y, z].

    Returns:
        list: A vector in frd frame
    """
    return quat_rotatevector(q_FLU_to_FRD, v, array_lib)

def body_to_inertial_frame(p, q, array_lib=np):
    """Convert position from body frame to inertial frame.

    Args:
        p (list): Position [x, y, z].
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Position [x, y, z].
    """
    return quat_rotatevector(q, p, array_lib)

def inertial_to_body_frame(p, q, array_lib=np):
    """Convert position from inertial frame to body frame.

    Args:
        p (list): Position [x, y, z].
        q (list): Quaternion [w, x, y, z].

    Returns:
        list: Position [x, y, z].
    """
    return quat_rotatevectorinv(q, p, array_lib)

def enu_to_ned_position(p, array_lib=np):
    """Convert position from ENU (FLU) to NED.

    Args:
        p (list): Position [x, y, z].

    Returns:
        list: Position [x, y, z].
    """
    return quat_rotatevector(q_ENU_to_NED, p, array_lib)

def enu_euler_to_ned_euler(roll, pitch, yaw, array_lib=np):
    """Convert Euler angles from ENU (FLU) to NED.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        list: Euler angles [roll, pitch, yaw].
    """
    q = quat_from_euler(roll, pitch, yaw, array_lib)
    q = enu_to_ned_orientation(q, array_lib)
    return quat_to_euler(q, array_lib)

def enu_to_ned_z(z):
    """Convert z position from ENU (FLU) to NED.

    Args:
        z (float): Z position.

    Returns:
        float: Z position.
    """
    return -z

# def ned2enu(x):
#     return jnp.concatenate((ned_to_enu_position(x[:3], jnp),
#                             ned_to_enu_position(x[3:6],jnp),
#                             ned_to_enu_orientation(x[6:10], jnp),
#                             frd_to_flu_conversion(x[10:], jnp)
#                             )
#             )

def enu2ned(x, array_lib=np):
    return array_lib.concatenate((enu_to_ned_position(x[:3], array_lib),
                            enu_to_ned_position(x[3:6],array_lib),
                            enu_to_ned_orientation(x[6:10], array_lib),
                            flu_to_frd_conversion(x[10:], array_lib)
                            )
            )

ned_to_enu_position = enu_to_ned_position

ned_to_enu_orientation = enu_to_ned_orientation

ned_to_enu_z = enu_to_ned_z

ned_euler_to_enu_euler = enu_euler_to_ned_euler

frd_to_flu_conversion = flu_to_frd_conversion

############################ Loading trajectories utilities ############################

# def load_yaml(config_path):
#     """Load the yaml file

#     Args:
#         config_path (str): The path to the yaml file

#     Returns:
#         dict: The dictionary containing the yaml file
#     """
#     import yaml
#     yml_file = open(os.path.expanduser(config_path))
#     yml_byte = yml_file.read()
#     cfg_train = yaml.load(yml_byte, yaml.SafeLoader)
#     yml_file.close()
#     return cfg_train

def load_trajectory(filename):
    """ Load the trajectory from a csv file
        return the dictionary of the trajectory using numpy array
    """
    import pandas as pd
    df = pd.read_csv(os.path.expanduser(filename))
    # Convert the dataframe to a dictionary
    dict_traj = df.to_dict('list')
    # Convert the lists to numpy arrays
    for key in dict_traj.keys():
        dict_traj[key] = np.array(dict_traj[key])
    return dict_traj

# def update_params(d, u):
#     """Update a dictionary with multiple levels

#     Args:
#         d (TYPE): The dictionary to update
#         u (TYPE): The dictionary that contains keys/values to add to d

#     Returns:
#         TYPE: The modified dictionary d
#     """
#     d = copy.deepcopy(d)
#     for k, v in u.items():
#         if isinstance(v, collections.abc.Mapping):
#             d[k] = update_params(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

# def apply_fn_to_allleaf(fn_to_apply, types_change, dict_val):
#     """Apply a function to all the leaf of a dictionary
#     """
#     res_dict = {}
#     for k, v in dict_val.items():
#         # if the value is a dictionary, convert it recursively
#         if isinstance(v, dict):
#             res_dict[k] = apply_fn_to_allleaf(fn_to_apply, types_change, v)
#         elif isinstance(v, types_change):
#             res_dict[k] = fn_to_apply(v)
#         else:
#             res_dict[k] = v
#     return res_dict

def find_consecutive_true(metrics, min_length=-1):
    """Return the set of indices of minimum length 'min_length' for which
    the boolean array 'metrics' is True.
    Example: Typically, this can be used to identify the sideslip angle offset
             or where the model is actually invertible

    Args:
        metrics (TYPE): Description
        min_length (TYPE, optional): Description

    Returns:
        TYPE: Description
    """
    full_auto = metrics
    section_inds_ = np.split(np.r_[:len(full_auto)], np.where(np.diff(full_auto) != 0)[0]+1)

    full_auto_inds = []

    for inds_ in section_inds_:
        if full_auto[inds_[0]] and len(inds_) > min_length:
            full_auto_inds.append(inds_)
    return full_auto_inds

def parse_ulog(ulog_file, topic='mpc_full_state', outlier_cond=lambda d : d['z']>0.1, min_length=500):
    """Parse the ulog file and return the data in a dictionary.

    Args:
        ulog_file (str): Path to the ulog file.
        topic_list (list): List of topics to be parsed.

    Returns:
        dict: Dictionary of the parsed data.
    """
    # Import the ulog module
    from pyulog.core import ULog
    from tqdm.auto import tqdm

    # Parse the ulog file
    ulog = ULog(os.path.expanduser(ulog_file), message_name_filter_list=[topic], disable_str_exceptions=True)
    # Check if the ulog is valid and not empty
    if len(ulog.data_list) <= 0:
        raise ValueError("The ulog file is empty.")
    
    # Convert the ulog to a dictionary
    res_dict = dict()

    # Go through the message that has been saved
    msg = ulog.data_list[0]

    res_dict['t'] = msg.data['timestamp_sample'] / 1e6
    # # Compute the delta time
    time_step = res_dict['t'][1:] - res_dict['t'][:-1]
    # Print the mean time step and the standard deviation
    print("Mean time step: {:.3f} s".format(np.mean(time_step)))

    # Save each fill of the message
    for msgname, msgdata in msg.data.items():
        # Pass the timestamp and timestamp_sample fields
        if msgname in ['timestamp', 'timestamp_sample']:
            continue
        msg_array = np.array(msgdata)
        if np.any(np.isnan(msg_array)):
            print("Warning: {} contains NaN values -> We removed it.".format(msgname))
            continue
        res_dict[msgname] = msg_array
    
    # Let's do some conversion
    # Check if x,y,z are in the message
    if 'x' in res_dict and 'y' in res_dict and 'z' in res_dict:
        # Convert them from NED to ENU
        for i in tqdm(range(len(res_dict['x'])), leave=False):
            res_dict['x'][i], res_dict['y'][i], res_dict['z'][i] = \
                ned_to_enu_position([res_dict['x'][i], res_dict['y'][i], res_dict['z'][i]])
    # Check if vx,vy,vz are in the message
    if 'vx' in res_dict and 'vy' in res_dict and 'vz' in res_dict:
        # Convert them from NED to ENU
        for i in tqdm(range(len(res_dict['vx'])), leave=False):
            res_dict['vx'][i], res_dict['vy'][i], res_dict['vz'][i] = \
                ned_to_enu_position([res_dict['vx'][i], res_dict['vy'][i], res_dict['vz'][i]])
    # Check if qw,qx,qy,qz are in the message
    if 'qw' in res_dict and 'qx' in res_dict and 'qy' in res_dict and 'qz' in res_dict:
        # Convert them from NED to ENU
        for i in tqdm(range(len(res_dict['qw'])), leave=False):
            res_dict['qw'][i], res_dict['qx'][i], res_dict['qy'][i], res_dict['qz'][i] = \
                ned_to_enu_orientation([res_dict['qw'][i], res_dict['qx'][i], res_dict['qy'][i], res_dict['qz'][i]])
    # Check if wx,wy,wz are in the message
    if 'wx' in res_dict and 'wy' in res_dict and 'wz' in res_dict:
        # Convert them from NED to ENU
        for i in tqdm(range(len(res_dict['wx'])), leave=False):
            res_dict['wx'][i], res_dict['wy'][i], res_dict['wz'][i] = \
                frd_to_flu_conversion([res_dict['wx'][i], res_dict['wy'][i], res_dict['wz'][i]])

    # Do some cleaning
    # Remove values that are too closed to the ground
    if outlier_cond is not None:
        conseq_ind = find_consecutive_true(outlier_cond(res_dict), min_length=min_length)
        _res_list = []
        if len(conseq_ind) > 0:
            for _indexes in conseq_ind:
                # Print the length of the sequence
                tqdm.write("The length of the sequence is {} | Original = {}".format(len(_indexes), len(res_dict['z'])))
                new_dict = {}
                # Keep only the longest sequence
                for key in res_dict.keys():
                    new_dict[key] = res_dict[key][_indexes]
                    # Check if nan are present
                    if np.any(np.isnan(new_dict[key])):
                        raise ValueError("The {} contains nan values.".format(key))
                _res_list.append(new_dict)
        return _res_list
    else:
        return [res_dict]