""" Very handcrafted and hard-coded script to compare the performance of the different controllers
"""
import os

# MPC seems to be faster on cpu because of the loop
# TODO: check if this is still true, and investiage how to make it faster on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np

import matplotlib.pyplot as plt

from sde4mbrlExamples.rotor_uav.utils import *

from matplotlib.gridspec import GridSpec

def load_logs(log_dir):
    """Load the trajectories from the file
        Args:
            log_dir (str): Directory where the log file is stored
            outlier_cond (function): Function that returns True if the data point is an outlier
            min_length (int): Minimum length of the trajectory when splitting using outlier
        Returns: (as a tuple)
            x (list): List of ndarray of shape (N, 13) containing the states
            u (list): List of ndarray of shape (N, 4) or (N, 6) containing the controls
    """
    from sde4mbrlExamples.rotor_uav.utils import parse_ulog
    log_dir = os.path.expanduser(log_dir)
    # Load the data from the ULog
    _log_data = parse_ulog(log_dir, outlier_cond=None, min_length=10, mavg_dict={})[0]

    # Ordered state names
    name_states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz']

    # Extract the states and controls
    x = np.stack([_log_data[_name] for _name in name_states], axis=1)

    # Compute the euler angles from the quaternion using 
    rpy = []
    for _q in x[:,6:10]:
        rpy.append(list(quat_to_euler(_q, np)))
    rpy = np.array(rpy)
    x = np.concatenate([x, rpy], axis=1)
    timev = _log_data['t']
    return timev, x


def load_mpc_part_from_ulog(ulog_file):
    """ Load the data from the ulog file
        ulog_file: path to the ulog file
    """
    from pyulog.core import ULog

    # NGet the mpc_motors_cmd message
    topic = 'mpc_motors_cmd'
    ulog = ULog(os.path.expanduser(ulog_file), message_name_filter_list=[topic], disable_str_exceptions=True)
    # Check if the ulog is valid and not empty
    if len(ulog.data_list) <= 0:
        raise ValueError("The ulog file is empty.")
    msg_motors = ulog.data_list[0]
    timestamp_motors = msg_motors.data['timestamp'] * 1e-6

    # Let's get the mpc_on message and check for when mpc_on == 5
    mpc_on_list = msg_motors.data['mpc_on']
    # Get the corresponding initial  and final time
    t_init = timestamp_motors[mpc_on_list == 5][0]
    t_final = timestamp_motors[mpc_on_list == 5][-1]
    # print('t_init: ', t_init)
    # print('t_final: ', t_final)
    # print('t_final - t_init: ', t_final - t_init)

    timestamp_pose, _x = load_logs(ulog_file)
    # Get the corresponding indexes
    idx_init = (timestamp_pose >= t_init) & (timestamp_pose <= t_final)
    # Now we can get the mpc_full_state corresponding to the t_init and t_final
    t_mpc = timestamp_pose[idx_init]
    t_mpc = t_mpc - t_mpc[0]
    print('t_mpc init: ', t_mpc[0])
    print('t_mpc final: ', t_mpc[-1])
    print('t_mpc final - t_mpc init: ', t_mpc[-1] - t_mpc[0])
    return t_mpc, _x[idx_init]

def parse_trajectory(_traj_path):
    """ Return the array of time and concatenate the other states
        _traj: a dictionary with the keys: t, x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz
    """
    _traj = load_trajectory(_traj_path)
    # List of states in order
    states = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz','roll', 'pitch', 'yaw']
    time_val = np.array(_traj['t'])
    # stack the states
    state_val = np.stack([_traj[state] for state in states], axis=1)
    return time_val, state_val

# def compute_rmse(ref_xyz, tref, veh_xyz, tveh):
#     """ Compute the RMSE between the reference and the vehicle trajectory.
#         The reference trajectory is interpolated to the vehicle trajectory.
#         Args:
#             ref_xyz (ndarray): Reference trajectory of shape (N, 3)
#             tref (ndarray): Time of the reference trajectory of shape (N,)
#             veh_xyz (ndarray): Vehicle trajectory of shape (M, 3)
#             tveh (ndarray): Time of the vehicle trajectory of shape (M,)
#         Returns:
#             rmse (ndarray): RMSE of shape (M,)
#     """
#     from scipy.interpolate import interp1d
#     # Interpolate the reference trajectory
#     f = interp1d(tref, ref_xyz, axis=0)
#     indx_in = (tref[0] < tveh) & (tref[-1] > tveh)
#     # Get the interpolated reference trajectory
#     ref_xyz_interp = f(tveh[indx_in])
#     # Compute the RMSE
#     rmse = np.sqrt(np.sum((veh_xyz[indx_in] - ref_xyz_interp)**2, axis=1))
#     return rmse, tveh[indx_in]

def compute_rmse(ref_xyz, tref, veh_xyz, tveh):
    """ Compute the RMSE between the reference and the vehicle trajectory.
        The reference trajectory is interpolated to the vehicle trajectory.
        Args:
            ref_xyz (ndarray): Reference trajectory of shape (N, 3)
            tref (ndarray): Time of the reference trajectory of shape (N,)
            veh_xyz (ndarray): Vehicle trajectory of shape (M, 3)
            tveh (ndarray): Time of the vehicle trajectory of shape (M,)
        Returns:
            rmse (ndarray): RMSE of shape (M,)
    """
    from scipy.interpolate import interp1d
    # Interpolate the reference trajectory
    f = interp1d(tveh, veh_xyz, axis=0)
    indx_in = (tveh[0] < tref) & (tveh[-1] > tref)
    # Get the interpolated reference trajectory
    ref_xyz_interp = f(tref[indx_in])
    # Compute the RMSE
    rmse = np.sqrt(np.sum((ref_xyz[indx_in] - ref_xyz_interp)**2, axis=1))
    return rmse, tref[indx_in]

def plot_trajs(labeled_controllers, ref_path, spacing=1):
    """
    """
    dir_for_path = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_data/'
    _ref_path = os.path.expanduser(dir_for_path + ref_path)
    time_ref, state_ref = parse_trajectory(_ref_path)
    tmin = 2
    tmax = 35.0
    idx_ref = (time_ref >= tmin) & (time_ref <= tmax)
    time_ref = time_ref[idx_ref]
    state_ref = state_ref[idx_ref]
    for ctrl_name, ctrl_info in labeled_controllers.items():
        _t_drone, _drone_traj = load_mpc_part_from_ulog(os.path.expanduser(dir_for_path + ctrl_info['traj']))
        # Reference offset time
        offset_t = ctrl_info.get('offset_t', 0.0)
        _t_drone = _t_drone - _t_drone[0] + offset_t
        idx_ref = (_t_drone >= tmin) & (_t_drone <= tmax)
        labeled_controllers[ctrl_name]['tevol'] = _t_drone[idx_ref]
        labeled_controllers[ctrl_name]['xevol'] = _drone_traj[idx_ref]

    
    # Grid of the plots
    nrows, ncols = 3, 4
    ref_zorder = 100
    ref_line_style = '--'
    ref_color = '#000000'
    general_style = {'linewidth': 2.0, 'markersize': 1.0}
    ref_dict = {'color': ref_color, 'zorder': ref_zorder, 'linestyle': ref_line_style}

    def ax_plot(ax, x, y, _spacing=spacing, **kwargs):
        assert len(x) == len(y), 'x and y must have the same length'
        if len(x) < _spacing:
            ax.plot(x, y, **{**general_style, **kwargs})
            ax.grid(True)
            return
        ax.plot(x[::_spacing], y[::_spacing], **{**general_style, **kwargs})
        ax.grid(True)

    # We are making a gridplot for our plots
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=None, hspace=None)

    ax_xy = fig.add_subplot(gs[:2,:2])
    ax_z = fig.add_subplot(gs[2,0])
    ax_rms = fig.add_subplot(gs[0,2])
    ax_rms_vel = fig.add_subplot(gs[0,3]) 
    ax_vel = fig.add_subplot(gs[1,2:])
    ax_roll = fig.add_subplot(gs[2,1])
    ax_pitch = fig.add_subplot(gs[2,2])
    ax_yaw = fig.add_subplot(gs[2,3])
                    
    for i, (ctrl_name, ctrl_info) in enumerate(labeled_controllers.items()):
        # First the trajectory in the x-y plane
        ax_plot(ax_xy, ctrl_info['xevol'][:,0], ctrl_info['xevol'][:,1], label=ctrl_name, **ctrl_info['style'])
        ax_xy.plot(state_ref[:,0], state_ref[:,1], label='Reference', **ref_dict)
        ax_xy.set_xlabel(r'x [m]')
        ax_xy.set_ylabel(r'y [m]')

        # Plot the z trajectory
        ax_plot(ax_z, ctrl_info['tevol'], ctrl_info['xevol'][:,2], label=ctrl_name, **ctrl_info['style'])
        ax_z.plot(time_ref, state_ref[:,2], label='Reference', **ref_dict)
        ax_z.set_xlabel(r't [s]')
        ax_z.set_ylabel(r'z [m]')

        # Plot the velocity magintude
        vel_value = np.linalg.norm(state_ref[:,3:6], axis=1)
        drone_vel_value = np.linalg.norm(ctrl_info['xevol'][:,3:6], axis=1)
        ax_plot(ax_vel, ctrl_info['tevol'], drone_vel_value, label=ctrl_name, **ctrl_info['style'])
        ax_vel.plot(time_ref, vel_value, label='Reference', **ref_dict)
        ax_vel.set_xlabel(r't [s]')
        ax_vel.set_ylabel(r'V  [m/s]')

        # Plot RMS
        rmse, t_rmse = compute_rmse(state_ref[:,0:3], time_ref, ctrl_info['xevol'][:,0:3], ctrl_info['tevol'])
        # Cumulative mean of the position error
        rmse = np.cumsum(rmse) / np.arange(1, len(rmse)+1)
        ax_plot(ax_rms, t_rmse, rmse, label=ctrl_name, **ctrl_info['style'])
        ax_rms.set_xlabel(r't [s]')
        ax_rms.set_ylabel(r'RMSE [m]')

        # Plot RMS velocity
        rmse_vel, t_rmse_vel = compute_rmse(state_ref[:,3:6], time_ref, ctrl_info['xevol'][:,3:6], ctrl_info['tevol'])
        # Cumulative mean of the position error
        rmse_vel = np.cumsum(rmse_vel) / np.arange(1, len(rmse_vel)+1)
        ax_plot(ax_rms_vel, t_rmse_vel, rmse_vel, label=ctrl_name, **ctrl_info['style'])
        ax_rms_vel.set_xlabel(r't [s]')
        ax_rms_vel.set_ylabel(r'RMSE V [m/s]')
        # Plot x, y and the reference on ax_roll and ax_pitch
        ax_plot(ax_roll, ctrl_info['tevol'], ctrl_info['xevol'][:,0], label=ctrl_name, **ctrl_info['style'])
        ax_plot(ax_roll, time_ref, state_ref[:,0], label='Reference', **ref_dict)
        
        ax_plot(ax_pitch, ctrl_info['tevol'], ctrl_info['xevol'][:,1], label=ctrl_name, **ctrl_info['style'])
        ax_plot(ax_pitch, time_ref, state_ref[:,1], label='Reference', **ref_dict)
    plt.show()

# controllers = {
#     'NSDE+MPC' : 
#         {'traj': 'circle_traj_NSDE_v1.ulg',
#          'offset_t' : 0.1,
#         'style': {'color': 'red', 'zorder': 10, 'linestyle': '-'},
#         },
#     'ODE+MPC' : 
#         {'traj': 'circle_traj_ODE_v1.ulg',
#          'offset_t' : 0.1,
#         'style': {'color': 'green', 'zorder': 5, 'linestyle': '-'},
#         },
#     'SysId+MPC' : 
#         {'traj': 'circle_traj_sysId_v1.ulg',
#          'offset_t' : 0.1,
#         'style': {'color': 'blue', 'zorder': 5, 'linestyle': '-'},
#         },
# }
# plot_trajs(controllers, 'fast_circle.csv', spacing=1)

# controllers = {
#     'NSDE+MPC' : 
#         {'traj': 'lemn2_traj_NeurSDE_test.ulg',
#         'offset_t' : 0.1,
#         'style': {'color': 'red', 'zorder': 10, 'linestyle': '-'},
#         },
#     'ODE+MPC' : 
#         {'traj': 'lemn2_traj_NeurODE_test.ulg',
#          'offset_t' : 0.1,
#         'style': {'color': 'green', 'zorder': 5, 'linestyle': '-'},
#         },
#     'SysId+MPC' : 
#         {'traj': 'lemn2_traj_sysId_v1.ulg',
#         'style': {'color': 'blue', 'zorder': 5, 'linestyle': '-'},
#         },
# }
# controllers = {
#     'NSDE+MPC' : 
#         {'traj': 'lemn2_traj_SDE_v2.ulg',
#         'offset_t' : 0.10,
#         'style': {'color': 'red', 'zorder': 10, 'linestyle': '-'},
#         },
#     'ODE+MPC' : 
#         {'traj': 'lemn2_traj_ODE_v1.ulg',
#          'offset_t' : 0.1,
#         'style': {'color': 'green', 'zorder': 5, 'linestyle': '-'},
#         },
#     'SysId+MPC' : 
#         {'traj': 'lemn2_traj_sysId_v1.ulg',
#         'style': {'color': 'blue', 'zorder': 5, 'linestyle': '-'},
#         },
# }
# plot_trajs(controllers, 'fast2_lemn.csv', spacing=1)

# controllers = {
#     'NSDE+MPC' : 
#         {'traj': 'circle2_traj_SDE_v1.ulg',
#         'offset_t' : 0.18,
#         'style': {'color': 'red', 'zorder': 10, 'linestyle': '-'},
#         },
#     'ODE+MPC' : 
#         {'traj': 'circle2_traj_ODE_v1.ulg',
#          'offset_t' : 0.18,
#         'style': {'color': 'green', 'zorder': 5, 'linestyle': '-'},
#         },
#     'SysId+MPC' : 
#         {'traj': 'circle2_traj_sysId_v1.ulg',
#         'style': {'color': 'blue', 'zorder': 5, 'linestyle': '-'},
#         },
# }
# plot_trajs(controllers, 'fast2_circle.csv', spacing=1)

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    if hasattr(obj, "_dash_pattern"):
        obj._us_dashOffset = obj._dash_pattern[0]
        obj._us_dashSeq = obj._dash_pattern[1]
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def plot_trajs_multi(labeled_controllers, ref_path, spacing=1, tmin=0, tmax=1000, rmse_compute=False, 
                     outname='circle_iris.png', dpi=500, figsize=None, tikz=False):
    """ Plot the trajectories of the controllers in the same plot
        This assumes multiple trajectories of the same controller are being given
    """
    dir_for_path = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_data/'
    _ref_path = os.path.expanduser(dir_for_path + ref_path)
    time_ref, state_ref = parse_trajectory(_ref_path)
    idx_ref = (time_ref >= tmin) & (time_ref <= tmax)
    time_ref = time_ref[idx_ref]
    state_ref = state_ref[idx_ref]
    for ctrl_name, ctrl_info in labeled_controllers.items():
        trajId = ctrl_info['trajId']
        num_traj = ctrl_info['num_traj']
        t_rmse_list = []
        pos_rmse, vel_rmse = [], []
        for i in range(1, num_traj+1):
            _t_drone, _drone_traj = load_mpc_part_from_ulog(os.path.expanduser(dir_for_path + ctrl_info['traj'].format(i)))
            # Ret_rmse_velference offset time
            offset_t = ctrl_info.get('offset_t', 0.0)
            _t_drone = _t_drone - _t_drone[0] + offset_t
            idx_ref = (_t_drone >= tmin) & (_t_drone <= tmax)
            _t_drone = _t_drone[idx_ref]
            _drone_traj = _drone_traj[idx_ref]
            if i == trajId:
                labeled_controllers[ctrl_name]['tevol'] = _t_drone
                labeled_controllers[ctrl_name]['xevol'] = _drone_traj
            # Compute rms error on position
            rmse, t_rmse = compute_rmse(state_ref[:,0:3], time_ref, _drone_traj[:,0:3], _t_drone)
            if rmse_compute:
                rmse = np.cumsum(rmse) / np.arange(1, len(rmse)+1)
            # compute the rms error on velocity
            rmse_vel, _ = compute_rmse(state_ref[:,3:6], time_ref, _drone_traj[:,3:6], _t_drone)
            if rmse_compute:
                rmse_vel = np.cumsum(rmse_vel) / np.arange(1, len(rmse_vel)+1)
            pos_rmse.append(rmse)
            vel_rmse.append(rmse_vel)
            t_rmse_list.append(t_rmse)
            # print (t_rmse.shape, rmse.shape, t_rmse[:-8])
        # Compute the mean and std of the rmse
        max_inter_t = np.min([t[-1] for t in t_rmse_list])
        min_inter_t = np.max([t[0] for t in t_rmse_list])
        pos_rmse = np.array([r[(t >= min_inter_t) & (t <= max_inter_t)] for r, t in zip(pos_rmse, t_rmse_list)])
        vel_rmse = np.array([r[(t >= min_inter_t) & (t <= max_inter_t)] for r, t in zip(vel_rmse, t_rmse_list)])
        labeled_controllers[ctrl_name]['pos_rmse_mean'] = np.mean(pos_rmse, axis=0)
        labeled_controllers[ctrl_name]['pos_rmse_std'] = np.std(pos_rmse, axis=0)
        labeled_controllers[ctrl_name]['vel_rmse_mean'] = np.mean(vel_rmse, axis=0)
        labeled_controllers[ctrl_name]['vel_rmse_std'] = np.std(vel_rmse, axis=0)
        labeled_controllers[ctrl_name]['t_rmse'] = t_rmse_list[0][(t_rmse_list[0] >= min_inter_t) & (t_rmse_list[0] <= max_inter_t)]

    # Grid of the plots
    nrows, ncols = 3, 4
    ref_zorder = 100
    ref_line_style = '--'
    ref_color = '#000000'
    general_style = {'linewidth': 2.0, 'markersize': 1.0}
    ref_dict = {'color': ref_color, 'zorder': ref_zorder, 'linestyle': ref_line_style}

    def ax_plot(ax, x, y, _spacing=spacing, **kwargs):
        assert len(x) == len(y), 'x and y must have the same length'
        if len(x) < _spacing:
            ax.plot(x, y, **{**general_style, **kwargs})
            ax.grid(True)
            return
        ax.plot(x[::_spacing], y[::_spacing], **{**general_style, **kwargs})
        ax.grid(True)

    # We are making a gridplot for our plots
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=None, hspace=None)

    ax_xy = fig.add_subplot(gs[:2,:2])
    ax_z = fig.add_subplot(gs[2,0])
    ax_rms = fig.add_subplot(gs[0,2])
    ax_rms_vel = fig.add_subplot(gs[0,3]) 
    ax_vel = fig.add_subplot(gs[1,2:])
    ax_roll = fig.add_subplot(gs[2,1])
    ax_pitch = fig.add_subplot(gs[2,2])
    ax_yaw = fig.add_subplot(gs[2,3])
                    
    for i, (ctrl_name, ctrl_info) in enumerate(labeled_controllers.items()):
        # First the trajectory in the x-y plane
        ax_plot(ax_xy, ctrl_info['xevol'][:,0], ctrl_info['xevol'][:,1], label=ctrl_name, **ctrl_info['style'])
        if i == 0:
            ax_xy.plot(state_ref[:,0], state_ref[:,1], label='Reference', **ref_dict)
            ax_xy.set_xlabel(r'x [m]')
            ax_xy.set_ylabel(r'y [m]')

        # Plot the z trajectory
        ax_plot(ax_z, ctrl_info['tevol'], ctrl_info['xevol'][:,2], **ctrl_info['style'])
        if i == 0:
            ax_z.plot(time_ref, state_ref[:,2], **ref_dict)
            ax_z.set_xlabel(r't [s]')
            ax_z.set_ylabel(r'z [m]')

        # Plot the velocity magintude
        vel_value = np.linalg.norm(state_ref[:,3:6], axis=1)
        drone_vel_value = np.linalg.norm(ctrl_info['xevol'][:,3:6], axis=1)
        ax_plot(ax_vel, ctrl_info['tevol'], drone_vel_value, **ctrl_info['style'])
        if i == 0:
            ax_vel.plot(time_ref, vel_value, **ref_dict)
            # ax_vel.set_xlabel(r't [s]')
            ax_vel.set_ylabel(r'V  [m/s]')

        # Plot RMS
        ax_plot(ax_rms, ctrl_info['t_rmse'], ctrl_info['pos_rmse_mean'], **ctrl_info['style'])
        ax_rms.fill_between(ctrl_info['t_rmse'], ctrl_info['pos_rmse_mean'] - ctrl_info['pos_rmse_std'], ctrl_info['pos_rmse_mean'] + ctrl_info['pos_rmse_std'], alpha=0.2, **ctrl_info['style'])
        if i == 0:
            # ax_rms.set_xlabel(r't [s]')
            ax_rms.set_ylabel(r'RMSE [m]' if rmse_compute else 'Inst. Err. [m]')

        # Plot RMS velocity
        ax_plot(ax_rms_vel, ctrl_info['t_rmse'], ctrl_info['vel_rmse_mean'], **ctrl_info['style'])
        ax_rms_vel.fill_between(ctrl_info['t_rmse'], ctrl_info['vel_rmse_mean'] - ctrl_info['vel_rmse_std'], ctrl_info['vel_rmse_mean'] + ctrl_info['vel_rmse_std'], alpha=0.2, **ctrl_info['style'])
        if i == 0:
            # ax_rms_vel.set_xlabel(r't [s]')
            ax_rms_vel.set_ylabel(r'RMSE V [m/s]' if rmse_compute else 'Inst. Err. V [m/s]')

        # Plot x, y and the reference on ax_roll and ax_pitch
        ax_plot(ax_roll, ctrl_info['tevol'], ctrl_info['xevol'][:,-3], **ctrl_info['style'])
        ax_plot(ax_pitch, ctrl_info['tevol'], ctrl_info['xevol'][:,-2], **ctrl_info['style'])
        ax_plot(ax_yaw, ctrl_info['tevol'], ctrl_info['xevol'][:,-1],  **ctrl_info['style'])
        # ax_plot(ax_roll, ctrl_info['tevol'], ctrl_info['xevol'][:,0], label=ctrl_name, **ctrl_info['style'])
        # ax_plot(ax_pitch, ctrl_info['tevol'], ctrl_info['xevol'][:,1], label=ctrl_name, **ctrl_info['style'])
        
        if i == 0:
            ax_plot(ax_roll, time_ref, state_ref[:,-3],  **ref_dict)
            ax_plot(ax_pitch, time_ref, state_ref[:,-2], **ref_dict)
            ax_plot(ax_yaw, time_ref, state_ref[:,-1],  **ref_dict)
            ax_roll.set_xlabel(r't [s]')
            ax_roll.set_ylabel(r'$\phi$ [rad]')
            ax_pitch.set_xlabel(r't [s]')
            ax_pitch.set_ylabel(r'$\theta$ [rad]')
            ax_yaw.set_xlabel(r't [s]')
            ax_yaw.set_ylabel(r'$\psi$ [rad]')
    
    # Let's center the figure legend
    lines = []
    labels = []
    for ax in [ax_xy, ax_z, ax_vel, ax_rms, ax_rms_vel, ax_roll, ax_pitch, ax_yaw]:
        l, l_ = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(l_)
    fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))

    # Save the figure as a png
    fig.savefig('my_data/'+outname, dpi = dpi, bbox_inches='tight', transparent=True)

    # Export the figure as tikz
    if tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.clean_figure(fig)
        outname = outname.replace('.png', '.tex') if outname.endswith('.png') else outname + '.tex'
        tikzplotlib.save('my_data/'+outname, figure=fig)

    plt.show()

# Colors scheme
sde_color = '#157F1F' # 'r'
sysid_color = '#8D8D92' # 'b'
gt_color = 'k'
ode_color = '#00A7E1' # 'g'

# controllers = {
#     'Sys ID' : 
#         {'traj': 'lemn2_traj_SysId_v{}.ulg',
#         #  'offset_t' : 0.1,
#          'num_traj' : 5,
#          'trajId' : 1,
#         'style': {'color': sysid_color, 'zorder': 5, 'linestyle': '-'},
#         },
#     'Neural SDE' : 
#         {'traj': 'lemn2_traj_NeurSDE_v{}.ulg',
#          'offset_t' : 0.15,
#          'num_traj' : 5,
#          'trajId' : 1,
#         'style': {'color': sde_color, 'zorder': 5, 'linestyle': '-'},
#         },
#     'Neural ODE' : 
#         {'traj': 'lemn2_traj_NeurODE_v{}.ulg',
#          'offset_t' : 0.15,
#          'num_traj' : 5,
#          'trajId' : 1,
#         'style': {'color': ode_color, 'zorder': 5, 'linestyle': '-'},
#         },
# }
# plot_trajs_multi(controllers, 'fast2_lemn.csv', spacing=1, tmin=5, tmax=30,
#                  rmse_compute=False, tikz=False, dpi=500, outname='Fig8_iris.png', figsize=(12, 8))

# controllers = {
#     'Sys ID' : 
#         {'traj': 'circle2_traj_SysId_v{}.ulg',
#         #  'offset_t' : 0.1,
#          'num_traj' : 6,
#          'trajId' : 1,
#         'style': {'color': sysid_color, 'zorder': 5, 'linestyle': '-'},
#         },
#     'Neural SDE' : 
#         {'traj': 'circle2_traj_NeurSDE_v{}.ulg',
#          'offset_t' : 0.13,
#          'num_traj' : 6,
#          'trajId' : 2,
#         'style': {'color': sde_color, 'zorder': 5, 'linestyle': '-'},
#         },
#     'Neural ODE' : 
#         {'traj': 'circle2_traj_NeurODE_v{}.ulg',
#          'offset_t' : 0.13,
#          'num_traj' : 6,
#          'trajId' : 1,
#         'style': {'color': ode_color, 'zorder': 5, 'linestyle': '-'},
#         },
# }
# plot_trajs_multi(controllers, 'fast2_circle.csv', spacing=1, 
#                  tmin=5, tmax=30, rmse_compute=False, 
#                  outname='circle_iris.png', dpi=500, figsize=(12, 8), tikz=False)


def n_steps_analysis(xtraj, utraj, jit_sampling_fn, time_evol, data_stepsize, traj_time_evol):
    """Compute the time evolution of the mean and variance of the SDE at each time step

    Args:
        xtraj (TYPE): The trajectory of the states
        utraj (TYPE): The trajectory of the inputs
        jit_sampling_fn (TYPE): The sampling function return an array of size (num_particles, horizon, state_dim)
        time_evol (TYPE): The time evolution of the sampling technique

    Returns:
        TYPE: The multi-sampled state evolution
        TYPE: The time step evolution for plotting
    """
    sampler_horizon = len(time_evol) - 1
    dt_sampler = time_evol[1] - time_evol[0]
    # Check if dt_sampler and data_stepsize are close enough
    if abs(dt_sampler - data_stepsize) < 1e-5:
        quot = 1
    else:
        assert dt_sampler > data_stepsize-1e-5, "The time step of the sampling function must be larger than the data step size"
        assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
        quot = dt_sampler / data_stepsize

    # print(dt_sampler, data_stepsize, dt_sampler % sampler_horizon, sampler_horizon % dt_sampler)
    # assert dt_sampler > data_stepsize-1e-6, "The time step of the sampling function must be larger than the data step size"
    # assert abs(dt_sampler % data_stepsize) <= 1e-6, "The time step of the sampling function must be a multiple of the data step size"
    quot = dt_sampler / data_stepsize
    # Take the closest integer to quot
    num_steps2data  = int(quot + 0.5)
    # Compute the actual horizon for splitting the trajectories
    traj_horizon = num_steps2data * sampler_horizon
    # Split the trajectory into chunks of size num_steps2data
    total_traj_size = (xtraj.shape[0] // traj_horizon) * traj_horizon
    xevol = xtraj[:total_traj_size+1]
    uevol = utraj[:total_traj_size]
    uevol = uevol.reshape(-1, sampler_horizon, num_steps2data, uevol.shape[-1])
    xevol_full = [ xevol[i:(i+traj_horizon)] for i in range(0, total_traj_size+1, traj_horizon)]
    tevol_full = [ traj_time_evol[i:(i+traj_horizon)] for i in range(0, total_traj_size+1, traj_horizon)]
    xevol = xevol[::traj_horizon]
    # Reshape the time evolution
    m_tevol = traj_time_evol[:total_traj_size+1][::traj_horizon]
    print(xevol.shape)
    print(uevol.shape)
    # assert xevol.shape[0] == uevol.shape[0], "The number of trajectories must be the same for the states and inputs"
    # Initial random number generator
    rng = jax.random.PRNGKey(0)
    rng, s_rng = jax.random.split(rng)
    xres = []
    tres = []
    for i in range(uevol.shape[0]):
        rng, s_rng = jax.random.split(rng)
        # _curr_u = np.mean(uevol[i], axis=-2)
        _curr_u = uevol[i,:,0,:]
        _curr_x = xevol[i]
        _xpred = np.array(jit_sampling_fn(_curr_x, _curr_u, s_rng)) # (num_particles, horizon+1, state_dim)
        _tevol = m_tevol[i] + time_evol
        if i < xevol.shape[0]-1:
            _xpred = _xpred[:,:-1,:]
            _tevol = _tevol[:-1]
        xres.append(_xpred)
        tres.append(_tevol)

    # # Merge the results along the horizon axis
    xres = np.concatenate(xres, axis=1)
    tres = np.concatenate(tres, axis=0)
    return [xres,], [tres,], [xtraj,], [traj_time_evol,]
    # return xres, tres, xevol_full, tevol_full


def plot_state_evol(spacing=1, path_traj='my_data/lemn2_traj_NeurSDE_v1.ulg', 
                    t_init=None, t_end=None, nrow=4, ncol=3, indx_plot=None,
                    tikz=False, dpi=500, figsize=(12, 8), outname='iris_state_evol.png'):
    """ Show the SDE predicted state
    """

    from sde4mbrlExamples.rotor_uav.sde_rotor_model import load_predictor_function, load_trajectory
    _traj_x, _traj_u = load_trajectory(path_traj, 
                                       outlier_cond=lambda d : d['z']>1.1,
                                        min_length=500)  # Only the first trajectory
    traj_data = {'y' : _traj_x[0], 'u' : _traj_u[0]}
    data_stepsize = 0.01
    traj_time_evol = np.array([i*data_stepsize for i in range(traj_data['y'].shape[0])])

    sde_path = '/home/franckdjeumou/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_models/iris_sitl_sde.pkl'
    ode_path = '/home/franckdjeumou/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_models/iris_sitl_ode_sde.pkl'
    
    modified_params = {'horizon' : 100, 'num_particles' : 10, 'stepsize': 0.01,
                       'noise_prior_params': [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.02, 0.02, 0.01]
                       }
    _sys_id_model, sys_id_times = load_predictor_function(sde_path, prior_dist=True, nonoise=True, modified_params= {**modified_params, 'num_particles' : 1}, return_time_steps=True)
    sys_id_model = jax.jit(_sys_id_model)

    _ode_model, ode_times = load_predictor_function(ode_path,  modified_params= {**modified_params, 'num_particles' : 1}, return_time_steps=True)
    ode_model = jax.jit(_ode_model)
    
    _posterior_model, post_times = load_predictor_function(sde_path, prior_dist=False, modified_params= modified_params, return_time_steps=True)
    posterior_model = jax.jit(_posterior_model)
    # Get the time evolution of the SDE
    xevol, tevol, _xfull, _tfull = n_steps_analysis(traj_data['y'], traj_data['u'], posterior_model, post_times, data_stepsize, traj_time_evol)
    # Get the time evolution of the sys_id
    xevol_sysid, tevol_sysid, _, _ = n_steps_analysis(traj_data['y'], traj_data['u'], sys_id_model, sys_id_times, data_stepsize, traj_time_evol)
    # Get the time evolution of the ode
    xevol_ode, tevol_ode, _, _ = n_steps_analysis(traj_data['y'], traj_data['u'], ode_model, ode_times, data_stepsize, traj_time_evol)

    indx_traj = 0
    # t_init, t_end = None , None
    # # t_init, t_end = 47, 60
    # t_init, t_end = 42, 60

    xevol = xevol[indx_traj]
    tevol = tevol[indx_traj]
    m_idx = None if t_init is None or t_end  is None else (tevol >= t_init) & (tevol <= t_end)
    xevol = xevol[:, m_idx, :] if m_idx is not None else xevol
    tevol = tevol[m_idx] if m_idx is not None else tevol

    gt_data = _xfull[indx_traj]
    gt_time = _tfull[indx_traj]
    m_idx_gt = None if t_init is None or t_end  is None else np.logical_and(gt_time >= t_init, gt_time <= t_end)
    gt_data = gt_data[m_idx_gt] if m_idx_gt is not None else gt_data
    gt_time = gt_time[m_idx_gt] if m_idx_gt is not None else gt_time

    xevol_sysid = xevol_sysid[indx_traj] if m_idx is None else xevol_sysid[indx_traj][:,m_idx,:]
    tevol_sysid = tevol_sysid[indx_traj] if m_idx is None else tevol_sysid[indx_traj][m_idx]

    xevol_ode = xevol_ode[indx_traj] if m_idx is None else xevol_ode[indx_traj][:,m_idx,:]
    tevol_ode = tevol_ode[indx_traj] if m_idx is None else tevol_ode[indx_traj][m_idx]

    # Plot the full state evolution but first transfor the quaternion to euler angles
    rpy = []
    for _q_s in xevol[:,:,6:10]:
        _rpy = []
        for _q in _q_s:
            _rpy.append(list(quat_to_euler(_q, np)))
        rpy.append(_rpy)
    rpy = np.array(rpy) * 180.0 / np.pi
    xevol = np.concatenate([xevol, rpy], axis=-1)

    # Do the same for the sys_id
    rpy = []
    for _q_s in xevol_sysid[:,:,6:10]:
        _rpy = []
        for _q in _q_s:
            _rpy.append(list(quat_to_euler(_q, np)))
        rpy.append(_rpy)
    rpy = np.array(rpy) * 180.0 / np.pi
    xevol_sysid = np.concatenate([xevol_sysid, rpy], axis=-1)

    # Do the same for the ode
    rpy = []
    for _q_s in xevol_ode[:,:,6:10]:
        _rpy = []
        for _q in _q_s:
            _rpy.append(list(quat_to_euler(_q, np)))
        rpy.append(_rpy)
    rpy = np.array(rpy) * 180.0 / np.pi
    xevol_ode = np.concatenate([xevol_ode, rpy], axis=-1)

    # Do the same for the groundtruth
    rpy = []
    for q in gt_data[:,6:10]:
        rpy.append(list(quat_to_euler(q, np)))
    rpy = np.array(rpy) * 180 / np.pi
    gt_data = np.concatenate([gt_data, rpy], axis=-1)

    # State to plot
    state2plot = {  0 : r'x [m]', 1 : r'y [m]', 2 : r'z [m]', 
                    3 : r'$v_x$ [m/s]', 4 : r'$v_y$ [m/s]', 5 : r'$v_z$ [m/s]', 
                    13 : r'$\phi$ [deg]', 14 : r'$\theta$ [deg]', 15 : r'$\psi$ [deg]',
                    10 :  r'$\omega_x$ [rad/s]', 11 : r'$\omega_y$ [rad/s]', 12 : r'$\omega_z$ [rad/s]'
                }
    indxEq = [0, 1, 2, 3, 4, 5, 13, 14, 15, 10, 11, 12] if indx_plot is None else indx_plot
    
    # Colors scheme
    sde_color = '#157F1F' # 'r'
    sysid_color = '#8D8D92' # 'b'
    gt_color = 'k'
    ode_color = '#00A7E1' # 'g'

    # Create the figure and axes
    fig, _axs = plt.subplots(nrow, ncol, constrained_layout=True, figsize=figsize, sharex=True)
    axs = _axs.flatten()
    for i, ax in zip(indxEq, axs):
        ax.plot(gt_time[::spacing], gt_data[::spacing,i], color=gt_color,  label='Ground Truth' if i == 0 else None, zorder=100)
        for j in range(xevol.shape[0]):
            ax.plot(tevol[::spacing], xevol[j,::spacing,i], color=sde_color, 
                    label = 'Neural SDE' if j == 0 and i == 0 else None, zorder=10)
        for j in range(xevol_sysid.shape[0]):
            ax.plot(tevol_sysid[::spacing], xevol_sysid[j,::spacing,i], color=sysid_color,
                    label = 'Sys ID' if j == 0 and i == 0 else None, zorder=30)
        for j in range(xevol_ode.shape[0]):
            ax.plot(tevol_ode[::spacing], xevol_ode[j,::spacing,i], color=ode_color,
                    label = 'Neural ODE' if j == 0 and i == 0 else None, zorder=50)
        ax.set_ylabel(state2plot[i])
        ax.grid(True)
        if i >= 9:
            ax.set_xlabel('t [s]')
    
    # Let's center the figure legend
    lines = []
    labels = []
    for ax in axs:
        l, l_ = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(l_)
    fig.legend(lines, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))

    # Save the figure as a png
    fig.savefig('my_data/'+outname, dpi = dpi, bbox_inches='tight', transparent=True)

    # Export the figure as tikz
    if tikz:
        import tikzplotlib
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.clean_figure(fig)
        outname = outname.replace('.png', '.tex') if outname.endswith('.png') else outname + '.tex'
        tikzplotlib.save('my_data/'+outname, figure=fig)
    
    plt.show()

# plot_state_evol(spacing=1, path_traj='my_data/circle2_traj_NeurSDE_v1.ulg', 
#                     t_init=28.05, t_end=40, nrow=2, ncol=3, indx_plot=[0,1,2,3,10,11],
#                     tikz=False, dpi=500, figsize=(12,6), outname='iris_state_evol.png')

# plot_state_evol(spacing=20, path_traj='my_data/circle2_traj_NeurSDE_v1.ulg', 
#                     t_init=28.05, t_end=40, nrow=2, ncol=3, indx_plot=[0,1,2,3,10,11],
#                     tikz=True, dpi=500, figsize=(12,6), outname='iris_state_evol.png')