import os

# MPC seems to be faster on cpu because of the loop
# TODO: check if this is still true, and investiage how to make it faster on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np

import matplotlib.pyplot as plt

from sde4mbrlExamples.rotor_uav.utils import *

from matplotlib.gridspec import GridSpec

def get_size_paper(width_pt, fraction=1, subplots=(1,1)):
    """ Get the size of the figure in inches
        width_pt: Width of the figure in latex points
        fraction: Fraction of the width which you wish the figure to occupy
        subplots: The number of rows and columns
    """
    # Width of the figure in inches
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

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
    # # Parse the ulog file
    # topic = 'mpc_full_state'
    # ulog = ULog(os.path.expanduser(ulog_file), message_name_filter_list=[topic], disable_str_exceptions=True)
    # # Check if the ulog is valid and not empty
    # if len(ulog.data_list) <= 0:
    #     raise ValueError("The ulog file is empty.")
    # msg_pose = ulog.data_list[0]
    # timestamp_pose = msg_pose.data['timestamp'] * 1e-6

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
    f = interp1d(tref, ref_xyz, axis=0)
    indx_in = (tref[0] < tveh) & (tref[-1] > tveh)
    # Get the interpolated reference trajectory
    ref_xyz_interp = f(tveh[indx_in])
    # Compute the RMSE
    rmse = np.sqrt(np.sum((veh_xyz[indx_in] - ref_xyz_interp)**2, axis=1))
    return rmse, tveh[indx_in]

def plot_simple_corl(ref_path, traj_path, spacing=1,z=None):
    """ Do a simple plot to illustrate the result of the real world experiment \.
        The layout of the plot looks like this:
        |   (x,y), (Pos,t)  |
        |   (x,y), (Vel,t)  |
        |   (z,t), (rpy,t)  | 
    """
    # Load the reference trajectory first
    dir_for_path = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_data/'
    _ref_path = os.path.expanduser(dir_for_path + ref_path)
    time_ref, state_ref = parse_trajectory(_ref_path)

    # Load the trajectory
    dir_for_logs = '~/Documents/log_flights/'
    t_drone, drone_traj = load_mpc_part_from_ulog(os.path.expanduser(dir_for_logs + traj_path))
    if z is not None:
        state_ref[:,2] = z

    # General plot settings
    general_style = {'linewidth': 2.0,
                     'markersize': 1.0,
                    }
    ref_line_style = '--'
    ref_color = '#000000'
    ref_zorder = 100
    drone_color = '#157F1F'
    drone_line_style = '-'
    drone_zorder = 10
    lwidth = 1.5
    # lwidth= 3

    # Get the figure size
    nrows, ncols = 1, 3
    figsize = (14, 8) # None

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
    gs = GridSpec(nrows, ncols, figure=fig, wspace=None, hspace=None, width_ratios=None, height_ratios=None)

    # Plot the position in the xy plane on the first 2 rows and first column
    ax_xy = fig.add_subplot(gs[0,0])
    ax_plot(ax_xy, state_ref[:,0], state_ref[:,1], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
    ax_plot(ax_xy, drone_traj[:,0], drone_traj[:,1], label='NMPC+SDE', color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
    # Set the axes labels and grid
    ax_xy.set_xlabel(r'$p_x$')
    ax_xy.set_ylabel(r'$p_y$')
    # ax_xy.legend()

    # Plot the position error on the first row and second column, the reference is the 0 line
    print(np.min(t_drone[1:] - t_drone[:-1]), np.max(t_drone[1:] - t_drone[:-1]))
    pos_err, tpos_err = compute_rmse(state_ref[:,0:3], time_ref, drone_traj[:,0:3], t_drone)
    # Cumulative mean of the position error
    pos_err = np.cumsum(pos_err) / np.arange(1, len(pos_err) + 1)
    ax_xyz_err = fig.add_subplot(gs[0,1])
    ax_plot(ax_xyz_err, [time_ref[0], time_ref[-1]], [0, 0], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
    ax_plot(ax_xyz_err, tpos_err, pos_err, color=drone_color, label='NMPC+SDE', linestyle=drone_line_style, zorder=drone_zorder)
    # ax_xyz_err.hlines(0, time_ref[0], time_ref[-1], color=ref_color, linestyle=ref_line_style, **general_style)
    # Set the axes labels and grid
    # ax_xyz_err.set_xlabel('t [s]')
    ax_xyz_err.set_ylabel(r'RMSE')
    # ax_xyz_err.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x*100)))
    # ax_xyz_err.legend()
    # set no x ticks labels
    ax_xyz_err.set_xticklabels([])
    # ax_xyz_err.legend(frameon=False, handletextpad=0.2)

    # Vlim
    Vlimit = 1.71
    # Plot the velocity error on the second row and second column, the reference is the 0 line
    vel_value = np.linalg.norm(state_ref[:,3:6], axis=1)
    drone_vel_value = np.linalg.norm(drone_traj[:,3:6], axis=1)
    ax_vel = fig.add_subplot(gs[0,2])
    ax_plot(ax_vel, time_ref, vel_value, color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
    ax_plot(ax_vel, t_drone, drone_vel_value, color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
    # Draw the limit line
    ax_vel.hlines(Vlimit, time_ref[0], time_ref[-1], color='r', linestyle='--')
    # Set the axes labels and grid
    # ax_vel.set_xlabel('t [s]')
    ax_vel.set_ylabel(r'V')
    # ax_vel.legend()
    # set no x ticks labels
    # ax_vel.set_xticklabels([])
    # Set the yticks max to 4
    ax_vel.set_ylim([0, 4])
    ax_vel.set_xlabel(r'Time')

    # # Save the figure as png with tight layout and dpi=500
    filename = ref_path.split('.')[0]

    # Export the figure as tikz
    import tikzplotlib
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure(fig)
    outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.tex'
    tikzplotlib.save(os.path.expanduser(outname), figure=fig)

    plt.show()

# def plot_simple(ref_path, traj_path, spacing=1, z=None, wrate=False):
#     """ Do a simple plot to illustrate the result of the real world experiment \.
#         The layout of the plot looks like this:
#         |   (x,y), (Pos,t)  |
#         |   (x,y), (Vel,t)  |
#         |   (z,t), (rpy,t)  | 
#     """
#     # Load the reference trajectory first
#     dir_for_path = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_data/'
#     _ref_path = os.path.expanduser(dir_for_path + ref_path)
#     time_ref, state_ref = parse_trajectory(_ref_path)
#     if z is not None:
#         state_ref[:,2] = z
    

#     # Load the trajectory
#     dir_for_logs = '~/Documents/log_flights/'
#     t_drone, drone_traj = load_mpc_part_from_ulog(os.path.expanduser(dir_for_logs + traj_path))

#     # General plot settings
#     general_style = {'linewidth': 2.0,
#                      'markersize': 1.0,
#                     }
#     ref_line_style = '--'
#     ref_color = '#000000'
#     ref_zorder = 100
#     drone_color = '#157F1F'
#     drone_line_style = '-'
#     drone_zorder = 10
#     # Fraction scaling for the figure
#     fraction = 1.0
#     paper_width_pt = 433.62
#     paper_width_pt = None
#     lwidth = 1.5
#     # lwidth= 3

#     # texConfig = {
#     #     'text.usetex' : True,
#     #     'font.family' : 'serif',
#     #     'font.size' : 10.95,
#     #     'axes.labelsize' : 9,
#     #     # Make the legend / label fonts a little smaller
#     #     'xtick.labelsize' : 7,
#     #     'ytick.labelsize' : 7,
#     #     'legend.fontsize' : 9,
#     #     'axes.titlesize' : 9,
#     #     'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}',
#     # }

#     # Powerpoint
#     texConfig = {
#         'text.usetex' : True,
#         'font.family' : 'serif',
#         'font.size' : 17,
#         'axes.labelsize' : 15,
#         # Make the legend / label fonts a little smaller
#         'xtick.labelsize' : 12,
#         'ytick.labelsize' : 12,
#         'legend.fontsize' : 16,
#         'axes.titlesize' : 16,
#         'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}',
#     }

#     # Set the matplotlib rcParams
#     plt.rcParams.update(texConfig)

#     # Get the figure size
#     nrows, ncols = 3, 4
#     if paper_width_pt is None:
#         figsize = (14, 8) # None
#     else:
#         figsize = get_size_paper(paper_width_pt, fraction=fraction, subplots=(2,2))
#         # figsize = (figsize[0], figsize[1] * 0.75)

#     def ax_plot(ax, x, y, _spacing=spacing, **kwargs):
#         assert len(x) == len(y), 'x and y must have the same length'
#         if len(x) < _spacing:
#             ax.plot(x, y, **{**general_style, **kwargs})
#             ax.grid(True)
#             return
#         ax.plot(x[::_spacing], y[::_spacing], **{**general_style, **kwargs})
#         ax.grid(True)

#     # We are making a gridplot for our plots
#     fig = plt.figure(constrained_layout=True, figsize=figsize)
#     gs = GridSpec(nrows, ncols, figure=fig, wspace=None, hspace=None, width_ratios=None, height_ratios=None)

#     # Plot the position in the xy plane on the first 2 rows and first column
#     ax_xy = fig.add_subplot(gs[:2,:2])
#     ax_plot(ax_xy, state_ref[:,0], state_ref[:,1], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_xy, drone_traj[:,0], drone_traj[:,1], label='NMPC+SDE', color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#     # Set the axes labels and grid
#     ax_xy.set_xlabel(r'x')
#     ax_xy.set_ylabel(r'y')
#     # ax_xy.legend()

#     # Plot the z position on the last row and first column
#     ax_z = fig.add_subplot(gs[2,0])
#     ax_plot(ax_z, time_ref, state_ref[:,2], color=ref_color, linestyle=ref_line_style, zorder=ref_zorder, linewidth=lwidth)
#     ax_plot(ax_z, t_drone, drone_traj[:,2], color=drone_color, linestyle=drone_line_style, zorder=drone_zorder, linewidth=lwidth)
#     # Set the axes labels and grid
#     ax_z.set_xlabel(r'Time')
#     ax_z.set_ylabel(r'z')
#     # ax_z.legend()

#     # Plot the position error on the first row and second column, the reference is the 0 line
#     print(np.min(t_drone[1:] - t_drone[:-1]), np.max(t_drone[1:] - t_drone[:-1]))
#     pos_err, tpos_err = compute_rmse(state_ref[:,0:3], time_ref, drone_traj[:,0:3], t_drone)
#     # Cumulative mean of the position error
#     pos_err = np.cumsum(pos_err) / np.arange(1, len(pos_err) + 1)
#     ax_xyz_err = fig.add_subplot(gs[0,2:])
#     ax_plot(ax_xyz_err, [time_ref[0], time_ref[-1]], [0, 0], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_xyz_err, tpos_err, pos_err, color=drone_color, label='NMPC+SDE', linestyle=drone_line_style, zorder=drone_zorder)
#     # ax_xyz_err.hlines(0, time_ref[0], time_ref[-1], color=ref_color, linestyle=ref_line_style, **general_style)
#     # Set the axes labels and grid
#     # ax_xyz_err.set_xlabel('t [s]')
#     ax_xyz_err.set_ylabel(r'RMSE $\cdot 10^{-2}$')
#     ax_xyz_err.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x*100)))
#     # ax_xyz_err.legend()
#     # set no x ticks labels
#     ax_xyz_err.set_xticklabels([])
#     # ax_xyz_err.legend(frameon=False, handletextpad=0.2)

#     # Plot the velocity error on the second row and second column, the reference is the 0 line
#     vel_value = np.linalg.norm(state_ref[:,3:6], axis=1)
#     drone_vel_value = np.linalg.norm(drone_traj[:,3:6], axis=1)
#     ax_vel = fig.add_subplot(gs[1,2:])
#     ax_plot(ax_vel, time_ref, vel_value, color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_vel, t_drone, drone_vel_value, color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#     # Set the axes labels and grid
#     # ax_vel.set_xlabel('t [s]')
#     ax_vel.set_ylabel(r'V')
#     # ax_vel.legend()
#     # set no x ticks labels
#     # ax_vel.set_xticklabels([])
#     # Set the yticks max to 4
#     ax_vel.set_ylim([0, 4])
#     ax_vel.set_xlabel(r'Time')

#     # Now we will plot the roll pitch and yaw on the last row and second column
#     wrate = False

#     coeff_spacing = 1
#     roll_marker_style = 'o'
#     pitch_marker_style = 'x'
#     yaw_marker_style = 'd'
#     rpy_line_style = '--'
#     drone_rpy_line_style = '-'
#     # ax_rpy = fig.add_subplot(gs[2,1])
#     ax_roll = fig.add_subplot(gs[2,1])
#     ax_pitch = fig.add_subplot(gs[2,2])
#     ax_yaw = fig.add_subplot(gs[2,3])

#     state_ref[:,13:] = state_ref[:,13:] * 180 / np.pi
#     drone_traj[:,13:] = drone_traj[:,13:] * 180 / np.pi

#     max_v = np.max(drone_traj, axis= 0)
#     min_v = np.min(drone_traj, axis= 0)
#     print('Max Value: ', max_v)
#     print('Min Value: ', min_v)

    
#     ax_plot(ax_roll, time_ref, state_ref[:,13], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=ref_color, zorder=ref_zorder, linewidth=lwidth)
#     ax_plot(ax_pitch, time_ref, state_ref[:,14], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=ref_color, zorder=ref_zorder, linewidth=lwidth)
#     ax_plot(ax_yaw, time_ref, state_ref[:,15], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=ref_color,zorder=ref_zorder, linewidth=lwidth)
    

#     ax_plot(ax_roll, t_drone, drone_traj[:,13], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=drone_color, zorder=drone_zorder, linewidth=lwidth)
#     ax_plot(ax_pitch, t_drone, drone_traj[:,14], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=drone_color, zorder=drone_zorder, linewidth=lwidth)
#     ax_plot(ax_yaw, t_drone, drone_traj[:,15], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=drone_color, zorder=drone_zorder, linewidth=lwidth)

#     # Set the x axis label
#     ax_roll.set_xlabel(r'Time')
#     ax_pitch.set_xlabel(r'Time')
#     ax_yaw.set_xlabel(r'Time')
    
#     # Set the ylabel
#     ax_roll.set_ylabel(r'$\phi \cdot 10$')
#     ax_pitch.set_ylabel(r'$\theta \cdot 10$')
#     ax_yaw.set_ylabel(r'$\Psi \cdot 10$')
#     ax_roll.set_ylim([-40, 40])
#     ax_pitch.set_ylim([-40, 40])

#     ax_roll.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x/10)))
#     ax_pitch.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x/10)))
#     ax_yaw.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x/10)))
    
#     # ax_rpy.set_xlabel(r'Time [s]')
#     # ax_rpy.set_ylabel(r'Angle [deg]')
#     # ax_rpy.grid(True)
#     # ax_rpy.legend(ncols=2, frameon=False, markerscale=1.5, handletextpad=-0.5, columnspacing=0.1, loc = 'upper left', bbox_to_anchor=(0.0, 1.1))

#     # Save the figure as png with tight layout and dpi=500
#     filename = ref_path.split('.')[0] if not wrate else ref_path.split('.')[0] + '_wrate'
#     outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.png'
#     fig.savefig(os.path.expanduser(outname), dpi=500, bbox_inches='tight', transparent=True)

#     # # Save the pdf version
#     # outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.pdf'
#     # fig.savefig(os.path.expanduser(outname), format='pdf', bbox_inches='tight')

#     # # Export the figure as tikz
#     # import tikzplotlib
#     # tikzplotlib_fix_ncols(fig)
#     # tikzplotlib.clean_figure(fig)
#     # outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.tex'
#     # tikzplotlib.save(os.path.expanduser(outname), figure=fig)

#     plt.show()

# def plot_simple(ref_path, traj_path, spacing=1, z=None, wrate=False):
#     """ Do a simple plot to illustrate the result of the real world experiment \.
#         The layout of the plot looks like this:
#         |   (x,y), (Pos,t)  |
#         |   (x,y), (Vel,t)  |
#         |   (z,t), (rpy,t)  | 
#     """
#     # Load the reference trajectory first
#     dir_for_path = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/iris_sitl/my_data/'
#     _ref_path = os.path.expanduser(dir_for_path + ref_path)
#     time_ref, state_ref = parse_trajectory(_ref_path)
#     if z is not None:
#         state_ref[:,2] = z
    

#     # Load the trajectory
#     dir_for_logs = '~/Documents/log_flights/'
#     t_drone, drone_traj = load_mpc_part_from_ulog(os.path.expanduser(dir_for_logs + traj_path))

#     # General plot settings
#     general_style = {'linewidth': 2.0,
#                      'markersize': 1.0,
#                     }
#     ref_line_style = '--'
#     ref_color = '#000000'
#     ref_zorder = 100
#     drone_color = 'red' # '#157F1F'
#     drone_line_style = '-'
#     drone_zorder = 10
#     # Fraction scaling for the figure
#     fraction = 1.0
#     paper_width_pt = 433.62
#     # paper_width_pt = None

#     texConfig = {
#         'text.usetex' : True,
#         'font.family' : 'serif',
#         'font.size' : 10.95,
#         'axes.labelsize' : 9,
#         # Make the legend / label fonts a little smaller
#         'xtick.labelsize' : 7,
#         'ytick.labelsize' : 7,
#         'legend.fontsize' : 9,
#         'axes.titlesize' : 9,
#         'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}',
#     }

#     # Set the matplotlib rcParams
#     plt.rcParams.update(texConfig)

#     # Get the figure size
#     nrows, ncols = 3, 2
#     if paper_width_pt is None:
#         figsize = None
#     else:
#         figsize = get_size_paper(paper_width_pt, fraction=fraction, subplots=(2,2))
#         # figsize = (figsize[0], figsize[1] * 0.75)

#     def ax_plot(ax, x, y, _spacing=spacing, **kwargs):
#         assert len(x) == len(y), 'x and y must have the same length'
#         if len(x) < _spacing:
#             ax.plot(x, y, **{**general_style, **kwargs})
#             ax.grid(True)
#             return
#         ax.plot(x[::_spacing], y[::_spacing], **{**general_style, **kwargs})
#         ax.grid(True)

#     # We are making a gridplot for our plots
#     fig = plt.figure(constrained_layout=True, figsize=figsize)
#     gs = GridSpec(nrows, ncols, figure=fig, wspace=None, hspace=None, width_ratios=[3,5], height_ratios=None)

#     # Plot the position in the xy plane on the first 2 rows and first column
#     ax_xy = fig.add_subplot(gs[:2,0])
#     ax_plot(ax_xy, state_ref[:,0], state_ref[:,1], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_xy, drone_traj[:,0], drone_traj[:,1], label='NMPC+SDE', color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#     # Set the axes labels and grid
#     ax_xy.set_xlabel(r'x [m]')
#     ax_xy.set_ylabel(r'y [m]')
#     # ax_xy.legend()

#     # Plot the z position on the last row and first column
#     ax_z = fig.add_subplot(gs[2,0])
#     ax_plot(ax_z, time_ref, state_ref[:,2], color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_z, t_drone, drone_traj[:,2], color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#     # Set the axes labels and grid
#     ax_z.set_xlabel(r'Time [s]')
#     ax_z.set_ylabel(r'z [m]')
#     # ax_z.legend()

#     # Plot the position error on the first row and second column, the reference is the 0 line
#     print(np.min(t_drone[1:] - t_drone[:-1]), np.max(t_drone[1:] - t_drone[:-1]))
#     pos_err, tpos_err = compute_rmse(state_ref[:,0:3], time_ref, drone_traj[:,0:3], t_drone)
#     # Cumulative mean of the position error
#     pos_err = np.cumsum(pos_err) / np.arange(1, len(pos_err) + 1)
#     ax_xyz_err = fig.add_subplot(gs[0,1])
#     ax_plot(ax_xyz_err, [time_ref[0], time_ref[-1]], [0, 0], label='Reference', color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_xyz_err, tpos_err, pos_err, color=drone_color, label='NMPC+SDE', linestyle=drone_line_style, zorder=drone_zorder)
#     # ax_xyz_err.hlines(0, time_ref[0], time_ref[-1], color=ref_color, linestyle=ref_line_style, **general_style)
#     # Set the axes labels and grid
#     # ax_xyz_err.set_xlabel('t [s]')
#     ax_xyz_err.set_ylabel(r'RMSE [m]')
#     # ax_xyz_err.legend()
#     # set no x ticks labels
#     ax_xyz_err.set_xticklabels([])
#     ax_xyz_err.legend(frameon=False, handletextpad=0.2)

#     # Plot the velocity error on the second row and second column, the reference is the 0 line
#     vel_value = np.linalg.norm(state_ref[:,3:6], axis=1)
#     drone_vel_value = np.linalg.norm(drone_traj[:,3:6], axis=1)
#     ax_vel = fig.add_subplot(gs[1,1])
#     ax_plot(ax_vel, time_ref, vel_value, color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#     ax_plot(ax_vel, t_drone, drone_vel_value, color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#     # Set the axes labels and grid
#     # ax_vel.set_xlabel('t [s]')
#     ax_vel.set_ylabel(r'V [m/s]')
#     # ax_vel.legend()
#     # set no x ticks labels
#     ax_vel.set_xticklabels([])
#     # Set the yticks max to 4
#     ax_vel.set_ylim([0, 4])

#     # Now we will plot the roll pitch and yaw on the last row and second column
#     if not wrate:
#         coeff_spacing = 2
#         roll_marker_style = 'o'
#         pitch_marker_style = 'x'
#         yaw_marker_style = 'd'
#         rpy_line_style = 'None'
#         drone_rpy_line_style = '-'
#         ax_rpy = fig.add_subplot(gs[2,1])
#         state_ref[:,13:] = state_ref[:,13:] * 180 / np.pi
#         drone_traj[:,13:] = drone_traj[:,13:] * 180 / np.pi

#         # color palette for roll, pitch and yaw
#         roll_c, pitch_c, yaw_c = ref_color, ref_color, ref_color
#         # roll_c, pitch_c, yaw_c = '#FF0000', '#0000FF', '#00FF00'
#         # I need 3 shades of the green color
        
#         ax_plot(ax_rpy, time_ref, state_ref[:,13], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=roll_c, marker=roll_marker_style, label='Roll', zorder=ref_zorder)
#         ax_plot(ax_rpy, time_ref, state_ref[:,14], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=pitch_c, marker=pitch_marker_style, label='Pitch', zorder=ref_zorder)
#         ax_plot(ax_rpy, time_ref, state_ref[:,15], _spacing=coeff_spacing*spacing, linestyle=rpy_line_style, color=yaw_c, marker=yaw_marker_style, zorder=ref_zorder, label='Yaw')
        
#         # Sufficiently distinct green colors
#         # roll_c, pitch_c, yaw_c = '#00FF00', '#00AA00', '#005500'
#         # Distinct shade of red colors
#         # roll_c, pitch_c, yaw_c = '#FF0000', '#AA0000', '#DD0000'
#         roll_c, pitch_c, yaw_c = drone_color, drone_color, drone_color

#         ax_plot(ax_rpy, t_drone, drone_traj[:,13], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=roll_c, zorder=drone_zorder, linewidth=1.5)
#         ax_plot(ax_rpy, t_drone, drone_traj[:,14], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=pitch_c, zorder=drone_zorder, linewidth=1.5)
#         ax_plot(ax_rpy, t_drone, drone_traj[:,15], _spacing=coeff_spacing*spacing, linestyle=drone_rpy_line_style, color=yaw_c, zorder=drone_zorder, linewidth=1.5)
        
#         ax_rpy.set_xlabel(r'Time [s]')
#         ax_rpy.set_ylabel(r'Angle [deg]')
#         ax_rpy.grid(True)
#         ax_rpy.legend(ncols=2, frameon=False, markerscale=1.5, handletextpad=-0.5, columnspacing=0.1, loc = 'upper left', bbox_to_anchor=(0.0, 1.1))
#     else:
#         # Plot the magnitude of the angular velocity
#         wrate_value = np.linalg.norm(state_ref[:,10:13], axis=1)
#         drone_wrate_value = np.linalg.norm(drone_traj[:,10:13], axis=1)
#         ax_wrate = fig.add_subplot(gs[2,1])
#         ax_plot(ax_wrate, time_ref, wrate_value, color=ref_color, linestyle=ref_line_style, zorder=ref_zorder)
#         ax_plot(ax_wrate, t_drone, drone_wrate_value, color=drone_color, linestyle=drone_line_style, zorder=drone_zorder)
#         # Set the axes labels and gridcoeff_spacing
#         ax_wrate.set_xlabel(r'Time [s]')
#         ax_wrate.set_ylabel(r'Angular velocity [rad/s]')
#         # ax_wrate.legend()

#     # Save the figure as png with tight layout and dpi=500
#     filename = ref_path.split('.')[0] if not wrate else ref_path.split('.')[0] + '_wrate'
#     outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.png'
#     fig.savefig(os.path.expanduser(outname), dpi=500, bbox_inches='tight')

#     # Save the pdf version
#     outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.pdf'
#     fig.savefig(os.path.expanduser(outname), format='pdf', bbox_inches='tight')

#     # Export the figure as tikz
#     import tikzplotlib
#     tikzplotlib_fix_ncols(fig)
#     tikzplotlib.clean_figure(fig)
#     outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + filename + '.tex'
#     tikzplotlib.save(os.path.expanduser(outname), figure=fig)

#     plt.show()

def plot_training_trajectories(traj_paths, spacing=10, wrate=False):
    my_trajs = []
    my_trajs_time = []
    dir_for_logs = '~/Documents/log_flights/'
    for traj_path in traj_paths:
        time_traj, traj = load_logs(os.path.expanduser(dir_for_logs + traj_path))
        my_trajs.append(traj)
        my_trajs_time.append(time_traj)
    # General plot settings
    general_style = {'linewidth': 3.0,
                        'markersize': 6.0,
                        }
    
    # Let's define the plot style
    bg_c = 'white'
    texConfig = {
        # 'text.usetex' : True,
        'font.family' : 'serif',
        'font.size' : 30.0,
        'font.weight' : 'bold',
        # 'font.color' : 'white',
        'axes.labelsize' : 24,
        # Make the legend / label fonts a little smaller
        'xtick.labelsize' : 20,
        'ytick.labelsize' : 20,
        'legend.fontsize' : 25,
        'legend.labelcolor' : bg_c,
        'axes.labelcolor' : bg_c,
        'xtick.labelcolor' : bg_c,
        'ytick.labelcolor' : bg_c,
        'xtick.color' : bg_c,
        'ytick.color' : bg_c,
        'axes.titlesize' : 9,
        'text.color' : bg_c,
        # 'axes.linewidth' : 3,
        'grid.color' : 'white',
        'grid.linewidth' : 1.0,
        'text.latex.preamble' : r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}',
    }
    # print(plt.rcParams)
    plt.rcParams.update(texConfig)

    # We are making a plot with 2 rows and 1 column
    # The first row shows the magnitude of the velocity
    # The second row shows the roll, pitch and yaw
    fig, axs = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)
    # fcolor = '#7A828D'
    # fig.set_facecolor(fcolor)

    # Plot the velocity magnitude
    ax_vel = axs[0]
    total_duration = 0
    full_vel = []
    for traj, time_traj in zip(my_trajs, my_trajs_time):
        vel_value = np.linalg.norm(traj[:,3:6], axis=1)
        full_vel.append(vel_value)
        duration = time_traj[-1] - time_traj[0]
        total_duration += duration
        print('duration: ', duration)
        time_traj = time_traj[::spacing]
        vel_value = vel_value[::spacing]
        ax_vel.plot(time_traj, vel_value, color='k', linestyle='None', marker='o', **general_style)
    # ax_vel.set_xlabel('Ti [s]')
    ax_vel.set_ylabel('Velocity')
    ax_vel.grid(True)
    ax_vel.set_xticklabels([])
    ax_vel.set_yticklabels([0, 1, 2, 2.9])
    ax_vel.set_yticks([0, 1, 2, 2.9])
    print('total_duration (s): ', total_duration)
    print('total_duration (min): ', total_duration / 60)
    velVal = np.concatenate(full_vel)
    tVal = np.concatenate(my_trajs_time)
    # velVal = np.linalg.norm(traj[:,3:6], axis=1)
    perc_val = 0.95
    perc_V = np.quantile(velVal, perc_val)
    print('{}% of the values are below {}'.format(perc_val * 100, np.quantile(velVal, perc_val)))
    # Draw a horizontal line at 95% of the maximum velocity
    ax_vel.hlines(perc_V, tVal[0], tVal[-1], color='k', linestyle='--', linewidth=3.0, zorder=100)
    

    # Plot the roll pitch and yaw
    ax_rpy = axs[1]
    firs_traj = True
    coeff_spacing = 2
    rollVal, pitchVal, yawVal = [], [], []
    for traj, time_traj in zip(my_trajs, my_trajs_time):
        rpy = traj[:,13:] * 180 / np.pi
        rollVal.append(np.abs(rpy[:,0]))
        pitchVal.append(np.abs(rpy[:,1]))
        yawVal.append(np.abs(rpy[:,2]))
        time_traj = time_traj[::spacing*coeff_spacing]
        rpy = rpy[::spacing*coeff_spacing]
        ax_rpy.plot(time_traj, rpy[:,0], color='r', label='Roll' if firs_traj else None, linestyle='None', marker='o', **general_style)
        ax_rpy.plot(time_traj, rpy[:,1], color='b', label='Pitch' if firs_traj else None, linestyle='None', marker='o', **general_style)
        ax_rpy.plot(time_traj, rpy[:,2], color='g', label='Yaw' if firs_traj else None, linestyle='None', marker='o', **general_style)
        ax_rpy.set_yticks([-80, -50, -20, 0, 20, 30])
        ax_rpy.set_yticklabels([-80, -50, -20, 0, 20, 30])
        # perc_Roll = np.quantile(rpy[:,0], perc_val)


        if firs_traj:
            ax_rpy.legend(ncols=3, frameon=False, columnspacing=0.1, handletextpad=-0.5)
            firs_traj = False
    ax_rpy.set_xlabel('Time')
    ax_rpy.set_ylabel('Euler angles')
    ax_rpy.grid(True)

    rollVal = np.concatenate(rollVal)
    pitchVal = np.concatenate(pitchVal)
    yawVal = np.concatenate(yawVal)
    perc_Roll = np.quantile(rollVal, perc_val)
    perc_Pitch = np.quantile(pitchVal, perc_val)
    perc_Yaw = np.quantile(yawVal, perc_val)

    print('{}% of the values are below {}'.format(perc_val * 100, perc_Roll))
    print('{}% of the values are below {}'.format(perc_val * 100, perc_Pitch))
    print('{}% of the values are below {}'.format(perc_val * 100, perc_Yaw))

    # # Draw a horizontal line for the roll
    # ax_rpy.hlines(perc_Roll, tVal[0], tVal[-1], color='r', linestyle='--', linewidth=3.0, zorder=100)
    # # Draw a horizontal line for the pitch
    # ax_rpy.hlines(perc_Pitch, tVal[0], tVal[-1], color='b', linestyle='--', linewidth=3.0, zorder=100)
    # # Draw a horizontal line for the yaw
    # ax_rpy.hlines(perc_Yaw, tVal[0], tVal[-1], color='g', linestyle='--', linewidth=3.0, zorder=100)

    # for ax in axs:
    #     ax.set_facecolor(fcolor) 

    # Save the figure as png with tight layout and dpi=500
    outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories_slides.png'
    fig.savefig(os.path.expanduser(outname), dpi=500, bbox_inches='tight', transparent=True)

    plt.show()


# def plot_training_trajectories(traj_paths, spacing=10, wrate=False):
#     my_trajs = []
#     my_trajs_time = []
#     dir_for_logs = '~/Documents/log_flights/'
#     for traj_path in traj_paths:
#         time_traj, traj = load_logs(os.path.expanduser(dir_for_logs + traj_path))
#         my_trajs.append(traj)
#         my_trajs_time.append(time_traj)
#     # General plot settings
#     general_style = {'linewidth': 3.0,
#                         'markersize': 5.0,
#                         }
#     # We are making a plot with 2 rows and 1 column
#     # The first row shows the magnitude of the velocity
#     # The second row shows the roll, pitch and yaw
#     fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
#     # Plot the velocity magnitude
#     ax_vel = axs[0]
#     total_duration = 0
#     for traj, time_traj in zip(my_trajs, my_trajs_time):
#         vel_value = np.linalg.norm(traj[:,3:6], axis=1)
#         duration = time_traj[-1] - time_traj[0]
#         total_duration += duration
#         print('duration: ', duration)
#         time_traj = time_traj[::spacing]
#         vel_value = vel_value[::spacing]
#         ax_vel.plot(time_traj, vel_value, color='k', linestyle='None', marker='x', **general_style)
#     ax_vel.set_xlabel('t [s]')
#     ax_vel.set_ylabel('Velocity [m/s]')
#     ax_vel.grid(True)
#     print('total_duration (s): ', total_duration)
#     print('total_duration (min): ', total_duration / 60)

#     # Plot the roll pitch and yaw
#     if not wrate:
#         ax_rpy = axs[1]
#         firs_traj = True
#         coeff_spacing = 2
#         for traj, time_traj in zip(my_trajs, my_trajs_time):
#             rpy = traj[:,13:] * 180 / np.pi
#             time_traj = time_traj[::spacing*coeff_spacing]
#             rpy = rpy[::spacing*coeff_spacing]
#             ax_rpy.plot(time_traj, rpy[:,0], color='r', label='Roll' if firs_traj else None, linestyle='None', marker='o', **general_style)
#             ax_rpy.plot(time_traj, rpy[:,1], color='b', label='Pitch' if firs_traj else None, linestyle='None', marker='x', **general_style)
#             ax_rpy.plot(time_traj, rpy[:,2], color='g', label='Yaw' if firs_traj else None, linestyle='None', marker='*', **general_style)
#             if firs_traj:
#                 ax_rpy.legend()
#                 firs_traj = False
#         ax_rpy.set_xlabel('t [s]')
#         ax_rpy.set_ylabel('Euler angles [deg]')
#         ax_rpy.grid(True)
#     else:
#         ax_wrate = axs[1]
#         firs_traj = True
#         for traj, time_traj in zip(my_trajs, my_trajs_time):
#             wrate_value = np.linalg.norm(traj[:,10:13], axis=1)
#             time_traj = time_traj[::spacing]
#             wrate_value = wrate_value[::spacing]
#             ax_wrate.plot(time_traj, wrate_value, color='k', linestyle='None', marker='x', **general_style)
#             if firs_traj:
#                 ax_wrate.legend()
#                 firs_traj = False
#         ax_wrate.set_xlabel('t [s]')
#         ax_wrate.set_ylabel('Angular velocity [rad/s]')
#         ax_wrate.grid(True)

#     # Save the figure as png with tight layout and dpi=500
#     if not wrate:
#         outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories.png'
#     else:
#         outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories_wrate.png'
#     # outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories.png'
#     fig.savefig(os.path.expanduser(outname), dpi=500, bbox_inches='tight')

#     # Export the figure as tikz
#     import tikzplotlib
#     tikzplotlib_fix_ncols(fig)
#     tikzplotlib.clean_figure(fig)
#     if not wrate:
#         outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories.tex'
#     else:
#         outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories_wrate.tex'
#     # outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'training_trajectories.tex'
#     tikzplotlib.save(os.path.expanduser(outname), figure=fig)

#     plt.show()


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

    # print(xres.shape, tres.shape)
    return xres, tres, xevol_full, tevol_full

def plot_state_evol(spacing=1):
    """ Show the SDE predicted state
    """

    from sde4mbrlExamples.rotor_uav.sde_rotor_model import load_predictor_function, load_trajectory
    # path_traj = '/home/franckdjeumou/Documents/log_flights/log_13_2023-4-28-19-33-06.ulg'
    path_traj = '/home/franckdjeumou/Documents/log_flights/log_7_2023-4-28-17-55-02.ulg'
    # path_traj = '/home/franckdjeumou/Documents/PX4-Autopilot/build/px4_sitl_default/rootfs/log/2023-01-21/02_13_43.ulg'
    _traj_x, _traj_u = load_trajectory(path_traj, 
                                       outlier_cond=lambda d : d['z']>1.1,
                                        min_length=500)  # Only the first trajectory
    traj_data = {'y' : _traj_x[0], 'u' : _traj_u[0]}
    # traj_data = {'y' : _traj_x[0][:100*(_traj_x[0].shape[0]//100)], 'u' : _traj_u[0][:100*(_traj_u[0].shape[0]//100)]}
    data_stepsize = 0.01
    traj_time_evol = np.array([i*data_stepsize for i in range(traj_data['y'].shape[0])])

    sde_path = '/home/franckdjeumou/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_models/hexa_ahg_video_final_sde.pkl'
    modified_params = {'horizon' : 100, 'num_particles' : 10, 'stepsize': 0.01}
    _sys_id_model, sys_id_times = load_predictor_function(sde_path, prior_dist=True, nonoise=True, modified_params= {**modified_params, 'num_particles' : 1}, return_time_steps=True)
    sys_id_model = jax.jit(_sys_id_model)
    
    _posterior_model, post_times = load_predictor_function(sde_path, prior_dist=False, modified_params= modified_params, return_time_steps=True)
    posterior_model = jax.jit(_posterior_model)
    # Get the time evolution of the SDE
    xevol, tevol, _xfull, _tfull = n_steps_analysis(traj_data['y'], traj_data['u'], posterior_model, post_times, data_stepsize, traj_time_evol)
    # Get the time evolution of the sys_id
    xevol_sysid, tevol_sysid, _, _ = n_steps_analysis(traj_data['y'], traj_data['u'], sys_id_model, sys_id_times, data_stepsize, traj_time_evol)

    indx_traj = 0
    t_init, t_end = None , None
    # t_init, t_end = 47, 60
    t_init, t_end = 42, 60

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

    # Do the same for the groundtruth
    rpy = []
    for q in gt_data[:,6:10]:
        rpy.append(list(quat_to_euler(q, np)))
    rpy = np.array(rpy) * 180 / np.pi
    gt_data = np.concatenate([gt_data, rpy], axis=-1)

    # State to plot
    state2plot = {  0 : 'x[m]', 1 : 'y[m]', 2 : 'z[m]', 
                    3 : 'vx[m/s]', 4 : 'vy[m/s]', 5 : 'vz[m/s]', 
                    13 : 'roll[deg]', 14 : 'pitch[deg]', 15 : 'yaw[deg]',
                    10 :  'wx[rad/s]', 11 : 'wy[rad/s]', 12 : 'wz[rad/s]'
                }
    indxEq = [0, 1, 2, 3, 4, 5, 13, 14, 15, 10, 11, 12]
    

    # Create the figure and axes
    fig, _axs = plt.subplots(4, 3, constrained_layout=True, figsize=(12, 8), sharex=True)
    axs = _axs.flatten()
    for i, ax in zip(indxEq, axs):
        ax.plot(gt_time[::spacing], gt_data[::spacing,i], color='k',  label='Groundtruth' if i == 0 else None, zorder=100)
        for j in range(xevol.shape[0]):
            ax.plot(tevol[::spacing], xevol[j,::spacing,i], color='r', 
                    label = 'Ours' if j == 0 and i == 0 else None, zorder=10)
        for j in range(xevol_sysid.shape[0]):
            ax.plot(tevol_sysid[::spacing], xevol_sysid[j,::spacing,i], color='b',
                    label = 'SysID' if j == 0 and i == 0 else None, zorder=20)
        ax.set_ylabel(state2plot[i])
        ax.grid(True)
        if i >= 9:
            ax.set_xlabel('t [s]')
    
    axs[0].legend()

    # Save the figure as png with tight layout and dpi=500
    outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'state_evol.png'
    fig.savefig(os.path.expanduser(outname), dpi=500, bbox_inches='tight')

    # Export the figure as tikz
    import tikzplotlib
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.clean_figure(fig)
    outname = '~/Documents/sde4mbrl/sde4mbrlExamples/rotor_uav/hexa_ahg/my_data/' + 'state_evol.tex'
    tikzplotlib.save(os.path.expanduser(outname), figure=fig)
    
    plt.show()


    # fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, sharex=True)

    # # Plot the velocity x, y, z
    # ax_vel = axs[0]
    # ax_vel.plot(gt_time[::spacing], gt_data[::spacing,3], color='k',  label='Groundtruth', zorder=100)

    
    # # for i in range(xevol.shape[0]):
    # #     ax_vel.plot(tevol[::spacing], vel_pred[i,::spacing], color='r', 
    # #                 label = 'Predicted' if i == 0 else None, zorder=10)
        
    # ax_vel.set_ylabel('Velocity [m/s]')
    # ax_vel.set_xlabel('t [s]')
    # ax_vel.grid(True)

    # plt.show()






plot_simple_corl('fast2_circle.csv', 'log_7_2023-4-28-17-55-02.ulg',spacing=10)
plot_simple_corl('fastOld_lemn.csv', 'log_2_2023-4-28-16-10-38.ulg',spacing=10, z=1.60)


# plot_simple('fast2_circle.csv', 'log_7_2023-4-28-17-55-02.ulg',spacing=10)
# plot_simple('fast2_circle.csv', 'log_7_2023-4-28-17-55-02.ulg',spacing=10, wrate=True)
# plot_simple('fast2_circle.csv', 'log_6_2023-4-28-17-01-02.ulg',spacing=10, z=1.63)
# plot_simple('fast2_circle.csv', 'log_0_2023-4-28-15-30-52.ulg',spacing=10, z=1.65)

# plot_simple('fast2_lemn.csv', 'log_13_2023-4-28-19-33-06.ulg',spacing=10)
# plot_simple('fast2_lemn.csv', 'log_13_2023-4-28-19-33-06.ulg',spacing=10, wrate=True)
# plot_simple('fast2_lemn.csv', 'log_8_2023-4-28-18-01-18.ulg',spacing=10, z=1.22)
# plot_simple('fastOld_lemn.csv', 'log_4_2023-4-28-16-52-52.ulg',spacing=10, z=1.65)
# plot_simple('fastOld_lemn.csv', 'log_2_2023-4-28-16-10-38.ulg',spacing=10, z=1.60)
# plot_simple('fastOld_lemn.csv', 'log_2_2023-4-28-16-10-38.ulg',spacing=10, z=1.60, wrate=True)

# # # # Training trajectories
# trajs_paths = ['hexa_ahg_video1.ulg', 'hexa_ahg_video2.ulg', 'hexa_ahg_video3.ulg']
# plot_training_trajectories(trajs_paths, spacing=10)
# # # plot_training_trajectories(trajs_paths, spacing=20, wrate=True)

# # # Plot the state evolution
# plot_state_evol(spacing=10)