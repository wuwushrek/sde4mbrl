import sys, os
import matplotlib.pyplot as plt
import yaml
import numpy as np
import tikzplotlib

data_file = os.path.abspath(os.path.join('..', 'my_data', 'barplot_data.yaml'))

with open(data_file, 'r') as f:
    data = yaml.safe_load(f)
    
# Define some constants for plotting.
learned_env_width = 0.04
true_env_width = 0.08
spacing = 0.1

small_gap = 0.1
big_gap = 0.2

learned_env_alpha = 0.3
true_env_alpha = 0.8

reward_translation_offset = 500

fig = plt.figure()
ax = fig.add_subplot(111)

mean_vals = []
std_vals = []
positions = []
widths = []
colors = []

# Begin by appending the mean and std of the PPO agent,
# trained and evaluated on the groundtruth environment.
mean_vals.append(data['groundtruth_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['groundtruth_PPO']['std_best_reward'])
widths.append(true_env_width)
positions.append(0)

mean_vals.append(data['cartpole_bb_random_sde_PPO']['best_mean_reward_gt'] + reward_translation_offset)
mean_vals.append(data['cartpole_bb_random_sde_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['cartpole_bb_random_sde_PPO']['std_best_reward_gt'])
std_vals.append(data['cartpole_bb_random_sde_PPO']['std_best_reward'])
widths.append(true_env_width)
widths.append(learned_env_width)
positions.append(positions[-1] + big_gap)
positions.append(positions[-1] + learned_env_width / 2 + true_env_width / 2)

# Next append the mean and std of the PPO agent,
# trained and evaluated on the Gaussian MLP random environment.
mean_vals.append(data['gaussian_mlp_ensemble_cartpole_random_PPO']['best_mean_reward_gt'] + reward_translation_offset)
mean_vals.append(data['gaussian_mlp_ensemble_cartpole_random_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['gaussian_mlp_ensemble_cartpole_random_PPO']['std_best_reward_gt'])
std_vals.append(data['gaussian_mlp_ensemble_cartpole_random_PPO']['std_best_reward'])
widths.append(true_env_width)
widths.append(learned_env_width)
positions.append(positions[-1] + small_gap)
positions.append(positions[-1] + learned_env_width / 2 + true_env_width / 2)

mean_vals.append(data['cartpole_bb_learned_si_sde_PPO']['best_mean_reward_gt'] + reward_translation_offset)
mean_vals.append(data['cartpole_bb_learned_si_sde_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['cartpole_bb_learned_si_sde_PPO']['std_best_reward_gt'])
std_vals.append(data['cartpole_bb_learned_si_sde_PPO']['std_best_reward'])
widths.append(true_env_width)
widths.append(learned_env_width)
positions.append(positions[-1] + big_gap)
positions.append(positions[-1] + learned_env_width / 2 + true_env_width / 2)

mean_vals.append(data['cartpole_bb_learned_sde_PPO']['best_mean_reward_gt'] + reward_translation_offset)
mean_vals.append(data['cartpole_bb_learned_sde_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['cartpole_bb_learned_sde_PPO']['std_best_reward_gt'])
std_vals.append(data['cartpole_bb_learned_sde_PPO']['std_best_reward'])
widths.append(true_env_width)
widths.append(learned_env_width)
positions.append(positions[-1] + small_gap)
positions.append(positions[-1] + learned_env_width / 2 + true_env_width / 2)

mean_vals.append(data['gaussian_mlp_ensemble_cartpole_learned_PPO']['best_mean_reward_gt'] + reward_translation_offset)
mean_vals.append(data['gaussian_mlp_ensemble_cartpole_learned_PPO']['best_mean_reward'] + reward_translation_offset)
std_vals.append(data['gaussian_mlp_ensemble_cartpole_learned_PPO']['std_best_reward_gt'])
std_vals.append(data['gaussian_mlp_ensemble_cartpole_learned_PPO']['std_best_reward'])
widths.append(true_env_width)
widths.append(learned_env_width)
positions.append(positions[-1] + small_gap)
positions.append(positions[-1] + learned_env_width / 2 + true_env_width / 2)


ax.bar(positions, mean_vals, widths, yerr=std_vals, align='center', alpha=0.5, ecolor='black', capsize=10)

# bottoms_md = np.zeros((len(data.keys()),))
# bottoms_base = np.zeros((len(data.keys()),))

# ind = np.arange(3)

# num_eps_trials = len(results[list(results.keys())[0]]['md'])

# print(num_eps_trials)

# for eps_ind in range(num_eps_trials):

#     heights_md = []
#     heights_base = []
#     for trial in ['0001', '0022', '2233']:
#         heights_md.append(results[trial]['md'][eps_ind])
#         heights_base.append(results[trial]['base'][eps_ind])

#     print(heights_md)

#     ax.bar(ind + eps_ind * (1.2 * width), heights_md, width)
#     ax.bar(ind + eps_ind * (1.2 * width) + num_eps_trials * (1.2 * width) + width/2, heights_base, width)

# # add some labels
# ax.set_ylabel('Probability of Success')
# ax.set_xticks(np.array([0.36, 1.35, 2.35]))
# ax.set_xticklabels(['0001', '0022', '2233'])

tikzplotlib.save('tikz/bar_chart.tex')

# plt.show()