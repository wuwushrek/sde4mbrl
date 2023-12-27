# Cartpole System Example

Below are the instructions to reproduce the results in the paper related to the Cartpole example. These are the results shown in Figure 6, and figures 10, 11, 12, 13, and 14 in the paper.

## Train the different  models for the Cartpole example

The dataset used for the Cartpole example were generated via the `generate_data.py` script. The data can be found in the folder `my_data/` under the names `learned.pkl` and `random.pkl`. The `learned.pkl` dataset was generated from a sub-optimal PPO policy trained on the ground truth dynamics. The `random.pkl` dataset was generated from a random policy.

This step can be skipped as we already provide the trained models in `my_models`. You can also stop the training of the models at any time if you observe that the test loss is not changing anymore (THough, early stopping is already implemented).

1. Train the neural SDE model by running the following command:

```bash
# SDE model on the learned.pkl dataset and without any side information
# GO INSIDE THE FOLDER `cartpole_sde.yaml` and changes the following:
# data_dir: learned.pkl`
# side_info: False
python cartpole_sde.py --fun train_sde --cfg cartpole_sde.yaml --out cartpole_bb_learned

# SDE model on the learned.pkl dataset and with side information
# GO INSIDE THE FOLDER `cartpole_sde.yaml` and changes the following:
# data_dir: learned.pkl`
# side_info: True
python cartpole_sde.py --fun train_sde --cfg cartpole_sde.yaml --out cartpole_bb_learned_si

# SDE model on the random.pkl dataset and without any side information
# GO INSIDE THE FOLDER `cartpole_sde.yaml` and changes the following:
# data_dir: random.pkl`
# side_info: False
python cartpole_sde.py --fun train_sde --cfg cartpole_sde.yaml --out cartpole_bb_random
```

2. Train the neural ODE model by running the following command:

```bash
# ODE model on the learned.pkl dataset and without any side information
# GO INSIDE THE FOLDER `ode_cartpole.yaml` and changes the following:
# data_dir: learned.pkl`
# side_info: False
python cartpole_sde.py --fun train_sde --cfg ode_cartpole.yaml --out cartpole_bb_learned_ode

# ODE model on the learned.pkl dataset and with side information
# GO INSIDE THE FOLDER `ode_cartpole.yaml` and changes the following:
# data_dir: learned.pkl`
# side_info: True
python cartpole_sde.py --fun train_sde --cfg ode_cartpole.yaml --out cartpole_bb_learned_ode_si

# ODE model on the random.pkl dataset and without any side information
# GO INSIDE THE FOLDER `ode_cartpole.yaml` and changes the following:
# data_dir: random.pkl`
# side_info: False
python cartpole_sde.py --fun train_sde --cfg ode_cartpole.yaml --out cartpole_bb_rand_ode
```

3. Train the Gaussian Ensemble models by running the following command:

```bash
# Gaussian Ensemble model on the learned.pkl dataset
python train_gaussian_mlp_cartpole.py --data learned.pkl

# Gaussian Ensemble model on the random.pkl dataset
python train_gaussian_mlp_cartpole.py --data random.pkl
```

## Figure 10: Epistemic Uncertainty Visualization

```bash
# GO INSIDE THE FOLDER `config_plot_dad_term.yaml` and uncomment all lines from
# `-----> To uncomment for Figure 10 experiments` to 
# `----> END Figure 10 in the paper`
python plot_dad_term.py --fun unc
```

## Figure 11: Comparison Neural SDE and Probabilistic Ensemble

```bash
# GO INSIDE THE FOLDER `config_plot_dad_term.yaml` and uncomment all lines from
# `-----> To uncomment for Figure 11 experiments` to
# `----> END Figure 11 in the paper`
python plot_dad_term.py --fun pred
```

## Figures 12, 13, and 14: Comparison Neural SDE and Neural ODE

```bash
# GO INSIDE THE FOLDER `config_plot_dad_term.yaml` and uncomment all lines from
# `-----> To uncomment for Figure 12, 13, and 14 experiments` to
# `----> END Figure 12, 13, and 14 experiments`

# Figure 12. Uncomment specifically the section about [Figure 12] in the config file
python plot_dad_term.py --fun pred

# Figure 13. Uncomment specifically the section about [Figure 13] in the config file
python plot_dad_term.py --fun pred

# Figure 14. Uncomment specifically the section about [Figure 14] in the config file
python plot_dad_term.py --fun pred

```

## Figure 6: RL Comparison

1. Learn the policies corresponding to the different models. The configuration file for everything below is `rl_config.yaml`.

```bash
# The model files are the models that has been trained in the previous section
# cartpole_bb_learned_ode_sde, cartpole_bb_learned_ode_si, cartpole_bb_learned, cartpole_bb_learned_si, cartpole_bb_random, cartpole_bb_rand_ode, gaussian_mlp_ensemble_cartpole_random, gaussian_mlp_ensemble_cartpole_learned
python mfrl_with_learned_dynamics.py --fun train --model_files groundtruth gaussian_mlp_ensemble_cartpole_learned cartpole_bb_learned # and so on if needed. 
# Groundtruth is the model free policy on the GT env. All this will take a LOT of time for all the models and all seeds.
```

2. Evaluate the different trained policies and store the results in a pickle file.

```bash
# GO INSIDE THE FOLDER `rl_config.yaml` and [find the key model_names] comment all the models 
# that haven't been trained and  and uncomment the models that have been trained.
# `outfile` key in the configuration file is the name of the pickle file that will be generated.`
python mfrl_with_learned_dynamics.py  --fun evalpol
```

3. Return the dictionary of data to plot the bar chart in Figure 6.

```bash
python mfrl_with_learned_dynamics.py  --fun barplot --outbar barplot_data.yaml # The file will be saved in my_data/barplot_data.yaml
```

4. Plot the reward evolution as a function of training steps.

```bash
# GO INSIDE THE FOLDER `rl_config.yaml` and find # -------> Different version to plots, change it accordingly,
# uncomment/modify the policies that you want to plot.
# Change fig_args and most importantly plot_configs accordingly
python mfrl_with_learned_dynamics.py  --fun plot
```