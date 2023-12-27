# Cartpole System Example

Below are the instructions to reproduce the results in the paper related to the Cartpole example. These are the results shown in Figure 6, and figures 10, 11, 12, 13, and 14 in the paper.

## Train the different  models for the Cartpole example

The dataset used for the Cartpole example were generated via the `generate_data.py` script. The data can be found in the folder `my_data/` under the names `learned.pkl` and `random.pkl`. The `learned.pkl` dataset was generated from a sub-optimal PPO policy trained on the ground truth dynamics. The `random.pkl` dataset was generated from a random policy.

This step can be skipped as we already provide the trained models in `my_models`.

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
python cartpole_ode.py --fun train_ode --cfg ode_cartpole.yaml --out cartpole_bb_learned_ode

# ODE model on the learned.pkl dataset and with side information
# GO INSIDE THE FOLDER `ode_cartpole.yaml` and changes the following:
# data_dir: learned.pkl`
# side_info: True
python cartpole_ode.py --fun train_ode --cfg ode_cartpole.yaml --out cartpole_bb_learned_ode_si

# ODE model on the random.pkl dataset and without any side information
# GO INSIDE THE FOLDER `ode_cartpole.yaml` and changes the following:
# data_dir: random.pkl`
# side_info: False
python cartpole_ode.py --fun train_ode --cfg ode_cartpole.yaml --out cartpole_bb_rand_ode
```

3. Train the Gaussian Ensemble models by running the following command:

```bash
# Gaussian Ensemble model on the learned.pkl dataset
python train_gaussian_mlp_cartpole.py --data learned.pkl

# Gaussian Ensemble model on the random.pkl dataset
python train_gaussian_mlp_cartpole.py --data random.pkl
```
