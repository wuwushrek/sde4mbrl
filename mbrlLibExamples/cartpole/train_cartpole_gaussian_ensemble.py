from mbrl.env.cartpole_continuous import CartPoleEnv
from mbrl.util.replay_buffer import ReplayBuffer
import mbrl.planning as planning
import numpy as np
import os

import mbrl.util.common as common_util

from mbrlLibExamples.cartpole.config.cem_agent_config import agent_cfg
from mbrlLibUtils.sgd_model_trainer import ProgressBarCallback, SGDModelTrainer
from mbrlLibUtils.save_and_load_models import save_model_and_config, load_model_and_config

from mbrl.env.cartpole_continuous import CartPoleEnv

import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)

# Initialize environment.
env = CartPoleEnv()

# Generate training data.
num_trajectories = 100
max_steps_per_trajectory = 1000

buffer_size = num_trajectories * max_steps_per_trajectory
replay_buffer = ReplayBuffer(buffer_size, env.observation_space.shape, env.action_space.shape, rng=rng)

common_util.rollout_agent_trajectories(
    env,
    buffer_size, # initial exploration steps
    planning.RandomAgent(env),
    {}, # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=max_steps_per_trajectory
)

print("# samples stored", replay_buffer.num_stored)

# Train the dynamics model.
from config.gaussian_mlp_ensemble_cartpole_config import ensemble_cfg
dynamics_model = common_util.create_one_dim_tr_model(
                        ensemble_cfg, 
                        env.observation_space.shape, 
                        env.action_space.shape
                    )

# Normalizer gets called automatically in dynamics_model._process_batch()
dynamics_model.update_normalizer(replay_buffer.get_all()) # update normalizer stats

train_dataset, val_dataset = common_util.get_basic_buffer_iterators(
                        replay_buffer, 
                        ensemble_cfg['trainer_setup']['batch_size'], 
                        ensemble_cfg['overrides']['validation_ratio'], 
                        ensemble_size=ensemble_cfg['dynamics_model']['model']['ensemble_size'], 
                        shuffle_each_epoch=True
                    )

trainer = SGDModelTrainer(
                dynamics_model,
                optim_lr=ensemble_cfg['trainer_setup']['optim_lr'],
                weight_decay=ensemble_cfg['trainer_setup']['weight_decay']
            )

pbar = ProgressBarCallback(ensemble_cfg['trainer_setup']['num_epochs'])

train_losses, val_losses = trainer.train(
                                train_dataset, 
                                val_dataset, 
                                num_epochs=ensemble_cfg['trainer_setup']['num_epochs'], 
                                patience=ensemble_cfg['trainer_setup']['patience'], 
                                callback=pbar.progress_bar_callback,
                                num_steps_per_epoch=ensemble_cfg['trainer_setup']['num_steps_per_epoch'],
                            )

# Save the learned model
experiment_name = 'gaussian_mlp_ensemble_cartpole'
save_folder = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))
save_model_and_config(dynamics_model, ensemble_cfg, save_folder)

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
ax[0].plot(train_losses)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train loss (gaussian nll)")
ax[1].plot(val_losses)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("val loss (mse)")
plt.show()