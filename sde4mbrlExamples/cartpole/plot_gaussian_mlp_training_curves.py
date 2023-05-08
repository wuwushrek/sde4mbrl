import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append('../..')
from mbrlLibUtils.save_and_load_models import save_model_and_config, load_model_and_config
    
from mbrlLibUtils.replay_buffer_utils import generate_sample_trajectories
import pickle

# Load the model
experiment_name = 'gaussian_mlp_ensemble_cartpole_random'
load_dir = os.path.abspath(os.path.join(os.path.curdir, 'my_models', experiment_name))

with open(os.path.abspath(os.path.join(load_dir, 'training_results.pkl')), 'rb') as f:
    training_results = pickle.load(f)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(training_results['epoch'], training_results['training_loss'], label='training')
# ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')

val_scores = [x.cpu().numpy().mean() for x in training_results['validation_score']]

ax = fig.add_subplot(1,2,2)
ax.plot(training_results['epoch'], val_scores, label='validation')
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')

plt.show()

