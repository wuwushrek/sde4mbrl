# sde4mbrl

| Title      | Physics-Constrained and Uncertainty-Aware Neural Stochastic Differential Equations                 |
|------------|----------------------------------------------------------------------------------------------|
| Authors    | Franck Djeumou*, Cyrus Neary*, and Ufuk Topcu                                                |
| Conference | Conference on Robot Learning (CoRL), 2023                                                            |

We present a framework and algorithms to learn controlled dynamics models using neural stochastic differential equations (SDEs)—SDEs whose drift and diffusion terms are both parametrized by neural networks. We construct the drift term to leverage a priori physics knowledge as inductive bias, and we design the diffusion term to represent a distance-aware estimate of the uncertainty in the learned model’s predictions. The proposed neural SDEs can be evaluated quickly enough for use in model predictive control algorithms, or they can be used as simulators for model-based reinforcement learning. Furthermore, they make accurate predictions over long
time horizons, even when trained on small datasets that cover limited regions of the state space. We demonstrate these capabilities through experiments on simulated robotic systems, as well as by using them to model and control a hexacopter’s flight dynamics: A neural SDE trained using only three minutes of manually collected flight data results in a model-based control policy that accurately tracks fast trajectories that push the hexacopter’s velocity and Euler angles to nearly double the maximum values observed in the training dataset.

## Installation

By installing the package as below, you also install dependencies such as Jax, dm-haiku, optax, skrl, etc...
```
python -m pip install -e .
```

If JAX needs to run on the GPU, please follow the instructions on [Jax website](https://github.com/google/jax)
```
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Examples and Reproducing the Paper Results

1. For the Mass Spring Damper example, please refer to the [README.md](sde4mbrlExamples/mass_spring_damper/README.md) in the `sde4mbrlExamples/mass_spring_damper/` folder.


2. For the Cartpole example, please refer to the [README.md](sde4mbrlExamples/cartpole/README.md) in the `sde4mbrlExamples/cartpole/` folder.

3. For the Quadcopter and HexaCopter experiments, please refer to the [README.md](sde4mbrlExamples/rotor_uav/README.md) in the `sde4mbrlExamples/rotor_uav/` folder.


# Read the Paper for Important Details

Most of the conceptual details for this repository are primarily described in the final version of the paper. Here, we provide the extended version on Arxiv of the conference paper.
```
@misc{djeumou2023learn,
      title={How to Learn and Generalize From Three Minutes of Data: Physics-Constrained and Uncertainty-Aware Neural Stochastic Differential Equations}, 
      author={Franck Djeumou and Cyrus Neary and Ufuk Topcu},
      year={2023},
      eprint={2306.06335},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Contact

Please contact Franck Djeumou (fdjeumou@utexas.edu) or Cyrus Neary (cneary@utexas.edu) for questions regarding this code.