# sde4mbrl
A framework for learning to control via (offline) learning of stochastic differential equation's representations of the dynamics, and the cost-to-go/value function of the underlying task through Iterative solving of the generalized (and linear PDE) HJB. These two quantities are then used online for solving stochastic optimal control problem using an accelerated gradient descent MPC formulation with near-optimal initialization of the variables.

## Installation

By installing the package as below, you also install dependencies such as Jax, dm-haiku, optax, diffrax, etc...
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
