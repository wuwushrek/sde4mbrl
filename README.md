# rl-girsanov
A reinforcement learning algorithm via forward-backward stochastic differential equations and Girsanov theorem

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