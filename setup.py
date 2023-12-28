from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

package_description = 'A framework for learning to control'
package_description += ' via learning uncertainty-aware stochastic differential equations representation'
package_description += ' of the system dynamics, and using model-based reinforcement learning or MPC for control.'
setup(
   name='sde4mbrl',
   version='1.0.0',
   description=package_description,
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou and Cyrus Neary',
   author_email='fdjeumou@utexas.edu and cneary@utexas.edu'
   url="https://github.com/wuwushrek/sde4mbrl.git",
   packages= find_packages(),
   # TODO: Clean up the dependencies
   install_requires=['numpy', 'scipy', 'matplotlib', 'tqdm', 'jax', 'dm-haiku', 'optax', 'mbrl', 'pandas', 'skrl', 'pynumdiff', 'pyulog', 'pyyaml'],
   tests_require=['pytest', 'pytest-cov'],
   python_requires=">=3.8"
)