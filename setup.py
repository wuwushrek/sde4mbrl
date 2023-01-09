from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

package_description = 'A framework for learning to control'
package_description += ' via learning (offline) stochastic differential equations representation'
package_description += ' of the dynamics, and the cost-to-go/value function of the underlying'
package_description += ' task through Iterative solving of the generalized (and linear PDE) HJB.'
package_description += ' These two quantities are used online for solving optimal control problem'
package_description += ' using an accelerated gradient descent MPC formulation with near-optimal initialization of the variables'
setup(
   name='sde4mbrl',
   version='1.0.0',
   description=package_description,
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou',
   author_email='fdjeumou@utexas.edu',
   url="https://github.com/wuwushrek/sde4mbrl.git",
   packages= find_packages(),
   install_requires=['numpy', 'scipy', 'matplotlib', 'tqdm', 'jax', 'dm-haiku', 'optax'], 
   tests_require=['pytest', 'pytest-cov'],
   python_requires=">=3.6"
)
# 'jax>=0.2.28', 'jaxlib', 'dm-haiku', 'optax'
# python_requires=">=3.7"
# 'jupyterlab', 'ipympl'
# 'jax>=0.2.28', 'jaxlib', 'dm-haiku', 'PolyakSGD', 'ArmijoSGD', 'pandas'