from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

package_description = 'A learning framework for learning to control'
package_description += ' via learning stochastic differential equations and'
package_description += ' solving optimal control problem through FBSDEs'
package_description += ' in an end-to-end fashion'
setup(
   name='rl_girsanov',
   version='1.0.0',
   description=package_description,
   license="GNU 3.0",
   long_description=long_description,
   author='Franck Djeumou',
   author_email='fdjeumou@utexas.edu',
   url="https://github.com/wuwushrek/rl_girsanov.git",
   packages=['rl_girsanov'],
   package_dir={'rl_girsanov': 'rl_girsanov/'},
   install_requires=['numpy', 'scipy', 'matplotlib', 'tqdm',
                     'jax>=0.3.4', 'jaxlib', 'dm-haiku', 'optax'],
   tests_require=['pytest', 'pytest-cov'],
   python_requires=">=3.7"
)
