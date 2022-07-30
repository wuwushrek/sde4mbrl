import jax
import jax.numpy as jnp
import haiku as hk

from jax.experimental.host_callback import id_print

from sde_wrapper import LatentSDE, differentiable_sde_solver, create_sampling_fn, create_loss_fn

class MySDE(LatentSDE):
    def __init__(self, params={}, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name)

    def prior_drift(self, t, x, u):
        return 3*x + u
        # return jnp.zeros_like(x)

    def prior_diffusion(self, t, x):
        prior_nn = hk.get_parameter("diff", shape=(1,),
                            init=hk.initializers.RandomUniform(minval=-1e-2,maxval=1e-2))
        return jnp.array([4.]) # * jnp.exp(prior_nn)

    def posterior_drift(self, t, x, u):
        f = hk.nets.MLP(output_sizes=(4,1),
                            w_init=hk.initializers.RandomUniform(minval=-1e-1, maxval=1e-1),
                            b_init=jnp.zeros, activation=jnp.tanh, name='f')
        G = hk.nets.MLP(output_sizes=(4,1),
                            w_init=hk.initializers.RandomUniform(minval=-1e-1, maxval=1e-1),
                            b_init=jnp.zeros, activation=jnp.tanh, name='G')
        return f(x) + G(x) * u
        # return self.prior_drift(t, x, u)

class KnownSDE(LatentSDE):
    def __init__(self, params={}, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name)

    def prior_drift(self, t, x, u):
        return 3*x + u

    def prior_diffusion(self, t, x):
        return jnp.array([4.])

m_rng = jax.random.PRNGKey(0)
m_rng, o_rng = jax.random.split(m_rng)

dt = 0.01
sde_solver = {'init_step' : dt, 'max_steps': 2048}
params_solver = {'sde_solver' : sde_solver, 'n_y' : 1, 'n_u' : 1}
tlin = jnp.linspace(0., 1., 100)
xinit = jnp.array([0.5])
uinit = jnp.array([0.])

def u_opt(t, x):
    return (3*jnp.exp(8*t-8) - 7)*x / (1 + 3*jnp.exp(8*t-8))


# prior_params, sampling_fn = create_sampling_fn(xinit, uinit, params_solver, sde_constr=KnownSDE,
#                     prior_sampling=True, num_sample=100, seed=0)

# jit_sampling = jax.jit(lambda t, x, rng_val:  sampling_fn(prior_params, t, x, u_opt, rng_val))

# mEvol = jit_sampling(tlin, xinit, o_rng)

# import matplotlib.pyplot as plt
# for i in range(mEvol.shape[0]):
#     plt.plot(tlin, mEvol[i,:,0])

# plt.show()


def create_data_trajectories(params_model, num_trajectories, seed=10):
    # Create a random key generator and parse it for state initialization
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_x_key, noisy_traj_key = jax.random.split(rng_key, 3)

    def u_subopt(t, x):
        return ((3*jnp.exp(8*t-8) - 7)*x / (1 + 3*jnp.exp(8*t-8))) * jnp.exp(2*(t-1))

    # Create the model and a function for predicting trajectories
    prior_params, sampling_fn = create_sampling_fn(params_model, sde_constr=KnownSDE,
                    prior_sampling=True, seed=0)

    # Jit the sampling function for a suboptimal control
    jit_sampling = jax.jit(lambda t, x, rng_val:  sampling_fn(prior_params, t, x, u_subopt, rng_val))

    # Initial state values
    init_x = jax.random.uniform(init_x_key, (num_trajectories, xinit.shape[0]), minval=0.49, maxval=0.51)

    # Generate the different time evolution
    step_size = 0.01
    num_points = 100
    dur_traj = int(num_points * step_size)
    init_rng, traj_length_rng = jax.random.split(noisy_traj_key)
    # t0_val = jax.random.uniform(init_rng, (num_trajectories,), minval=0., maxval=0.2)
    t0_val = jnp.zeros((num_trajectories))
    tevol = jax.vmap(lambda t : jnp.linspace(t, t+dur_traj, num_points))(t0_val)

    rng_key = jax.random.split(rng_key, num_trajectories)
    state_data = jax.vmap(jit_sampling)(tevol, init_x, rng_key)
    state_data = state_data.reshape((-1, tevol.shape[1], init_x.shape[1]))
    uval = u_subopt(tevol.reshape((*tevol.shape, 1)), state_data)
    return {'y' : state_data, 't' : tevol, 'u' : uval}

dt = 0.01
sde_solver = {'init_step' : dt, 'max_steps': 4096}
params_model = {'sde_solver' : sde_solver, 'n_y' : 1, 'n_u' : 1, 'fixed_ts' : False,
                'num_particles' : 1}
dataset = create_data_trajectories(params_model, 1000, seed=10)
# dataset = create_data_trajectories(params_model, 10, seed=5)

# bs = 10
# m_rng = jax.random.split(m_rng, bs)
# print(loss_fn(loss_params, dataset['y'][:bs, :5], dataset['u'][:bs, :5], m_rng, dataset['t'][:bs, :5]))



import yaml
from train_sde_helper import train_model

# Open the yaml file containing the configuration to train the model
yml_file = open('config_linreg.yaml')
yml_byte = yml_file.read()
m_config = yaml.load(yml_byte, yaml.SafeLoader)
yml_file.close()

print(m_config)

no_improv = m_config['training']['no_improvement_bound']
improved_est = lambda curr_opt, test_opt, train_opt: \
                    curr_opt['Total Loss'] > test_opt['Total Loss'] + no_improv

train_model(m_config, dataset, 'outfile', improved_est, MySDE)

from sde_wrapper import load_model_from_file
num_particles = 100
_prior_fn, _posterior_fn, extra = load_model_from_file('outfile.pkl', MySDE, 0, num_particles)
jit_prior_fn, jit_posterior_fn = jax.jit(_prior_fn), jax.jit(_posterior_fn)
# jit_prior_fn = jax.jit(lambda t,y,r : _prior_fn(t,y,u_opt,r))
# jit_posterior_fn = jax.jit(lambda t,y,r : _posterior_fn(t,y,u_opt,r))

tevol, y0, uevol = dataset['t'][1,:], dataset['y'][1,0,:], dataset['u'][1,:,:]

ys_prior = jit_prior_fn(tevol, y0, uevol, m_rng)
ys_posterior = jit_posterior_fn(tevol, y0, uevol, m_rng)
# ys_prior = jit_prior_fn(tevol, y0, m_rng)
# ys_posterior = jit_posterior_fn(tevol, y0, m_rng)

sde_solver = {'init_step' : 0.01, 'max_steps': 4096}
params_model = {'sde_solver' : sde_solver, 'n_y' : 1, 'n_u' : 1, 'fixed_ts' : False,
                'num_particles' : num_particles}
prior_params, sampling_fn = create_sampling_fn(params_model, sde_constr=KnownSDE,
                    prior_sampling=True, seed=0)
ys_true = jax.jit(sampling_fn)(prior_params, tevol, y0, uevol, o_rng)
# ys_true = jax.jit(lambda p,t,y,r : sampling_fn(p,t,y,u_opt,r))(prior_params, tevol, y0, m_rng)
print(extra)


import matplotlib.pyplot as plt
# for i in range(200):
#     plt.plot(dataset['t'][i,:20], dataset['y'][i,:20,0], color='black')
for i in range(num_particles):
    plt.plot(tevol[:80], ys_true[i,:80,0], color='red')
    # plt.plot(tevol[:20], ys_prior[i,:20,0], color='blue')
    plt.plot(tevol[:80], ys_posterior[i,:80,0], color='green')

plt.show()

# class MySDE(LatentSDE):
#     def __init__(self, params={}, name=None):
#         # Define the params here if needed before initialization
#         super().__init__(params, name)

#     def init_posterior_drift(self):
#         self.f = hk.nets.MLP(output_sizes=(2,2,1),
#                             w_init=hk.initializers.RandomUniform(minval=-1e-2, maxval=1e-2),
#                             b_init=jnp.zeros, activation=jnp.tanh, name='f')
#         self.G = hk.nets.MLP(output_sizes=(2,2,1),
#                             w_init=hk.initializers.RandomUniform(minval=-1e-2, maxval=1e-2),
#                             b_init=jnp.zeros, activation=jnp.tanh, name='G')
#         self.posterior_drift = lambda t, x, u : self.f(x) + self.G(x) * u

#     def init_prior_drift(self):
#         self.prior_drift = lambda t, x, u : jnp.zeros_like(x)

#     def init_prior_diffusion(self):
#         self.prior_nn = hk.get_parameter("diff", shape=(1,),
#                             init=hk.initializers.RandomUniform(minval=-1e-2,maxval=1e-2))
#         self.prior_diffusion = lambda t, x: 10. * t * jnp.exp(self.prior_nn)

#     def init_encoder(self):
#         self.obs2state = self.transform_obs2state
#         self.logprob = lambda obs, s: 0. # -jnp.sum(jnp.square())

#     def transform_obs2state(self, obs, rng):
#         return obs

# m_rng = jax.random.PRNGKey(0)
# m_rng, o_rng = jax.random.split(m_rng)
# def prior_sample(*x):
#     return KnownSDE().sample_prior(*x)

# def u_opt(t, x):
#     return (3*jnp.exp(8*t-8) - 7)*x / (1 + 3*jnp.exp(8*t-8))

# # Transform the function into a pure one
# estimate_pure =  hk.without_apply_rng(hk.transform(prior_sample))
# nn_params = estimate_pure.init(m_rng, jnp.array([0.1, 1]), jnp.array([0.5]), jnp.array([0.]), m_rng)
# print(nn_params)

# dt = 0.001
# params_solver = {'init_step' : dt, 'max_steps': 1100}
# tlin = jnp.linspace(0., 1., 1000)
# xinit = jnp.array([0.5])


# vmap_estimate = jax.vmap(lambda t, obs, rng : estimate_pure.apply(nn_params, t, obs, u_opt, rng, params_solver),
#                     in_axes=(None,None,0))

# jit_estimate = jax.jit(lambda *x : vmap_estimate(*x))
# o_rng = jax.random.split(o_rng, 100)
# mEvol = jit_estimate(tlin, xinit, o_rng)

# import matplotlib.pyplot as plt
# for i in range(mEvol.shape[0]):
#     plt.plot(tlin, mEvol[i,:,0])

# plt.show()
