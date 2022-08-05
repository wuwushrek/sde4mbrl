import jax
import jax.numpy as jnp
import haiku as hk

from jax.experimental.host_callback import id_print

from sde_wrapper import LatentSDE, create_cost_sampling_fn, create_sampling_fn

class KnownSDE(LatentSDE):
    def __init__(self, params={}, name=None):
        # Define the params here if needed before initialization
        super().__init__(params, name)

    def prior_drift(self, t, x, u):
        return 3*x + u

    posterior_drift = prior_drift

    def prior_diffusion(self, t, x):
        return jnp.array([4.])

def cost_fn(x, u):
    return jnp.sum(3.5 * jnp.square(x) + 0.5 * jnp.square(u))

def terminal_fn(x):
    return 0.5 * jnp.sum(jnp.square(x))

import yaml
# Open the yaml file containing the configuration to train the model
yml_file = open('config_linreg.yaml')
yml_byte = yml_file.read()
m_config = yaml.load(yml_byte, yaml.SafeLoader)
yml_file.close()

print(m_config)

nn_params, cos_eval, vmapped_prox, constr_cost, construct_opt_params = \
    create_cost_sampling_fn(m_config['model'], cost_fn, terminal_cost=terminal_fn,
            sde_constr= KnownSDE, seed=0)

print(nn_params)
print(vmapped_prox)
print(constr_cost)

rng = jax.random.PRNGKey(0)

ts = jnp.linspace(0., 1., 100)
# param_free_cost = lambda opt_params, y, rng: cos_eval(nn_params, ts, y, opt_params, rng)

# Initial point
y0 = jnp.array([jnp.sqrt(2.)])

# Initial control
u0 = jnp.ones((ts.shape[0], 1))
opt_init = construct_opt_params(y0, u0)

composed_cost = jax.jit(lambda u_params: cos_eval(nn_params, ts, y0, u_params, rng)[0])
import time

m_t = time.time()
opt_state = composed_cost(opt_init)
n_t = time.time() - m_t
print('Loss initialization time : ', n_t)

m_t = time.time()
opt_state = composed_cost(opt_init)
n_t = time.time() - m_t
print('Loss initialization time : ', n_t)

m_t = time.time()
opt_state = composed_cost(opt_init)
n_t = time.time() - m_t
print('Loss initialization time : ', n_t)


from online_sde_control import apg, init_apg
import time
from jax.tree_util import tree_flatten

jit_init_apg = jax.jit(lambda x0 : init_apg(x0, composed_cost, m_config['apg_mpc']))
opt_state = jit_init_apg(opt_init)
m_t = time.time()
opt_state = jit_init_apg(opt_init)
n_t = time.time() - m_t
print('initialization time : ', n_t)
m_t = time.time()
opt_state = jit_init_apg(opt_init)
n_t = time.time() - m_t
print('initialization time : ', n_t)
# print(opt_state)

jit_grad_step = jax.jit(lambda _opt_state : apg(_opt_state, composed_cost, m_config['apg_mpc'], proximal_fn=vmapped_prox))

num_iter = 100

apg_res =  dict(opt_state._asdict())
apg_res = {k : [float(v)] for k, v in apg_res.items() if k not in ('xk', 'yk', 'grad_yk')}
apg_res['elapsed'] = [0]
apg_res['time'] = [0]
print({ k : '{:.2e}'.format(v[0]) for k, v in apg_res.items()})
elapsed_time = 0
for i in range(num_iter):
    past_time = time.time()
    opt_state = jit_grad_step(opt_state)
    opt_state.xk.block_until_ready()
    # tree_flatten(opt_state)[0][0].block_until_ready()
    curr_time = time.time() - past_time
    if i > 1:
        elapsed_time += curr_time
    m_state = dict(opt_state._asdict())
    m_state = {k : float(v) for k, v in m_state.items() if k not in ('xk', 'yk', 'grad_yk')}
    m_state['elapsed'] = elapsed_time
    m_state['time'] = curr_time
    for _k, v in m_state.items():
        apg_res[_k].append(v)
    if m_state['not_done'] < 1:
        print(opt_state.yk)
        break
    print({ k : '{:.2e}'.format(v) for k, v in m_state.items()})
opt_u = opt_state.yk
