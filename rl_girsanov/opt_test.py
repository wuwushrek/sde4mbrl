import yaml

import jax
import jax.numpy as jnp

from jax.tree_util import tree_flatten
import time

# Load optimizer params
yml_file = open('test_opt.yaml')
yml_byte = yml_file.read()
opt_params = yaml.load(yml_byte, yaml.SafeLoader)
yml_file.close()

print(opt_params)

rng = jax.random.PRNGKey(0)
rng, rng_y, rng_q, rng_x0 = jax.random.split(rng, 4)
n_row, n_col = 30, 20
sdp = False

# Generate two random matrices
x0_val = jax.random.uniform(rng_x0, (n_col, ), minval=-2, maxval=2)
y = jax.random.uniform(rng_y, (n_row,), minval=-2, maxval=2)
q_sqrt = jax.random.uniform(rng_q, (n_row, n_col), minval=-4, maxval=4.)

num_iter = 100

if sdp:
    Q = q_sqrt * jnp.transpose(q_sqrt)
else:
    Q = q_sqrt

def loss_fn(opt_var):
    return 0.5 * jnp.sum(jnp.square(y - Q @ opt_var))

from online_sde_control import apg, init_apg

jit_init_apg = jax.jit(lambda x0 : init_apg(x0, loss_fn, opt_params))
opt_state = jit_init_apg(x0_val)
print(opt_state)

jit_grad_step = jax.jit(lambda _opt_state : apg(_opt_state, loss_fn, opt_params, proximal_fn=None))

apg_res =  dict(opt_state._asdict())
apg_res = {k : [float(v)] for k, v in apg_res.items() if k not in ('xk', 'yk', 'grad_yk')}
print({ k : '{:.2e}'.format(v[0]) for k, v in apg_res.items()})
for _ in range(num_iter):
    curr_time = time.time()
    opt_state = jit_grad_step(opt_state)
    tree_flatten(opt_state)[0][0].block_until_ready()
    elapsed_time = time.time() - curr_time
    m_state = dict(opt_state._asdict())
    m_state = {k : float(v) for k, v in m_state.items() if k not in ('xk', 'yk', 'grad_yk')}
    for _k, v in m_state.items():
        apg_res[_k].append(v)
    if m_state['not_done'] < 1:
        break
    print({ k : '{:.2e}'.format(v) for k, v in m_state.items()})


# Check how adam does on this problem
import optax

@jax.jit
def step(params, opt_state):
    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

optimizer = optax.adam(learning_rate=0.5)
opt_state = optimizer.init(x0_val)
list_loss = []
for _ in range(num_iter):
    x0_val, opt_state, loss_value = step(x0_val, opt_state)
    tree_flatten(opt_state)[0][0].block_until_ready()
    list_loss.append(float(loss_value))
    print(float(loss_value))
