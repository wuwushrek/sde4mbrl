from modified_cartpole_continuous import CartPoleEnv

import numpy as np

def sample_gym_trajectory(env, policy, max_steps=200):
    obs, done = env.reset()
    

def main():
    enf = CartPoleEnv(render_mode='rgb_array')

if __name__ == "__main__":
    env = CartPoleEnv(render_mode='rgb_array')
    env.reset(0)
    for i in range(100):
        env.render()
        env.step(np.random.uniform(-1, 1))
    env.close()