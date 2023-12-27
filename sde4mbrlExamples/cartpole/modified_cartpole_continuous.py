""" Ground truth dynamics model for the cartpole environment. 
"""

import math
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled

class CartPoleEnv(gym.Env):
    # This is a continuous version of gym's cartpole environment, with the only difference
    # being valid actions are any numbers in the range [-1, 1], and the are applied as
    # a multiplicative factor to the total force.
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": [50]}

    def __init__(
                self, 
                max_steps=200, 
                init_lb: list = [-1.0, -1.0, np.pi - 0.8, -0.8],
                init_ub: list = [1.0, 1.0, np.pi + 0.8, 0.8],
                measurement_noise_diag: list = [0.005, 0.01, 0.009, 0.05],
                render_mode: Optional[str] = None
                ):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 25 # 50.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.max_steps = max_steps

        self.init_lb = init_lb
        self.init_ub = init_ub
        self.measurement_noise_diag = measurement_noise_diag

        # Position at which to fail the episode
        self.x_threshold = 120.0 # 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                # self.theta_threshold_radians * 2,
                1.0,
                1.0,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        act_high = np.array((1,), dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.state = None

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step_dynamics(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)


    def step(self, action):   
        self.elapsed_steps += 1     
        action = action.squeeze()
        
        self.step_dynamics(action)

        x, x_dot, theta, theta_dot = self.state
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or self.elapsed_steps >= self.max_steps
        )

        if not terminated:
            pass
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1

        reward = np.cos(theta) \
                - 0.01 * np.abs(x) \
                - np.sum(np.abs(action)) \
                - 0.1 * np.abs(theta_dot) \
                    - 0.1 * np.abs(x_dot)

        if self.render_mode == "human":
            self.render()

        return self.get_obs(self.state), reward, terminated, False, {}

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        # x = 0.0
        # x_dot = self.np_random.uniform(low=-0.05, high=0.05)
        # theta = np.pi
        # theta_dot = self.np_random.uniform(low=-0.05, high=0.05)
        # self.state = (x, x_dot, theta, theta_dot)
        self.state = np.random.uniform(self.init_lb, self.init_ub)
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        return self.get_obs(self.state), {}

    def get_obs(self, state):
        x, x_dot, theta, theta_dot = state + \
            np.random.multivariate_normal(np.zeros((4)), np.diag(self.measurement_noise_diag))
        return np.array((x, x_dot, np.sin(theta), np.cos(theta), theta_dot))

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame  # type: ignore
            from pygame import gfxdraw  # type: ignore
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = 30.0 * scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(30)#self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame  # type: ignore

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

if __name__ == "__main__":
    env = CartPoleEnv(render_mode="human")
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        env.render()
    env.close()
