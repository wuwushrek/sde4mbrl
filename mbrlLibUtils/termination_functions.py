import torch

def cartpole_swingup(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 120.0
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
    )
    done = ~not_done
    done = done[:, None]
    return done