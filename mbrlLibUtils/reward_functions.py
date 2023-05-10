import torch

def cartpole_swingup(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    
    # x = next_obs[:, 0]
    # cos_theta = next_obs[:, 2]
    reward = torch.cos(next_obs[:, 3]) - 0.1 * torch.abs(next_obs[:, 0])

    return (reward).float().view(-1, 1)