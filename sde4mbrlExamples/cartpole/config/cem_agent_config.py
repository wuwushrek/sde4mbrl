import omegaconf
import torch

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

agent_cfg = omegaconf.OmegaConf.create({
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 15,
    "replan_freq": 5,
    "verbose": False,
    "action_lb": [-1.0],
    "action_ub": [1.0],
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 50,
        "elite_ratio": 0.1,
        "population_size": 50,
        "alpha": 0.1,
        "device": device,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True,
        # "clipped_normal": False
    }
})