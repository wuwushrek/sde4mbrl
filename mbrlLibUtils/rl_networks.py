import torch
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin

class Policy(GaussianMixin, Model):
    
    def __init__(self, observation_space, action_space, device,
                clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = torch.nn.Sequential(torch.nn.Linear(self.num_observations, 64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 32),
                                torch.nn.ReLU(),
                                torch.nn.Linear(32, self.num_actions),
                                torch.nn.Tanh())

        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))
        
    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
    
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(torch.nn.Linear(self.num_observations, 64),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(64, 32),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}