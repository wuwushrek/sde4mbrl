import omegaconf
import torch

device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ensemble_cfg = omegaconf.OmegaConf.create({
    # dynamics model configuration
    "obs_shape" : (5,),
    "action_shape" : (1,),
    "trainer_setup" : {
        "optim_lr" : 0.001,
        "weight_decay" : 5e-5,
        "num_epochs" : 5000,
        "patience" : 2000,
        "batch_size" : 32,
        "model_checkpoint_frequency" : 100,
    },
    "dynamics_model": {
        "_target_": "mbrl.models.GaussianMLP",
        "device": device_str,
        "num_layers": 3,
        "ensemble_size": 5,
        "hid_size": 64,
        "in_size": 6,
        "out_size": 5,
        "deterministic": False,
        # "propagation_method": "fixed_model",
        "activation_fn_cfg": {
            "_target_": "torch.nn.SiLU",
            # "negative_slope": 0.01
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": True,
    },
    "overrides": {
        "validation_ratio": 0.05
    }
})