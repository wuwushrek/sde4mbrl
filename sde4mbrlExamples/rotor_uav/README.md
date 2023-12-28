# Quadcopter and HexaCopter Experiments

This folder contains the code for the experiments on the quadcopter and hexacopter. These are the results shown in Figures 8, 16 - 22.

## Warm-up With Gazebo SITL Simulation for Iris Quadcopter

1. Generate the right dataset format from .ulg files from Gazebo SITL.

We collected three trajectories `iris_gazebo_traj1.ulg`, `iris_gazebo_traj2.ulg`, and `iris_gazebo_traj3.ulg` from Gazebo SITL by manually flying iris around. We saved it and uploaded it in `iris_sitl/my_data/iris_gazebo/`. Use the following command to convert the .ulg files to the right format for training the SDE/ODE/SysID models.

```bash
python sde_rotor_model.py --trajs --cfg iris_sitl/data_generation.yaml
# The output model will be in iris_sitl/my_data/trajs.pkl (depending on the `outfile` argument in the yaml file)
```

2. Train the SysID/SDE/ODE model on the dataset generated from Gazebo SITL.

```bash
# First let's train the SysID model. This will be used as a warm-up for the SDE/ODE models.
python train_static_model.py --cfg iris_sitl/optimizer_prior.yaml --out prior_iris
# The output model will be in iris_sitl/my_models/prior_iris.yaml

# Now let's train the SDE model. Make sure the learned_nominal in `optimizer_sde.yaml` is set to the prior model trained above.
python sde_rotor_model.py --train --cfg iris_sitl/optimizer_sde.yaml --out iris_sitl

# Now let's train the ODE model. Set `noise_prior_params` to zero in the `optimizer_sde.yaml` file. 
# Besides, comment the key `diffusion_density_nn` and all its subkeys to not train the diffusion density network.
python sde_rotor_model.py --train --cfg iris_sitl/optimizer_sde.yaml --out iris_sitl_ode
```

3. Evaluate and plot the different models' prediction accuracy

```bash
# Modify the file accordingly to the models you want to evaluate. This was not the one used for the paper.
python python display_learned_model.py
```


4. Open Loop Control Simulation Without Gazebo SITL

By default, the command below will learn a simulation of the SDE trying to track a reference setpoint from a random initial condition. Check the `iris_sitl/mpc_test_cfg.yaml` and `iris_sitl/test_mpc.py` for configuration details and how to run the simulation.

```bash
python test_mpc.py
```

5. Closed Loop Control Simulation With Gazebo SITL

For SITL, we need a modification of [PX4 Firmware](https://github.com/wuwushrek/PX4-Autopilot/tree/mpc_franck) to run the Simulator and interface it with Gazebo and [the MPC Node Controller](https://github.com/wuwushrek/sde4mbrl_px4) that runs the MPC based on the specified learned model while exchanging optimal control actions with the PX4 Firmware.

- Set up first the [PX4 Firmware](https://github.com/wuwushrek/PX4-Autopilot/tree/mpc_franck)
- Set up the [MPC Node Controller](https://github.com/wuwushrek/sde4mbrl_px4) and follow the guide to select a specific model and MPC hyperparameters.
- Plotting and reproducing the paper results can be done through `iris_sitl/compare_perf.py`


## Hexacopter Experiments

The steps are the same as for the quadcopter. We set-up Gazebo Hexacopter simulation with details provided in [PX4 Firmware](https://github.com/wuwushrek/PX4-Autopilot/tree/mpc_franck). We also provide the dataset used in our experiments as well as the models used on the hexacopter in real-world simulation.

`plot_figures.py` is the main script used for the plotting data from the real-world hexacopter experiments.