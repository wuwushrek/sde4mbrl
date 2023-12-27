# Mass Spring Damper Example

Below are the instructions to reproduce the results in the paper related to the Mass Spring Damper example. These are the results shown in Figure 3, Figure 4, and Figure 5 in the paper.

## Figure 3 Experiment

1. Generate the data for the experiment by running the following command:

```bash
cd sde4mbrlExamples/mass_spring_damper
python density_script.py --fun gen_data
```
This will generate files `my_data/MSD_LowNoise_DensityExp_X_config`, where X is 3, 6, and 25 corresponding to the number of trajectories used to generate the data. Check the configuration file `config_density_dataset.yaml` for details on the dataset generation.

2. Train the density model for the dataset with 6 and 25 trajectories by doing as follows:

```bash
# 1. In config_density_model.yaml, modify trajId to be DensityExp_6, then train the model
python density_script.py --fun train
# 2. In config_density_model.yaml, modify trajId to be DensityExp_25, then train the model
python density_script.py --fun train
```

3. Generate the plots for the experiment by running the following command:

```bash
python density_script.py --fun plot
```

## Figure 4 Experiment

1. Generate the data for the experiment by running the following command:

```bash
cd sde4mbrlExamples/mass_spring_damper
python mass_spring_model.py --fun gen_traj
```
This will generate a set of files in `my_data` with different number of trajectories, level of noise, and where the initial conditions of the trajectories are sampled from. Check the configuration file `data_generation.yaml` for details on the dataset generation.

2. Train the different models used for comparison in the papers. This was done for the dataset `MSD_MeasHigh_TopRight_5.pkl` with 5 trajectories and high noise.

```bash
# 1. Train the neural SDE model
python mass_spring_model.py --fun train --model_type nesde_bboxes --data MSD_MeasHigh_TopRight_5
# 2. Train the neural ODE Model by setting the `noise_prior_params` to 0 in the config file `mass_spring_damper.yaml` and commenting the key `diffusion_density_nn` with all its subkeys.
python mass_spring_model.py --fun train --model_type node_bboxes --data MSD_MeasHigh_TopRight_5
# 3. Train the Gausian Ensemble Models if needed
python train_gaussian_mlp_ensemble_msd.py --data MSD_MeasHigh_TopRight_5
```

3. Generate the plots for the experiment by running the following command:

```bash
python pred_analysis_script.py --fun unc
# Check the config_pred_analysis.yaml for the different options
```

## Figure 5 and Figure 9 Experiments

Use the same dataset as for Figure 4.

1. Open the file `config_pred_analysis.yaml` and uncomment everything after the line `-----> To uncomment for Figure 5 or Figure 9 experiments`.

2. Generate the plots for the experiment by running the following command:

```bash
python pred_analysis_script.py --fun pred
# Check the config_pred_analysis.yaml for the different options
```

