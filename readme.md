# GNeuralFlow
This repository includes the supporting code for

Giangiacomo Mercatali, Andre Freitas, Jie Chen. Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series. In Advances in Neural Information Processing Systems 37, 2024.


## Installation
Install required packages from `env.yml`. The code was tested with python `3.10` and pytorch `2.1.1`.

## Training models
The argument `--training_scheme` indicates what model to train. Choices are:
  - `lgnf`: GNeuralFlow together with DAG learning.
  - `tgnf`: GNeuralFlow with a given DAG (only for synthetic data).
  - `nfe`: Neural flow from Bilos et al. 2021 (serving as a non-graph baseline).


## Experiments of synthetic datasets
Use the following commands to train GNeuralFlow models (`--training_scheme=lgnf`). `--data` is one of `sink`, `triangle`, `sawtooth`, `square`. The following example is for sink.

```bash
sink=(--experiment=synthetic --data=sink --training_scheme=lgnf --n_ts=20 --epochs=1000 --batch-size=50 --weight-decay=1e-05 --model=flow --flow-layers=4 --time-hidden-dim=8 --hidden-layers=2 --hidden-dim=64 --activation=ReLU --final-activation=Identity)
python -m nfe.train "${sink[@]}" --flow-model=resnet --time-net=TimeFourierBounded --rho=7 --h_tol=1e-10 --h_par=0.19 --rho_max=1e24 --dag_epochs=10
python -m nfe.train "${sink[@]}" --flow-model=coupling --time-net=TimeFourier --rho=8 --h_tol=1e-11 --h_par=0.22 --rho_max=1e27 --dag_epochs=10
python -m nfe.train "${sink[@]}" --flow-model=gru --time-net=TimeFourier --rho=8 --h_tol=1e-11 --h_par=0.22 --rho_max=1e27 --dag_epochs=10
```
(안됨)
## Experiments of latent variable modeling: Smoothing approach

### Activity
```bash
activity=(--experiment=latent_ode --data activity --training_scheme=lgnf --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --hidden-layers 3 --hidden-dim 100 --rec-dims 30 --latents 20 --gru-units 100 --model flow --flow-layers 2 --time-hidden-dim 8)
python -m nfe.train "${activity[@]}" --flow-model gru --time-net TimeLinear --rho=15  --h_par=0.21
python -m nfe.train "${activity[@]}" --flow-model coupling --time-net TimeLinear --rho=7  --h_par=0.21
python -m nfe.train "${activity[@]}" --flow-model resnet --time-net TimeTanh --rho=7 --h_tol=1e-9  --h_par=0.5
```
(안됨)
### MuJoCo 
```bash
mujoco=(--experiment=latent_ode --data hopper --training_scheme=lgnf --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 0 --hidden-layers 2 --hidden-dim 100 --rec-dims 100 --gru-units 50 --model flow --flow-layers 2 --time-hidden-dim 8)
python -m nfe.train "${mujoco[@]}" --flow-model gru --time-net TimeLinear --latents 20
python -m nfe.train "${mujoco[@]}" --flow-model coupling --time-net TimeLinear --latents 20
python -m nfe.train "${mujoco[@]}" --flow-model resnet --time-net TimeTanh --latents 20
```
(안됨)

### Physionet
```bash
physionet=(--experiment=latent_ode --data physionet --training_scheme=lgnf --dag_epochs=100 --batch-size 100 --epochs 1000 --lr-decay 0.5 --lr-scheduler-step 20 --weight-decay 0.0001 --extrap 0 --classify 1 --hidden-layers 3 --hidden-dim 50 --rec-dims 40 --latents 20 --gru-units 50 --model flow --flow-layers 2 --time-hidden-dim 8)
python -m nfe.train "${physionet[@]}" --flow-model gru --time-net TimeLinear --h_par=0.5 --rho=15
python -m nfe.train "${physionet[@]}" --flow-model coupling --time-net TimeLinear --h_par=0.5 --rho=10
python -m nfe.train "${physionet[@]}" --flow-model resnet --time-net TimeTanh --rho=15
```

## Experiments of latent variable modeling: Filtering approach
### Prepare data (MIMIC IV)
1) Download csv datasets. Need to register at `physionet.org`. Place the data in the folder `nfe/experiments/data/physionet.org/files/mimiciv/2.2`.
2) Run the scripts in the [prep_data](nfe/experiments/gru_ode_bayes/prep_data) folder, in this order: 1. `admission`, 2. `prescription`, 3. `labevents`, 4. `inputsevents`, 5. `outputs`.

### Train GNeuralFlow
```bash
mimic4=(--experiment=gru_ode_bayes --data=mimic4 --training_scheme=lgnf --patience=5 --dag_epochs=1000 --epochs=1000 --batch-size=100 --weight-decay=0.0001 --lr-decay=0.33 --lr-scheduler-step=20 --model=flow --flow-layers=4 --time-net=TimeTanh --time-hidden-dim=8 --hidden-layers=2 --hidden-dim=64 --activation=ReLU --final-activation=Identity)
python -m nfe.train "${mimic4[@]}" --flow-model=gru --rho_par=10 --h_tol=1e-10 --h_par=0.15 --rho_max=1e24 --dag_epochs=1
python -m nfe.train "${mimic4[@]}" --flow-model=coupling --rho_par=10 --h_tol=1e-10 --h_par=0.15 --rho_max=1e24 --dag_epochs=1
python -m nfe.train "${mimic4[@]}" --flow-model=resnet --rho_par=10 --h_tol=1e-10 --h_par=0.15 --rho_max=1e24 --dag_epochs=1
```

## Acknowledgements
Our code was developed based on the following repositories.
- https://github.com/mbilos/neural-flows-experiments
- https://github.com/EnyanDai/GANF

