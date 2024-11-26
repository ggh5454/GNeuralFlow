# GNeuralFlow
This work is based on the codebase provided in the [GNeuralFlow repository](https://github.com/gmerca/GNeuralFlow).

Giangiacomo Mercatali, Andre Freitas, Jie Chen. *Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series*. In *Advances in Neural Information Processing Systems 37, 2024*.

### Training Models
The argument `--training_scheme` specifies the type of model to be trained. Available options include:
- `lgnf`: GNeuralFlow with integrated DAG learning.
- `tgnf`: GNeuralFlow using a predefined DAG (only applicable for synthetic datasets).
- `nfe`: Neural flow from Bilos et al., 2021, used as a non-graph baseline.

### Experiments on Synthetic Datasets
To train GNeuralFlow models (`--training_scheme=lgnf`), specify `--data` as one of the following options: `sink`, `triangle`, `sawtooth`, or `square`. Below is an example for training with `sink`:

```bash
sink=(--experiment=synthetic --data=sink --training_scheme=lgnf --n_ts=20 --epochs=1000 --batch-size=50 --weight-decay=1e-05 --model=flow --flow-layers=4 --time-hidden-dim=8 --hidden-layers=2 --hidden-dim=64 --activation=ReLU --final-activation=Identity)
python -m nfe.train "${sink[@]}" --flow-model=resnet --time-net=TimeFourierBounded --rho=7 --h_tol=1e-10 --h_par=0.19 --rho_max=1e24 --dag_epochs=10
python -m nfe.train "${sink[@]}" --flow-model=coupling --time-net=TimeFourier --rho=8 --h_tol=1e-11 --h_par=0.22 --rho_max=1e27 --dag_epochs=10
python -m nfe.train "${sink[@]}" --flow-model=gru --time-net=TimeFourier --rho=8 --h_tol=1e-11 --h_par=0.22 --rho_max=1e27 --dag_epochs=10
```
