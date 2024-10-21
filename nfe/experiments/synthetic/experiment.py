import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from nfe.experiments import BaseExperiment
from nfe.experiments.synthetic.data import get_data_loaders, get_single_loader
from nfe.experiments.synthetic.utils import plot_trajectory_ax
from nfe.models import ODEModel, CouplingFlow, ResNetFlow
from nfe.models.gnn import GNN
from nfe.models.mlp import MLP




class Synthetic(BaseExperiment):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        # self.net = MLP(self.dim * 2 + 1, hidden_dims, dim, activation, final_activation, wrapper_func=None)
        features = {'sink': 2, 'ellipse': 2, 'activity': 3}
        self.nfeats = features[args.data] if args.data in list(features.keys()) else 1
        # self.gcn = GNN(input_size=self.nfeats, hidden_size=32).to(self.device)

        self.learned_model_path = f'{self.exp_path}{self.noise}_model.pt'
        self.learned_graph_path = f'{self.exp_path}{self.noise}_dag.pt'

        gcn_layers = []
        for _ in range(args.gnn_layers):
            gcn_layers.append(GNN(input_size=self.nfeats, hidden_size=32))
        self.gcn = torch.nn.ModuleList(gcn_layers).to(self.device)

        self.emb = MLP(in_dim=self.nfeats,
                       hidden_dims=[args.hidden_dim] * args.hidden_layers,
                       out_dim=self.nfeats,
                       activation='ReLU',
                       final_activation='Tanh',
                       wrapper_func=None).to(self.device)

    def get_model(self, args):
        if args.model == 'ode':
            return ODEModel(self.dim, args.odenet, [args.hidden_dim] * args.hidden_layers, args.activation,
                            args.final_activation, args.solver, args.solver_step, args.atol, args.rtol)
        elif args.model == 'flow':
            if args.flow_model == 'coupling':
                return CouplingFlow(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                    args.time_net, args.time_hidden_dim)
            elif args.flow_model == 'gru':
                return GRUFlow(self.dim, args.flow_layers, args.time_net, args.time_hidden_dim)
            elif args.flow_model == 'resnet':
                return ResNetFlow(self.dim, args.flow_layers, [args.hidden_dim] * args.hidden_layers,
                                  args.time_net, args.time_hidden_dim, data=args.data, gnn_layers=args.gnn_layers,
                                  seed=args.seed)
        raise NotImplementedError

    def get_data(self, args):
        _args = {'n_ts': args.n_ts, 'training_scheme': args.training_scheme, 'dag_data': args.dag_data, 'data': args.data, 'seed':args.seed}
        return get_data_loaders(args.data, _args, args.batch_size)

    def _get_loss(self, batch, adj=None, return_sol=False):
        x, t, y_true = batch
        h = None
        if adj is not None:
            h = x.view(x.shape[0], 1, x.shape[-1] // self.nfeats, self.nfeats)
            for gnn in self.gcn:
                h = gnn(h, adj)
            h = h.view(x.shape)
        y = self.model(x, h, t)
        loss = torch.mean((y - y_true) ** 2)
        if not return_sol:
            return loss
        else:
            return y, loss

    def _get_loss2(self, batch, adj=None):
        x, t, y_true = batch
        y = self.model(x, t, adj)
        loss = torch.mean((y - y_true)**2)
        return loss

    def _get_loss_on_dl(self, dl, adj=None):
        losses = []
        for batch in dl:
            losses.append(self._get_loss(batch, adj).item())
        return np.mean(losses)

    def training_step(self, batch, adj=None):
        return self._get_loss(batch, adj)

    def validation_step(self, adj=None):
        return {'mse': self._get_loss_on_dl(self.dlval, adj)}

    def test_step(self, adj=None):
        return {'mse': self._get_loss_on_dl(self.dltest, adj)}

    def _plot_trajectory(self, name, y, loss, n, t, noise):
        plt.figure()
        # case of 2d feature
        if self.args.data in ['sink', 'ellipse']:
            for i, sol in enumerate(y):
                for ix, j in zip(range(1, n + 1), range(0, n * 2, 2)):
                    plt.plot(sol[:, j], sol[:, j + 1])
        # case of 1d feature
        else:
            for sol in y:
                for i in range(n):
                    plt.plot(t.detach().cpu().numpy().squeeze(), sol[:, i][:None])

        plt.title(f"{name}   {noise}   {loss}")
        plt.savefig(f'{self.exp_path}_traj_{noise}{name}.png')


    def sample_noise_multiple(self, noises):
        """Produce graph plot and trajectory for multiple noise levels with fixed n_ts and dataset_name"""
        f, ax = plt.subplots(nrows=2, ncols=len(noises)+1, figsize=(5*(len(noises)+1), 5*2))
        figs_path = 'nfe/output/figs/'
        os.makedirs(f'{figs_path}', exist_ok=True)
        for i, noise in enumerate(noises):
            noise = f'_n{noise:.0e}'
            self.learned_model_path = f'{self.exp_path}{noise}_model.pt'
            self.learned_graph_path = f'{self.exp_path}{noise}_dag.pt'
            self.sample_noise_single(noise, ax, i + 1)
        f.savefig(f'{figs_path}{self.args.data}_multiple.png')

    def sample_noise_single(self, noise, ax, ix):
        """sample from trained model"""
        n_traj = 1
        batch = self.dltrain.dataset[:n_traj]
        _, t, y_true = batch
        n = self.args.n_ts

        self.model.load_state_dict(torch.load(self.learned_model_path, map_location=torch.device('cpu')))
        a_learned = torch.load(self.learned_graph_path, map_location=torch.device('cpu')).to(self.device)
        a_true = torch.tensor(self.true_dag, dtype=torch.float32).to(self.device)

        y_l, loss_l = self._get_loss(batch, a_learned, return_sol=True)
        y_l = y_l.detach().cpu().numpy()
        plot_trajectory_ax(self.args.data, 'Learned-gnf', ax[1, ix], y_l, loss_l, n, t, noise)
        # torch.save(y_l, f'traj_{noise}2.pt')

        ax[0, ix].imshow(a_learned.detach().cpu(), cmap='viridis')
        ax[0, ix].set_title(f"Learned DAG {self.args.data} {noise}")

        # column 0 row 0
        ax[0, 0].imshow(a_true.detach().cpu(), cmap='viridis')
        ax[0, 0].set_title(f"True DAG {self.args.data}")

        # column 0 raw 1
        y_true = y_true.detach().cpu().numpy()
        plot_trajectory_ax(self.args.data, 'Data', ax[1, 0], y_true, 0, n, t, '')
        # torch.save(y_true, f'tra_data2.pt')

        # torch.save(a_learned.detach().cpu(), f'dag_{noise}.pt')

    def sample_trajectories(self):
        """sample from trained model"""
        n_traj = 1
        batch = self.dltrain.dataset[:n_traj]
        _, t, y_true = batch
        n = self.args.n_ts
        noise = self.noise

        self.model.load_state_dict(torch.load(self.learned_model_path, map_location=torch.device('cpu')))
        a_learned = torch.load(self.learned_graph_path, map_location=torch.device('cpu')).to(self.device)
        a_true = torch.tensor(self.true_dag, dtype=torch.float32).to(self.device)

        y_l, loss_l = self._get_loss(batch, a_learned, return_sol=True)
        y_l = y_l.detach().cpu().numpy()
        self._plot_trajectory('Learned-gnf', y_l, f'{loss_l:.1e}', n, t, noise)



    def finish(self):

        torch.save(self.model.state_dict(), self.learned_model_path)
        if self.learned_dag is not None:
            torch.save(obj=self.learned_dag, f=self.learned_graph_path)
            # np.savetxt(f'{self.learned_graph_path}.csv', self.learned_dag.cpu().numpy(), delimiter=',')
