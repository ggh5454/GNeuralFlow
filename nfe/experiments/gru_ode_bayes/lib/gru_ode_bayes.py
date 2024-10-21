import torch
import math
import numpy as np
import torch.nn as nn

from itertools import compress

from nfe.models import CouplingFlow, ResNetFlow, ODEModel, GRUFlow
from nfe.models.gnn import *
from nfe.models.mlp import *

class ModelH(nn.Module):
    def __init__(self, hidden_dim, p_hidden, n_sensors):
        super(ModelH, self).__init__()
        hdim = 32
        self.gcn = GNN(input_size=1, hidden_size=hdim)
        self.net1 = nn.Sequential(
            nn.Linear(hidden_dim, hdim),
            # nn.BatchNorm1d(hdim),
            # nn.LayerNorm(hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, n_sensors),
            # nn.BatchNorm1d(n_sensors),
            # nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(n_sensors, hdim),
            # nn.BatchNorm1d(hdim),
            # nn.LayerNorm(hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )
        self.graph_emb = nn.Sequential(
            nn.Linear(n_sensors, n_sensors),
            nn.BatchNorm1d(n_sensors),
            nn.ReLU()
        )

    def inlcude_gcn(self, h, adj):
        h = self.net1(h)
        adj = self.graph_emb(adj)
        h = self.gcn(h.unsqueeze(1).unsqueeze(-1), adj).squeeze(1).squeeze(-1)
        h = self.net2(h)
        return h

    def forward(self, h, adj):
        h1 = self.inlcude_gcn(h, adj)
        return h1


class NNFOwithBayesianJumps(nn.Module):
    def __init__(self, input_size, args, p_hidden, prep_hidden, cov_size, cov_hidden, mixing):
        super().__init__()

        self.p_model = nn.Sequential(
            nn.Linear(args.hidden_dim, p_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(p_hidden, 2 * input_size, bias=True),
        )

        self.model_h = ModelH(args.hidden_dim, p_hidden, args.n_sensors)

        if args.model == 'ode':
            if args.odenet == 'gru':
                gru = FullGRUODECell_Autonomous(args.hidden_dim, bias=True)
                self.odeint = ODEModel(args.hidden_dim, gru, None, None, None, args.solver, args.solver_step, args.atol, args.rtol)
            else:
                self.odeint = ODEModel(args.hidden_dim, args.odenet, [args.hidden_dim] * args.hidden_layers, args.activation,
                                       args.final_activation, args.solver, args.solver_step, args.atol, args.rtol)

        elif args.model == 'flow':
            if args.flow_model == 'coupling':
                flow = CouplingFlow
                mlp_ = 'MlpLN'
            elif args.flow_model == 'resnet':
                flow = ResNetFlow
                mlp_ = 'MLP'
                # mlp_ = 'MlpBN2'
            elif args.flow_model == 'gru':
                flow = GRUFlow
                mlp_ = 'MLP'
            else:
                raise NotImplementedError
            self.odeint = flow(dim=args.hidden_dim,
                               n_layers=args.flow_layers,
                               hidden_dims=[args.hidden_dim] * args.hidden_layers,
                               time_net=args.time_net,
                               mlp=mlp_,
                               time_hidden_dim=args.time_hidden_dim,
                               invertible=bool(args.invertible))

        self.input_size = input_size
        self.mixing = mixing
        self.gru_obs = GRUObservationCellLogvar(input_size, args.hidden_dim, prep_hidden)

        self.covariates_map = nn.Sequential(
            nn.Linear(cov_size, cov_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(cov_hidden, args.hidden_dim, bias=True),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, times, num_obs, X, M, delta_t, cov, val_times,
                     return_path=False, smoother=False, class_criterion=None, labels=None, adj=None):

        h = self.covariates_map(cov)
        p = self.p_model(h)
        num_obs = num_obs.to(h.device)
        current_time = 0.0
        counter, loss_1, loss_2 = 0, 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            idx = num_obs > ind
            idx = idx.cpu()
            current_times = torch.Tensor([x[ind] for x in times[idx]])
            diff = current_times - last_times[idx]

            h2 = self.model_h(h[idx], adj).unsqueeze(1)
            solution = self.odeint(h[idx].unsqueeze(1), h2, diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[idx] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[idx])

            zero_tens = torch.Tensor([0]).to(h.device)
            X_slice = X[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            M_slice = M[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]

            h, losses = self.gru_obs(h, p, X_slice, M_slice, idx)

            real_NLL.append(losses.mean())

            assert not  torch.isnan(losses).any()
            assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[idx])

            loss_2 = loss_2 + compute_KL_loss(p_obs=p, X_obs=X_slice, M_obs=M_slice, logvar=True)

            last_times[idx] = current_times

        if val_times is not None:
            val_numobs = torch.Tensor([len(x) for x in val_times])

            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                current_times = torch.Tensor([x[ind] for x in list(compress(val_times, idx.cpu()))])
                diff = current_times - last_times[idx]

                h2 = self.model_h(h[idx], adj).unsqueeze(1)
                solution = self.odeint(h[idx].unsqueeze(1), h2, diff.view(-1, 1, 1).to(h.device))

                temp = h.clone()
                temp[idx] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[idx])

                if return_path:
                    path_p.append(p)

                last_times[idx] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2


    def forward_(self, times, num_obs, X, M, delta_t, cov, val_times,
                 return_path=False, smoother=False, class_criterion=None, labels=None, adj=None):

        h = self.covariates_map(cov)
        p = self.p_model(h)
        num_obs = num_obs.to(h.device)
        current_time = 0.0
        counter, loss_1, loss_2 = 0, 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            idx = num_obs > ind
            idx = idx.cpu()
            current_times = torch.Tensor([x[ind] for x in times[idx]])
            diff = current_times - last_times[idx]

            solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[idx] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[idx])

            zero_tens = torch.Tensor([0]).to(h.device)
            X_slice = X[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            M_slice = M[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]

            h, losses = self.gru_obs(h, p, X_slice, M_slice, idx)
            real_NLL.append(losses.mean())

            assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[idx])

            loss_2 = loss_2 + compute_KL_loss(p_obs=p, X_obs=X_slice, M_obs=M_slice, logvar=True)

            last_times[idx] = current_times

        if val_times is not None:
            val_numobs = torch.Tensor([len(x) for x in val_times])

            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                current_times = torch.Tensor([x[ind] for x in list(compress(val_times, idx.cpu()))])
                diff = current_times - last_times[idx]

                solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))

                temp = h.clone()
                temp[idx] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[idx])

                if return_path:
                    path_p.append(p)

                last_times[idx] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2

    def forward_2odes(self, times, num_obs, X, M, delta_t, cov, val_times,
                      return_path=False, smoother=False, class_criterion=None, labels=None, adj=None):

        h = self.covariates_map(cov)
        p = self.p_model(h)
        h2 = self.covariates_map2(cov)
        self.p_model2(h2)

        num_obs = num_obs.to(h.device)
        current_time = 0.0
        counter, loss_1, loss_2 = 0, 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            idx = num_obs > ind
            idx = idx.cpu()
            current_times = torch.Tensor([x[ind] for x in times[idx]])
            diff = current_times - last_times[idx]

            # 1st ODE solver
            solution2 = self.odeint2(x=h[idx].unsqueeze(1), h=h2[idx].unsqueeze(1), t=diff.view(-1, 1, 1).to(h.device))
            temp2 = h2.clone()
            temp2[idx] = solution2.squeeze(1)
            h2 = temp2.clone()
            p2 = self.p_model2(h2[idx])

            zero_tens = torch.Tensor([0]).to(h.device)
            X_slice = X[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            M_slice = M[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            H_slice = self.gcn(X_slice.unsqueeze(1).unsqueeze(3), adj).view(X_slice.shape[0], -1)
            h2, _ = self.gru_obs2(h2, p2, H_slice, M_slice, idx)

            # 2nd ODE solver
            solution = self.odeint(x=h[idx].unsqueeze(1), h=h2[idx].unsqueeze(1), t=diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[idx] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[idx])

            h, losses = self.gru_obs(h, p, X_slice, M_slice, idx)
            real_NLL.append(losses.mean())
            assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[idx])
            loss_2 = loss_2 + compute_KL_loss(p_obs=p, X_obs=X_slice, M_obs=M_slice, logvar=True)
            last_times[idx] = current_times

        if val_times is not None:
            val_numobs = torch.Tensor([len(x) for x in val_times])

            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                idx = idx.cpu()
                current_times = torch.Tensor([x[ind] for x in list(compress(val_times, idx))])
                diff = current_times - last_times[idx]

                solution2 = self.odeint2(h[idx].unsqueeze(1), h2[idx].unsqueeze(1), diff.view(-1, 1, 1).to(h.device))
                temp2 = h2.clone()
                temp2[idx] = solution2.squeeze(1).clone()
                h2 = temp2.clone()

                solution = self.odeint(h[idx].unsqueeze(1), h2[idx].unsqueeze(1), diff.view(-1, 1, 1).to(h.device))
                temp = h.clone()
                temp[idx] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[idx])

                if return_path:
                    path_p.append(p)

                last_times[idx] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2

    def forward_h(self, times, num_obs, X, M, delta_t, cov, val_times,
                  return_path=False, smoother=False, class_criterion=None, labels=None, adj=None):

        h = self.covariates_map(cov)
        p = self.p_model(h)
        num_obs = num_obs.to(h.device)
        counter, loss_1, loss_2 = 0, 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            idx = num_obs > ind
            idx = idx.cpu()
            current_times = torch.Tensor([x[ind] for x in times[idx]])
            diff = current_times - last_times[idx]

            solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[idx] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[idx])

            zero_tens = torch.Tensor([0]).to(h.device)
            X_slice = X[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            M_slice = M[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]

            h, losses = self.gru_obs_h(h, p, X_slice, M_slice, idx, adj)

            real_NLL.append(losses.mean())
            # assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[idx])
            loss_2 = loss_2 + compute_KL_loss(p_obs=p, X_obs=X_slice, M_obs=M_slice, logvar=True)
            last_times[idx] = current_times

        if val_times is not None:
            val_numobs = torch.Tensor([len(x) for x in val_times])

            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                current_times = torch.Tensor([x[ind] for x in list(compress(val_times, idx.cpu()))])
                diff = current_times - last_times[idx]

                solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))

                temp = h.clone()
                temp[idx] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[idx])

                if return_path:
                    path_p.append(p)

                last_times[idx] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2

    def forward_orig(self, times, num_obs, X, M, delta_t, cov, val_times,
                     return_path=False, smoother=False, class_criterion=None, labels=None, adj=None):

        h = self.covariates_map(cov)
        p = self.p_model(h)
        num_obs = num_obs.to(h.device)
        current_time = 0.0
        counter, loss_1, loss_2 = 0, 0, 0
        real_NLL = []
        path_p = []

        last_times = torch.zeros((len(num_obs)))
        for ind in range(0, int(torch.max(num_obs).item())):
            idx = num_obs > ind
            idx = idx.cpu()
            current_times = torch.Tensor([x[ind] for x in times[idx]])
            diff = current_times - last_times[idx]

            solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))
            temp = h.clone()
            temp[idx] = solution.squeeze(1)
            h = temp.clone()
            p = self.p_model(h[idx])

            zero_tens = torch.Tensor([0]).to(h.device)
            X_slice = X[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]
            M_slice = M[(torch.cat((zero_tens, torch.cumsum(num_obs, dim=0)))[:-1][idx] + ind).long()]

            h, losses = self.gru_obs(h, p, X_slice, M_slice, idx)
            real_NLL.append(losses.mean())

            # assert not losses.sum() != losses.sum()

            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h[idx])

            loss_2 = loss_2 + compute_KL_loss(p_obs=p, X_obs=X_slice, M_obs=M_slice, logvar=True)

            last_times[idx] = current_times

        if val_times is not None:
            val_numobs = torch.Tensor([len(x) for x in val_times])

            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                current_times = torch.Tensor([x[ind] for x in list(compress(val_times, idx.cpu()))])
                diff = current_times - last_times[idx]

                solution = self.odeint(h[idx].unsqueeze(1), None, diff.view(-1, 1, 1).to(h.device))

                temp = h.clone()
                temp[idx] = solution.squeeze(1).clone()
                h = temp.clone()
                p = self.p_model(h[idx])

                if return_path:
                    path_p.append(p)

                last_times[idx] = current_times

        loss = loss_1 + self.mixing * loss_2

        if return_path:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2, torch.cat(path_p, dim=0)
        else:
            return h, loss, torch.mean(torch.Tensor(real_NLL)), loss_1, loss_2


class FullGRUODECell_Autonomous(nn.Module):

    def __init__(self, hidden_size, bias=True):
        super().__init__()

        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, t, inp):
        h, diff = inp[0], inp[1]
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        # change of variables
        dh = ( (1 - z) * (u - h) ) * diff
        return dh, torch.zeros_like(diff)

class GRUObservationCellLogvar2(nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))
        self.gcn1 = GNN(input_size=1, hidden_size=32)
        self.gcn2 = GNN(input_size=1, hidden_size=32)
        self.gcn3 = GNN(input_size=prep_hidden, hidden_size=32)

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p_obs, X_obs, M_obs, i_obs, adj):
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        log_lik_c = np.log(np.sqrt(2*np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        assert not losses.sum() != losses.sum()

        X_obs = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        X_obs = torch.matmul(X_obs, self.w_prep).squeeze(2) + self.bias_prep
        X_obs.relu_()
        X_obs = self.gcn3(X_obs.unsqueeze(1), adj).squeeze(1)

        X_obs = X_obs.permute(2, 0, 1)
        X_obs = (X_obs * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp = self.gru_d(X_obs, h[i_obs].clone())
        h[i_obs] = temp.clone()

        return h, losses

class GRUObservationCellLogvarH(nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True, **kwargs):
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.fc1 = nn.Linear(prep_hidden, 1)
        self.fc2 = nn.Linear(1, prep_hidden)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden

        self.gcn = GNN(input_size=prep_hidden, hidden_size=32)

    def forward(self, h, p_obs, X_obs, M_obs, i_obs, adj):
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        log_lik_c = np.log(np.sqrt(2*np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        assert not losses.sum()!=losses.sum()

        X_obs = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        X_obs = torch.matmul(X_obs, self.w_prep).squeeze(2) + self.bias_prep
        X_obs.relu_()

        X_obs = X_obs.permute(2, 0, 1)
        X_obs = (X_obs * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = self.gru_d(X_obs, h[i_obs].clone())
        h[i_obs] = temp.clone()

        return h, losses

class GRUObservationCellLogvar3(nn.Module):
    """Merge Adj via stack and mat mul"""
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True, **kwargs):
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_d2 = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gcn = Gnn(input_size=1, hidden_size=32, **kwargs)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.w_prep2 = nn.Parameter(std * torch.randn(input_size, 5, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.merge = MLP(hidden_size*2, [32, 32], hidden_size, 'ReLU', 'SiLU')

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p_obs, X_obs, M_obs, i_obs, **kwargs):
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        log_lik_c = np.log(np.sqrt(2*np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        assert not losses.sum() != losses.sum()

        X_obs_orig = X_obs.clone()
        X_obs = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        X_obs = torch.matmul(X_obs, self.w_prep).squeeze(2) + self.bias_prep
        X_obs.relu_()

        X_obs = X_obs.permute(2, 0, 1)
        X_obs = (X_obs * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp = self.gru_d(X_obs, h[i_obs].clone())

        H_obs = self.gcn(X_obs_orig, kwargs.get('adj'))
        X_obs = torch.stack([X_obs_orig, H_obs, mean, logvar, error], dim=2).unsqueeze(2)
        X_obs = torch.matmul(X_obs, self.w_prep2).squeeze(2) + self.bias_prep
        X_obs.relu_()

        X_obs = X_obs.permute(2, 0, 1)
        X_obs = (X_obs * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp2 = h.clone()
        temp2 = self.gru_d2(X_obs, h[i_obs].clone())
        temp = self.merge(torch.cat([temp, temp2], -1))
        h[i_obs] = temp.clone()

        return h, losses

class GRUObservationCellLogvar(nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)

        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p_obs, X_obs, M_obs, i_obs):
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        log_lik_c = np.log(np.sqrt(2*np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2*log_lik_c) * M_obs)

        assert not losses.sum() != losses.sum()

        X_obs = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        X_obs = torch.matmul(X_obs, self.w_prep).squeeze(2) + self.bias_prep
        X_obs.relu_()

        X_obs = X_obs.permute(2, 0, 1)
        X_obs = (X_obs * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)

        temp = h.clone()
        temp = self.gru_d(X_obs, h[i_obs].clone())
        h[i_obs] = temp.clone()

        return h, losses


def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.pow(torch.abs(var) + 1e-5,0.5)

    return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)
