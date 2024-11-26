from datetime import datetime
from argparse import Namespace
from copy import deepcopy
from logging import Logger
import random
from random import randint
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn.init import xavier_uniform_
from torch.nn.functional import normalize
from tqdm import tqdm

from nfe.experiments.synthetic.generate import make_A
from nfe.experiments.synthetic.utils import count_accuracy, is_dag
from nfe.train_utils import delta_time, set_seed
from torch.nn.parallel import DistributedDataParallel, DataParallel

torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
np.set_printoptions(suppress=False, precision=3)
cuda = torch.cuda.is_available()

class BaseExperiment:
    """ Base experiment class """
    def __init__(self, args: Namespace, logger: Logger):
        self.logger = logger
        self.args = args
        self.epochs = args.epochs
        self.patience = args.patience
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.training_scheme = args.training_scheme
        self.seed = args.seed

        set_seed(self.seed)

        self.device = torch.device('cuda' if cuda else 'cpu')

        self.model_name = self.args.flow_model if self.args.model == 'flow' else self.args.model
        self.exp_path = f'nfe/output/{self.args.data}/{self.training_scheme}_{self.model_name}'
        self.noise = f'_n{self.args.scale:.0e}' if self.args.init == 'truth' else ''
        self.run_details = f's{self.args.seed}_ts{self.args.n_ts}{self.noise}'

        self.dim, self.n_classes, self.dltrain, self.dlval, self.dltest = self.get_data(args)
        self.model = self.get_model(args).to(self.device)
        self.learned_dag = None
        self.thresh = self.args.thresh

        model_ = args.flow_model if args.model == 'flow' else args.solver
        d = {'sink': 2, 'ellipse': 2, 'activity': 3}
        self.n_sensors = self.dim // d[args.data] if args.data in list(d.keys()) else self.dim

        time = datetime.now().strftime('%d %b %Y %H:%M')
        nts = f'n={args.n_ts}' if args.experiment == 'synthetic' else ""
        self.logger.info(f'\n\n{time}\n{args.training_scheme} {args.data} {args.model} {model_} seed:{args.seed} {nts}')
        self.model = DataParallel(self.model).to(self.device)
        if args.experiment == 'synthetic' and args.training_scheme in ['lgnf', 'tgnf']:
            self.true_dag = torch.load(f'nfe/experiments/data/synth{args.n_ts}g/dag.pt', map_location=torch.device(self.device))

        _freq = {'sink': 15 if args.training_scheme in ['lgnf'] else 25, 'triangle': 15, 'sawtooth': 20, 'square': 30,
                 'physionet': 1, 'mimic4': 1, 'mimic3': 1}

        self.eval_freq = _freq[args.data]
        self.scheduler, self.optim_adj, self.A = None, None, None
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, self.args.lr_scheduler_step, self.args.lr_decay)
        self.best_loss = np.inf
        self.waiting = 0
        self.best_model_dic = None
        self.start = datetime.now()
        self.best_A = None

    def train(self):
        """Training loop"""
        if self.args.training_scheme == "lgnf":
            return self.learn_dag()
        best_loss = float('inf')
        waiting = 0
        best_model = deepcopy(self.model.state_dict())
        start = datetime.now()
        adj = None
        if self.args.experiment == 'synthetic' and self.args.training_scheme == "tgnf":
            a = self.true_dag.copy()
            adj = torch.tensor(a, dtype=torch.float32).to(self.device)

        for epoch in tqdm(range(self.epochs), desc=f'{self.args.training_scheme} {self.args.data}'):
            self.model.train()

            for batch in self.dltrain:
                self.optim.zero_grad()
                train_loss = self.training_step(batch, adj)
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optim.step()

            # Validation step
            self.model.eval()
            val_loss = self.validation_step(adj)
            end = delta_time(start, datetime.now())
            if epoch % self.eval_freq == 0:
                val_str = list(map(lambda k, v: f'{k}:{v:.3e}', val_loss.keys(), val_loss.values()))[0]
                self.logger.info(f'[epoch={epoch + 1:04d}]\t{val_str}\twait {waiting}\t{end}')
            val_loss = list(val_loss.values())[0]

            # Learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early stopping procedure
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(self.model.state_dict())
                waiting = 0
            elif waiting > self.patience:
                break
            else:
                waiting += 1

        # Load best model
        self.model.load_state_dict(best_model)

        # Held-out test set step
        test_loss = self.test_step(adj)
        loss = test_loss
        test_str = list(map(lambda k, v: f'{k}:{v:.3e}', loss.keys(), loss.values()))[0]
        self.logger.info(f'test: {self.args.training_scheme} s{self.args.seed} {test_str}')

        return test_loss

    def init_dag(self):
        if self.args.init == 'xavier':
            set_seed(self.seed)
            init = xavier_uniform_(torch.zeros([self.n_sensors, self.n_sensors]))  # .abs()
            init = init.fill_diagonal_(0.0)
            A = torch.tensor(init, requires_grad=True, device=self.device)
        elif self.args.init == 'truth':
            a = self.true_dag.copy()
            noise = np.random.normal(loc=0, scale=self.args.scale * np.max(a), size=a.shape)
            # a += self.args.scale
            a += noise
            A = torch.tensor(a, dtype=torch.float32, requires_grad=True, device=self.device)
        else:
            raise SystemExit('init not set')
        self.A = A

    def train_epoch(self, rho, alpha):
        self.model.train()
        for batch in self.dltrain:
            self.optim_adj.zero_grad()
            A_hat = torch.divide(self.A.T, self.A.max().detach()).T
            loss = self.training_step(batch, A_hat)
            h = torch.trace(torch.matrix_exp(A_hat * A_hat)) - self.n_sensors
            penalty = (0.5 * rho * h * h) + (alpha * h)
            total_loss = loss + penalty
            assert torch.isnan(total_loss) is not False
            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args.clip)
            clip_grad_norm_(self.A, self.args.clip)
            self.optim_adj.step()
            self.A.data.copy_(torch.clamp(self.A.data, min=0, max=1))

        self.scheduler.step()

        # Validation step
        self.model.eval()
        with torch.no_grad():
            val_loss = self.validation_step(self.A.data)
            val_loss_f = list(val_loss.values())[0]
        if val_loss_f < self.best_loss:
            self.best_loss = val_loss_f
            self.best_model_dic = deepcopy(self.model.state_dict())
            self.best_A = self.A.data.clone()
            self.waiting = 0
        elif self.waiting > self.patience:
            self.waiting = -1
        else:
            self.waiting += 1

        return h, val_loss

    def log_val_str(self, val_loss, epoch):
        end = delta_time(self.start, datetime.now())
        val_str = list(map(lambda k, v: f'{k}:{v:.3e}', val_loss.keys(), val_loss.values()))[0]
        self.logger.info(f'[epoch={epoch + 1:04d}]\t{val_str}\twait {self.waiting}\t{end}')

    def train_loop(self, n_epochs, rho, alpha, dag_iter):
        self.waiting = 0
        self.best_loss = np.inf
        self.optim_adj = torch.optim.Adam(params=[{'params': self.model.parameters(), 'weight_decay': self.weight_decay}, {'params': [self.A]}], lr=self.lr, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim_adj, self.args.lr_scheduler_step, self.args.lr_decay)

        for epoch in range(n_epochs):
            h, val_loss = self.train_epoch(rho, alpha)
            if epoch % self.eval_freq == 0 and dag_iter == -1:
                self.log_val_str(val_loss, epoch)
            if self.waiting == -1:
                self.log_val_str(val_loss, epoch)
                break

        # Held-out test set step
        with torch.no_grad():
            self.model.load_state_dict(self.best_model_dic)
            test_loss = self.test_step(self.best_A.data)

        test_str = list(map(lambda k, v: f'{k}:{v:.3e}', test_loss.keys(), test_loss.values()))[0]
        if dag_iter == 0:
            a_str = f'a:{alpha:.2e}'
            dag_str = f'\trho:{rho:.2e}\th:{h.item():.2e}'
            log_str = f'{test_str}\t{dag_str}\t{a_str}'
            self.logger.info(log_str)

        del self.optim_adj
        del self.scheduler
        torch.cuda.empty_cache()

        return test_str[4:], h

    def evaluate_dag(self):
        dag = self.A.data.cpu().numpy()
        thres = self.thresh
        b_est = (abs(dag) >= thres).astype(int)
        b_true = (abs(self.true_dag) >= thres).astype(int)
        metrics = count_accuracy(b_est, b_true)
        self.learned_dag = self.A.data.clone()
        return metrics

    def learn_dag(self):
        """loop to learn DAG"""
        rho = 1
        alpha = 0
        rho_max = self.args.rho_max
        max_iter = self.args.max_iter
        h_A_old = np.inf
        h_tol = self.args.h_tol
        rho_ = self.args.rho
        h_par = self.args.h_par
        dag_epoch = self.args.dag_epochs
        _thresh = self.args.thresh

        dag_ = f'h:{h_par}\trho:{rho_}\trho_max:{rho_max:.0e}\th_tol:{h_tol:.0e}\tgnn:{self.args.gnn_layers}'
        train_ = f'eps:{dag_epoch}\tthresh:{self.thresh}\tinit:{self.args.init}'
        train_ = f'{train_}\tnoise:{self.args.scale}' if self.args.init == 'truth' else train_
        self.logger.info(f'{dag_}\t{train_}')

        self.init_dag()
        test_loss = ''

        # optim algo for DAG parameters
        for _ in tqdm(range(max_iter), desc=f'dag {self.args.data} l:{test_loss}'):
            ix = 0
            while rho < rho_max:
                _, h = self.train_loop(self.args.dag_epochs, rho, alpha, dag_iter=ix)
                ix += 1
                if h.item() > h_par * h_A_old:
                    rho *= rho_
                else:
                    break
            h_A_old = h.item()
            alpha += rho * h.item()
            if h_A_old <= h_tol or rho >= rho_max:
                break

        metrics = self.evaluate_dag()

        # train after DAG params are found
        test_str, _ = self.train_loop(self.args.epochs, rho, alpha, dag_iter=-1)
        self.evaluate_dag()
        metrics['test_loss'] = test_str
        self.logger.info(metrics)

    def reg_step(self):
        l1 = self.model.l1_reg()
        l2 = self.model.l2_reg()
        return l1, l2

    def get_model(self, args: Namespace) -> Module:
        raise NotImplementedError

    def get_data(
            self,
            args: Namespace,
        ) -> Tuple[int, int, DataLoader, DataLoader, DataLoader]:
        # Returns dim, n_classes, 3 dataLoaders (train, val, test)
        raise NotImplementedError

    def training_step(self, batch: Any, adj=None) -> Tensor:
        # Returns training loss (scalar)
        raise NotImplementedError

    def validation_step(self, adj=None):
        # Returns validation loss (scalar)
        raise NotImplementedError

    def test_step(self, adj=None):
        # Returns test loss (scalar)
        raise NotImplementedError

    def finish(self) -> None:
        # Performs (optional) final operations
        # e.g. save model, plot...
        return


