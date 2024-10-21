import torch

from nfe.experiments.latent_ode.lib.create_latent_ode_model import create_LatentODE_model
from nfe.experiments.latent_ode.lib.parse_datasets import parse_datasets
from nfe.experiments.latent_ode.lib.utils import compute_loss_all_batches

from nfe.experiments import BaseExperiment


class LatentODE(BaseExperiment):
    def get_model(self, args):

        z0_prior = torch.distributions.Normal(
            torch.Tensor([0.0]).to(self.device),
            torch.Tensor([1.]).to(self.device)
        )

        obsrv_std = 0.001 if args.data == 'hopper' else 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(self.device)

        model = create_LatentODE_model(args, self.dim, z0_prior, obsrv_std, self.device, n_labels=self.n_classes)
        return model

    def get_data(self, args):
        return parse_datasets(args, self.device)

    def training_step(self, batch, adj=None):
        loss = self.model.module.compute_all_losses(batch, adj)
        return loss['loss']

    def _get_loss(self, dl, adj, **kwargs):
        loss = compute_loss_all_batches(model=self.model,
                                        dl=dl, adj=adj,
                                        args=self.args,
                                        device=self.device,
                                        n_traj_samples=1,
                                        **kwargs)
        return loss['loss'], loss['mse'], loss['acc']

    def validation_step(self, adj=None, **kwargs):
        loss, mse, acc = self._get_loss(self.dlval, adj, **kwargs)
        # self.logger.info(f'val_mse={mse:.5f}')
        # self.logger.info(f'val_acc={acc:.5f}')
        # return loss
        return {'loss': f'{loss.detach().item():.4e}', 'mse': f'{mse.detach().item():.4e}', 'acc': f'{acc:.4e}'}

    def test_step(self, adj=None, **kwargs):
        loss, mse, acc = self._get_loss(self.dltest, adj, **kwargs)
        # self.logger.info(f'test_mse={mse:.5f}\ttest_acc={acc:.5f}')
        return {'loss': f'{loss.detach().item():.4e}', 'mse': f'{mse.detach().item():.4e}', 'acc': f'{acc:.4e}'}

    def finish(self):
        pass
        # OUT_DIR = ...
        # torch.save(self.model.state_dict(), OUT_DIR / 'model.pt')
