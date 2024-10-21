import torch
from tqdm import tqdm

from nfe.experiments.gru_ode_bayes.lib.data_utils import *
from nfe.train_utils import delta_time
from datetime import datetime


def validate(model, dl, device, delta_t, adj, **kwargs):
    with torch.no_grad():
        loss_val = 0
        mse_val = 0
        num_obs = 0
        pbar = tqdm(dl)
        for b in pbar:
            # just for demo dataset
            if not torch.cuda.is_available():
                b['X_val'] = b['X']
                b['times_val'] = b['times']
                b['M_val'] = b['M']
            assert b['X_val'] is not None

            _, _, _, _, _, p_vec = model(times=b['times'], num_obs=b['num_obs'], X=b['X'].to(device),
                                         M=b['M'].to(device), delta_t=delta_t, cov=b['cov'].to(device),
                                         return_path=True, val_times=b['times_val'], adj=adj)
            m, v = torch.chunk(p_vec, 2, dim=1)

            z_reord, mask_reord = [], []
            val_numobs = torch.Tensor([len(x) for x in b['times_val']])
            for ind in range(0, int(torch.max(val_numobs).item())):
                idx = val_numobs > ind
                zero_tens = torch.Tensor([0])
                z_reord.append(b['X_val'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                               [:-1][idx] + ind).long()])
                mask_reord.append(b['M_val'][(torch.cat((zero_tens, torch.cumsum(val_numobs, dim=0)))
                                                     [:-1][idx] + ind).long()])

            X_val = torch.cat(z_reord).to(device)
            M_val = torch.cat(mask_reord).to(device)

            last_loss = (log_lik_gaussian(X_val, m, v) * M_val).sum()
            mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()

            loss_val += last_loss.cpu().numpy()
            num_obs += M_val.sum().cpu().numpy()
            mse_val += mse_loss.cpu().numpy()

            stage = kwargs.get('stage')
            ep = kwargs.get('ep')
            start = kwargs.get('s')
            d = {
                'data': kwargs.get('data'),
                'mod': kwargs.get('mod'),
                'nll': f'{loss_val / num_obs:.4e}',
                'mse': f'{mse_val / num_obs:.4e}',
                'run': kwargs.get('run'),
                'w': kwargs.get('w')
            }
            end = delta_time(start, datetime.now())
            pbar.set_description_str(f'{stage} {ep} \t\t {d} \t\t{end} \t\t\t')

        loss_val /= num_obs
        mse_val /= num_obs

    return loss_val, mse_val
