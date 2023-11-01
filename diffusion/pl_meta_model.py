import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from pytorch_lightning.utilities import rank_zero_info

from models.gnn_encoder import GNNEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import CategoricalDiffusion
import time

import seaborn as sns
import matplotlib.pyplot as plt


class COMetaModel(pl.LightningModule):
    def __init__(self,
                 param_args,
                 node_feature_only=False):
        super(COMetaModel, self).__init__()
        self.args = param_args
        self.diffusion_schedule = self.args.diffusion_schedule
        self.diffusion_steps = self.args.diffusion_steps
        self.sparse = self.args.sparse_factor > 0 or node_feature_only

        out_channels = 2
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule)

        self.model = GNNEncoder(
            n_layers=self.args.n_layers,
            hidden_dim=self.args.hidden_dim,
            out_channels=out_channels,
            aggregation=self.args.aggregation,
            sparse=self.sparse,
            use_activation_checkpoint=self.args.use_activation_checkpoint,
            node_feature_only=node_feature_only,
        )

    def test_epoch_end(self, outputs):
        unmerged_metrics = {}
        for metrics in outputs:
            for k, v in metrics.items():
                if k not in unmerged_metrics:
                    unmerged_metrics[k] = []
                unmerged_metrics[k].append(v)

        merged_metrics = {}
        for k, v in unmerged_metrics.items():
            merged_metrics[k] = float(np.mean(v))
        self.logger.log_metrics(merged_metrics, step=self.global_step)

    def categorical_posterior(self, target_t, t, x0_pred_prob, xt):
        """Sample from the categorical posterior for a given time step.
       See https://arxiv.org/pdf/2107.03006.pdf for details.
    """
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)

        Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        # plt.clf()
        # sns.heatmap(sum_x_t_target_prob.clamp(0, 1).float().detach().cpu()[0])
        # plt.savefig('imgs/original_{}.png'.format(target_t[0]))

        if target_t > 0:
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        if self.sparse:
            xt = xt.reshape(-1)
        return xt, sum_x_t_target_prob.clamp(0, 1)

    def guided_categorical_posterior(self, target_t, t, x0_pred_prob, xt, grad=None):
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with torch.no_grad():
            diffusion = self.diffusion
            if target_t is None:
                target_t = t - 1
            else:
                target_t = torch.from_numpy(target_t).view(1)

            if target_t > 0:
                Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
                Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)  # [2, 2], transition matrix
            else:
                Q_t = torch.eye(2).float().to(x0_pred_prob.device)
            Q_bar_t_source = torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
            Q_bar_t_target = torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)

            xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2), \
                torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 1, 2)
            xt_grad_zero[..., 0] = (1 - xt) * grad
            xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
            xt_grad_one[..., 1] = xt * grad
            xt_grad_one[..., 0] = -xt_grad_one[..., 1]
            xt_grad = xt_grad_zero + xt_grad_one

            xt = F.one_hot(xt.long(), num_classes=2).float()
            xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

            # q(xt−1|xt,x0=0)pθ(x0=0|xt)
            x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
            x_t_target_prob_part_2 = Q_bar_t_target[0]
            x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

            x_t_target_prob = (x_t_target_prob_part_1 * x_t_target_prob_part_2) / x_t_target_prob_part_3  # [b, n, n, 2]

            sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

            # q(xt−1|xt,x0=1)pθ(x0=1|xt)
            x_t_target_prob_part_2_new = Q_bar_t_target[1]
            x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

            x_t_source_prob_new = (x_t_target_prob_part_1 * x_t_target_prob_part_2_new) / x_t_target_prob_part_3_new

            sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

            p_theta = torch.cat((1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)), dim=-1)
            p_phi = torch.exp(-xt_grad)
            if self.sparse:
                p_phi = p_phi.reshape(p_theta.shape)
            posterior = (p_theta * p_phi) / torch.sum((p_theta * p_phi), dim=-1, keepdim=True)

            # plt.clf()
            # sns.heatmap(posterior[..., 1].clamp(0, 1).float().detach().cpu()[0])
            # plt.savefig('imgs/guided_{}.png'.format(target_t[0]))

            if target_t > 0:
                xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
            else:
                xt = posterior[..., 1].clamp(min=0)
            if self.sparse:
                xt = xt.reshape(-1)
            return xt

    def duplicate_edge_index(self, parallel_sampling, edge_index, num_nodes, device):
        """Duplicate the edge index (in sparse graphs) for parallel sampling."""
        edge_index = edge_index.reshape((2, 1, -1))
        edge_index_indent = torch.arange(0, parallel_sampling).view(1, -1, 1).to(device)
        edge_index_indent = edge_index_indent * num_nodes
        edge_index = edge_index + edge_index_indent
        edge_index = edge_index.reshape((2, -1))
        return edge_index

    def test_dataloader(self):
        batch_size = 1
        print("Test dataset size:", len(self.test_dataset))
        test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader

    def val_dataloader(self):
        batch_size = 1
        val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
        print("Validation dataset size:", len(val_dataset))
        val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader

