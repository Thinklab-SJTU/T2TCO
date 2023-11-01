import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_sparse import SparseTensor
from co_datasets.mis_dataset import MISDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
from utils.mis_utils import mis_decode_np


class MISModel(COMetaModel):
  def __init__(self, param_args=None):
    super(MISModel, self).__init__(param_args=param_args, node_feature_only=True)

    self.test_dataset = MISDataset(
      data_file=os.path.join(self.args.storage_path, self.args.test_split),
    )

    self.validation_dataset = MISDataset(
      data_file=os.path.join(self.args.storage_path, self.args.validation_split),
    )

  def forward(self, x, t, edge_index):
    return self.model(x, t, edge_index=edge_index)

  def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
      )
      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      xt, _ = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt

  def guided_categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    torch.set_grad_enabled(True)
    xt = xt.float()  # n if sparse
    xt.requires_grad = True
    t = torch.from_numpy(t).view(1)

    with torch.inference_mode(False):
      x0_pred = self.forward(
        xt.to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
      )

      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
      num_nodes = xt.shape[0]
      adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(num_nodes, num_nodes),
      ).to_dense()
      adj_matrix.fill_diagonal_(0)

      pred_nodes = x0_pred_prob[..., 1].squeeze(0)
      # cost_est = 1 - pred_nodes / num_nodes
      f_mis = -pred_nodes.sum()
      g_mis = adj_matrix @ pred_nodes
      g_mis = (pred_nodes * g_mis).sum()
      cost_est = f_mis + 0.5 * g_mis
      cost_est.requires_grad_(True)
      cost_est.backward()
      assert xt.grad is not None

      if self.args.norm is True:
        xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
      xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)

    return xt.detach()

  def test_step(self, batch, batch_idx, draw=False, split='test'):
    device = batch[-1].device

    real_batch_idx, graph_data, point_indicator = batch
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    stacked_predict_labels = []
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy()
    adj_mat = scipy.sparse.coo_matrix(
      (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
    )

    if self.args.parallel_sampling > 1:
      edge_index = self.duplicate_edge_index(self.args.parallel_sampling, edge_index, node_labels.shape[0], device)

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(node_labels.float())
      if self.args.parallel_sampling > 1:
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randn_like(xt)
      xt = (xt > 0).long()
      xt = xt.reshape(-1)

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)

        xt = self.categorical_denoise_step(xt, t1, device, edge_index, target_t=t2)

      predict_labels = xt.float().cpu().detach().numpy() + 1e-6

      stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis=0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [mis_decode_np(predict_labels, adj_mat) for predict_labels in splitted_predict_labels]
    solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)
    best_solved_id = np.argmax(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum()

    guided_gap, g_best_solved_cost = -1, -1
    if self.args.rewrite:
      g_best_solution = solved_solutions[best_solved_id]
      for _ in range(self.args.rewrite_steps):
        g_stacked_predict_labels = []
        g_x0 = torch.from_numpy(g_best_solution).unsqueeze(0).to(device)
        g_x0 = F.one_hot(g_x0.long(), num_classes=2).float()

        steps_T = int(self.args.diffusion_steps * self.args.rewrite_ratio)
        # steps_inf = int(self.args.inference_diffusion_steps * self.args.rewrite_ratio)
        steps_inf = self.args.inference_steps

        time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                          T=steps_T, inference_T=steps_inf)

        Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0.device)
        g_xt_prob = torch.matmul(g_x0, Q_bar)  # [B, N, 2]
        g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1)).to(g_x0.device)  # [B, N]
        g_xt = g_xt * 2 - 1  # project to [-1, 1]
        g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise

        if self.args.parallel_sampling > 1:
          g_xt = g_xt.repeat(self.args.parallel_sampling, 1, 1)

        g_xt = (g_xt > 0).long().reshape(-1)
        for i in range(steps_inf):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1]).astype(int)
          t2 = np.array([t2]).astype(int)
          g_xt = self.guided_categorical_denoise_step(g_xt, t1, device, edge_index, target_t=t2)

        g_predict_labels = g_xt.float().cpu().detach().numpy() + 1e-6
        g_stacked_predict_labels.append(g_predict_labels)
        g_predict_labels = np.concatenate(g_stacked_predict_labels, axis=0)

        g_splitted_predict_labels = np.split(g_predict_labels, self.args.parallel_sampling)
        g_solved_solutions = [mis_decode_np(g_predict_labels, adj_mat) for g_predict_labels in g_splitted_predict_labels]
        g_solved_costs = [g_solved_solution.sum() for g_solved_solution in g_solved_solutions]
        g_best_solved_cost = np.max([g_best_solved_cost, np.max(g_solved_costs)])
        g_best_solved_id = np.argmax(g_solved_costs)

        g_best_solution = g_solved_solutions[g_best_solved_id]

      print(f'tot_points: {g_x0.shape[-2]}, gt_cost: {gt_cost}, selected_points: {best_solved_cost} -> {g_best_solved_cost}')

    metrics = {
        f"{split}/rewrite_ratio": self.args.rewrite_ratio,
        f"{split}/norm": self.args.norm,
        # f"{split}/gap": gap,
        # f"{split}/guided_gap": guided_gap,
        f"{split}/gt_cost": gt_cost,
        f"{split}/guided_solved_cost": g_best_solved_cost,
    }
    for k, v in metrics.items():
        self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

    return metrics

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')

