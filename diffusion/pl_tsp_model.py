import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
# import sys
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, "..")  # for pyconcorde
# from pyconcorde.concorde.tsp import TSPSolver
# import elkai
from co_datasets.tsp_graph_dataset import TSPGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_tours
from torch.autograd import Variable
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt


class TSPModel(COMetaModel):
  def __init__(self, param_args=None):
    super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    self.test_dataset = TSPGraphDataset(
      data_file=os.path.join(self.args.storage_path, self.args.test_split),
      sparse_factor=self.args.sparse_factor,
    )

    self.validation_dataset = TSPGraphDataset(
      data_file=os.path.join(self.args.storage_path, self.args.validation_split),
      sparse_factor=self.args.sparse_factor,
    )

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
        points.float().to(device),
        xt.float().to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)  # b, n, n, 2
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      xt, xt_prob = self.categorical_posterior(target_t, t, x0_pred_prob, xt)  # [b, n, n] 0 or 1
      return xt, xt_prob

  def guided_categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    torch.set_grad_enabled(True)
    xt = xt.float()  # b, n, n
    xt.requires_grad = True
    t = torch.from_numpy(t).view(1)

    # [b, 2, n, n]
    with torch.inference_mode(False):
      # print(points.shape, xt.shape)
      x0_pred = self.forward(
        points.float().to(device),
        xt.to(device),
        t.float().to(device),
        edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

      if not self.sparse:
        dis_matrix = self.points2adj(points)
        cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
        cost_est.requires_grad_(True)
        cost_est.backward()
      else:
        dis_matrix = torch.sqrt(torch.sum((points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2, dim=1))
        dis_matrix = dis_matrix.reshape((1, points.shape[0], -1))
        cost_est = (dis_matrix * x0_pred_prob[..., 1]).sum()
        cost_est.requires_grad_(True)
        cost_est.backward()
      assert xt.grad is not None

      if self.args.norm is True:
        xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
      xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)

      return xt.detach()

  def test_step(self, batch, batch_idx, split='test'):
    edge_index = None
    original_edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      original_edge_index = edge_index.clone()
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
    gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

    # print(points.shape, edge_index.shape, batch_size, adj_matrix.shape)

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(self.args.parallel_sampling, edge_index, np_points.shape[0], device)

    # Initialize with original diffusion
    stacked_tours = []
    ns, merge_iterations = 0, 0

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)  # [B, E]
        xt = torch.randn_like(xt)

      xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)  # [E]

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        # [B, N, N], heatmap score
        xt, xt_prob = self.categorical_denoise_step(
          points, xt, t1, device, edge_index, target_t=t2)

      adj_mat = xt.float().cpu().detach().numpy() + 1e-6  # [B, N, N]

      if self.args.save_numpy_heatmap and not self.args.rewrite:
        self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)

      tours, merge_iterations = merge_tours(  # [B, N+1], list
          adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
      )

      # Refine using 2-opt
      # solver_tours,  [B, N+1] ndarray, the visiting sequence of each city
      solved_tours, ns = batched_two_opt_torch(
          np_points.astype("float64"), np.array(tours).astype('int64'),
          max_iterations=self.args.two_opt_iterations, device=device
      )
      stacked_tours.append(solved_tours)

    solved_tours = np.concatenate(stacked_tours, axis=0)  # [B, N+1] ndarray

    tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
    gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_costs = [tsp_solver.evaluate(solved_tours[i]) for i in range(total_sampling)]
    best_solved_cost, best_id = np.min(all_solved_costs), np.argmin(all_solved_costs)
    gap = (best_solved_cost - gt_cost) / gt_cost * 100

    # print("gap: {}%".format((best_solved_cost - gt_cost) / gt_cost * 100))

    # select the best tour
    g_best_tour = solved_tours[best_id]  # [N+1] ndarray

    guided_gap, g_ns, g_merge_iterations, g_best_solved_cost = -1, -1, -1, -1

    # Local Rewrite
    if self.args.rewrite:
      g_best_solved_cost = best_solved_cost

      for _ in range(self.args.rewrite_steps):
        g_stacked_tours = []
        # optimal adjacent matrix
        g_x0 = self.tour2adj(g_best_tour, np_points, self.sparse, self.args.sparse_factor, original_edge_index)
        g_x0 = g_x0.unsqueeze(0).to(device)  # [1, N, N] or [1, N]
        if self.args.parallel_sampling > 1:
          if not self.sparse:
            g_x0 = g_x0.repeat(self.args.parallel_sampling, 1, 1)  # [1, N ,N] -> [B, N, N]
          else:
            g_x0 = g_x0.repeat(self.args.parallel_sampling, 1)

        if self.sparse:
          g_x0 = g_x0.reshape(-1)

        g_x0_onehot = F.one_hot(g_x0.long(), num_classes=2).float()  # [B, N, N, 2]
        # if self.sparse:
        #   g_x0_onehot = g_x0_onehot.unsqueeze(1)

        steps_T = int(self.args.diffusion_steps * self.args.rewrite_ratio)
        steps_inf = self.args.inference_steps
        time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                          T=steps_T, inference_T=steps_inf)

        # g_xt = self.diffusion.sample(g_x0_onehot, steps_T)
        Q_bar = torch.from_numpy(self.diffusion.Q_bar[steps_T]).float().to(g_x0_onehot.device)
        g_xt_prob = torch.matmul(g_x0_onehot, Q_bar)  # [B, N, N, 2]

        # add noise for the steps_T samples, namely rewrite
        g_xt = torch.bernoulli(g_xt_prob[..., 1].clamp(0, 1))  # [B, N, N]
        g_xt = g_xt * 2 - 1  # project to [-1, 1]
        g_xt = g_xt * (1.0 + 0.05 * torch.rand_like(g_xt))  # add noise
        g_xt = (g_xt > 0).long()

        for i in range(steps_inf):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1]).astype(int)
          t2 = np.array([t2]).astype(int)

          # [1, N, N], denoise, heatmap for edges
          g_xt = self.guided_categorical_denoise_step(points, g_xt, t1, device, edge_index, target_t=t2)

        g_adj_mat = g_xt.float().cpu().detach().numpy() + 1e-6
        if self.args.save_numpy_heatmap:
          self.run_save_numpy_heatmap(g_adj_mat, np_points, real_batch_idx, split)

        g_tours, g_merge_iterations = merge_tours(
          g_adj_mat, np_points, np_edge_index,
          sparse_graph=self.sparse,
          parallel_sampling=self.args.parallel_sampling,
        )

        # Refine using 2-opt
        g_solved_tours, g_ns = batched_two_opt_torch(
            np_points.astype("float64"), np.array(g_tours).astype('int64'),
            max_iterations=self.args.two_opt_iterations, device=device
        )
        g_stacked_tours.append(g_solved_tours)

        g_solved_tours = np.concatenate(g_stacked_tours, axis=0)

        tsp_solver = TSPEvaluator(np_points)  # np_points: [N, 2] ndarray
        gt_cost = tsp_solver.evaluate(np_gt_tour)  # np_gt_tour: [N+1] ndarray

        g_total_sampling = self.args.parallel_sampling
        g_all_solved_costs = [tsp_solver.evaluate(g_solved_tours[i]) for i in range(g_total_sampling)]
        g_best_solved_cost_tmp, g_best_id = np.min(g_all_solved_costs), np.argmin(g_all_solved_costs)
        g_best_solved_cost = min(g_best_solved_cost, g_best_solved_cost_tmp)

        guided_gap = (g_best_solved_cost - gt_cost) / gt_cost * 100

        # select the best tour
        g_best_tour = g_solved_tours[g_best_id]

      # print("gap: {}% -> {}%".format(gap, guided_gap))

    metrics = {
        f"{split}/rewrite_ratio": self.args.rewrite_ratio,
        f"{split}/norm": self.args.norm,
        f"{split}/inference_step": self.args.inference_diffusion_steps,
        f"{split}/gap": gap,
        f"{split}/guided_gap": guided_gap,
        f"{split}/gt_cost": gt_cost,
        f"{split}/2opt_iterations": g_ns,
        f"{split}/merge_iterations": g_merge_iterations,
        f"{split}/guided_solved_cost": g_best_solved_cost,
    }

    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split, cost=None, path=None):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap') if path is None else path
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)
    if cost is not None:
      np.save(os.path.join(heatmap_path, f"{split}-cost-{real_batch_idx}.npy"), cost)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')

  def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
    if not sparse:
      adj_matrix = torch.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
    else:
      adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
      adj_matrix[tour[:-1]] = tour[1:]
      adj_matrix = torch.from_numpy(adj_matrix)
      adj_matrix = adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
    return adj_matrix

  def points2adj(self, points):
    """
    return distance matrix
    Args:
      points: b, n, 2
    Returns: b, n, n
    """
    assert points.dim() == 3
    return torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5

