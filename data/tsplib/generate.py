import time
import argparse
import pprint as pp
import os

import numpy as np
# from concorde.tsp import TSPSolver
from tsp_file_parser import TSPParser
import scipy
import elkai

class TSPEvaluator(object):
  def __init__(self, points):
    self.dist_mat = scipy.spatial.distance_matrix(points, points)

  def evaluate(self, route):
    total_cost = 0
    for i in range(len(route) - 1):
      total_cost += self.dist_mat[route[i], route[i + 1]]
    return total_cost
  
def lkh_solve(np_points):
  # lkh
  cities = elkai.Coordinates2D({idx: (x, y) for idx, (x, y) in enumerate(np_points)},)
  return np.array(cities.solve_tsp(runs=10))


# tsplib50-200
solution_list = []
with open("tsplib50-200.txt", "w") as f:
  start_time = time.time()

  for filename in os.listdir("./tsplib50-200"):
    nodes_coord = np.array(TSPParser("./tsplib50-200/" + filename, False).tsp_cities_list)
    # normalize each dimension to [0,1]
    if nodes_coord[:, 0].max() - nodes_coord[:, 0].min() > 0:
      nodes_coord[:, 0] = (nodes_coord[:, 0] - nodes_coord[:, 0].min()) / (nodes_coord[:, 0].max() - nodes_coord[:, 0].min())
    if nodes_coord[:, 1].max() - nodes_coord[:, 1].min() > 0:
      nodes_coord[:, 1] = (nodes_coord[:, 1] - nodes_coord[:, 1].min()) / (nodes_coord[:, 1].max() - nodes_coord[:, 1].min())

    solved_tour = lkh_solve(nodes_coord*10000)

    tsp_solver = TSPEvaluator(nodes_coord)  # np_points: [N, 2] ndarray
    gt_cost = tsp_solver.evaluate(solved_tour)  # np_gt_tour: [N+1] ndarray
    solution_list.append(gt_cost)

    # Only write instances with valid solutions
    if (np.sort(solved_tour[:-1]) == np.arange(nodes_coord.shape[0])).all():
        f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
        f.write( str(" ") + str('output') + str(" ") )
        f.write( str(" ").join( str(node_idx+1) for node_idx in solved_tour[:-1]) )
        f.write( str(" ") + str(solved_tour[:-1][0]+1) + str(" ") )
        f.write( "\n" )

# tsplib200-1000
solution_list = []
with open("tsplib200-1000.txt", "w") as f:
  start_time = time.time()

  for filename in os.listdir("./tsplib200-1000"):
    nodes_coord = np.array(TSPParser("./tsplib200-1000/" + filename, False).tsp_cities_list)
    # normalize each dimension to [0,1]
    if nodes_coord[:, 0].max() - nodes_coord[:, 0].min() > 0:
      nodes_coord[:, 0] = (nodes_coord[:, 0] - nodes_coord[:, 0].min()) / (nodes_coord[:, 0].max() - nodes_coord[:, 0].min())
    if nodes_coord[:, 1].max() - nodes_coord[:, 1].min() > 0:
      nodes_coord[:, 1] = (nodes_coord[:, 1] - nodes_coord[:, 1].min()) / (nodes_coord[:, 1].max() - nodes_coord[:, 1].min())

    solved_tour = lkh_solve(nodes_coord*100000)

    tsp_solver = TSPEvaluator(nodes_coord)  # np_points: [N, 2] ndarray
    gt_cost = tsp_solver.evaluate(solved_tour)  # np_gt_tour: [N+1] ndarray
    solution_list.append(gt_cost)

    # Only write instances with valid solutions
    if (np.sort(solved_tour[:-1]) == np.arange(nodes_coord.shape[0])).all():
        f.write( " ".join( str(x)+str(" ")+str(y) for x,y in nodes_coord) )
        f.write( str(" ") + str('output') + str(" ") )
        f.write( str(" ").join( str(node_idx+1) for node_idx in solved_tour[:-1]) )
        f.write( str(" ") + str(solved_tour[:-1][0]+1) + str(" ") )
        f.write( "\n" )