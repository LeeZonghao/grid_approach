import numpy as np
from numba import njit

class DeltaCalculator:
    @staticmethod
    @njit
    def delta1(i, proposed_v, V, k):
        # Your delta_1 implementation
        pass

    @staticmethod
    @njit
    def delta2_optimized(k, e, V, proposed_v, i, A, M_3, m):
        # Your delta_2_optimized_fast implementation
        pass

class NeighborTracker:
    @staticmethod
    @njit(parallel=True)
    def cal_nearest_graph(V, near_matrix, graph):
        # Your cal_nearest_graph implementation
        pass

    @staticmethod
    @njit(parallel=True)
    def cal_distance_matrix(V, graph, dist_matrix):
        # Your cal_distance_matrix implementation
        pass