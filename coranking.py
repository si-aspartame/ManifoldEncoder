import numpy as np
from multiprocessing import Pool
import multiprocessing as multi
from sklearn import metrics
from scipy.stats import rankdata
import gc
def _parallel_calculate_rank(vector):
    return rankdata(vector, method='ordinal')

class CoRanking(object):
    def __init__(self, original_data, metric='euclidean'):
        self.original_data = original_data
        self.metric = metric
        self.original_rank = self._calculate_rank_matrix(original_data)

    def _calculate_rank_matrix(self, data):
        # calculate distance matrix
        distance_matrix = metrics.pairwise.pairwise_distances(data, data, metric=self.metric)

        # calculate rank matrix
        rank_matrix = np.array(list(map(_parallel_calculate_rank, distance_matrix)))
        return np.array(rank_matrix)

    def evaluate_corank_matrix(self,  dr_data, significance=96, tolerance=70):
        # calculate rank matrix for dimensionality reduction data
        self.dr_rank = self._calculate_rank_matrix(dr_data)

        # reduct to only tolerance data
        difference_rank = np.absolute(self.original_rank - self.dr_rank)
        remain_tolerance = difference_rank < tolerance

        # reduct to only siginifcance data
        remain_significance = np.array([self.original_rank < significance, self.original_rank < significance]).any(axis=0)

        # reduct both
        result = np.sum(np.array([remain_tolerance, remain_significance]).all(axis=0)) / (significance * len(dr_data))
        return result

    def multi_evaluate_corank_matrix(self, dr_data, significance_range = range(10, 96),
                                     tolerance_range = range(10, 70)):
        # calculate rank matrix for dimensionality reduction data
        self.dr_rank = self._calculate_rank_matrix(dr_data)

        # calculate difference between rank matrices
        difference_rank = np.absolute(self.original_rank - self.dr_rank)

        # result
        result = np.zeros((len(significance_range), len(tolerance_range)))

        for i, significance in enumerate(significance_range):
            if i % (len(significance_range) / 10) == 0:
                print('*', end='')
            for j, tolerance in enumerate(tolerance_range):
                print(j)
                # reduct to only tolerance data
                remain_tolerance = difference_rank < tolerance

                # reduct to only siginifcance data
                remain_significance = np.array([self.original_rank < significance, self.original_rank < significance]).any(axis=0)

                # reduct both
                result[i, j] = np.sum(np.array([remain_tolerance, remain_significance]).all(axis=0)) / (significance * len(dr_data))
        return result