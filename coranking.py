import numpy as np
from multiprocessing import Pool
import multiprocessing as multi
from sklearn import metrics
from scipy.stats import rankdata
import gc

def get_rank(vector):
    return rankdata(vector, method='ordinal')
    
class CRM(object):
    def __init__(self, input_rho_data):
        self.input_rho_data = input_rho_data
        self.metric = 'euclidean'
        self.input_rho_rank = self.calc(input_rho_data)

    def calc(self, input_vector):
        distance_matrix = metrics.pairwise.pairwise_distances(input_vector, input_vector, metric=self.metric)
        rank_matrix = np.array(list(map(get_rank, distance_matrix)))
        return rank_matrix

    def evaluate_crm(self,  reduced_r_data, ks, kt):
        #print(f'reduced_r_rank{reduced_r_data}')
        self.reduced_r_rank = self.calc(reduced_r_data)
        difference_rank = np.abs(self.input_rho_rank - self.reduced_r_rank)
        remain_kt = difference_rank < kt
        remain_ks = np.array([self.input_rho_rank < ks, self.input_rho_rank < ks]).any(axis=0)
        return np.sum(np.array([remain_kt, remain_ks]).all(axis=0)) / (ks * len(reduced_r_data))