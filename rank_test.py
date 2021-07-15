#%%
import numpy as np
from multiprocessing import Pool
import multiprocessing as multi
from sklearn import metrics
from scipy.stats import rankdata
import gc
from coranking import *


A = [[1, 1, 1],[2, 2, 2],[3, 3, 3],[4, 4, 4]]
B = [[1, 1],[3, 3],[2, 2],[4, 4]]

print(CRM(A).evaluate_crm(B, 2, 1))
