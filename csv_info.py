#%%
from IPython import get_ipython
from IPython.display import Image
import os
import gc
import numpy as np
import random
import torch
import torchvision
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from radam import *
from linear_model_cosine import *
import time
from coranking import *
from torchvision.utils import save_image
import matplotlib.pyplot as plt

#plotly.offline.init_notebook_mode()
plotly.io.kaleido.scope.default_format = "png"
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# %%
#ハイパーパラメータ
n_samples = 83804
# %%
#実験用に乱数を固定
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# %%
df = pd.read_csv('./data/CSV/ofm_struct_gap.csv').dropna(how='any', axis=0)
color = df[df.columns[1]].values
in_data = df.drop(df.columns[[0, 1]], axis=1).values

#%%
print(len(in_data), len(color))
print(len(in_data[0]))
print(color[0])
#%%
# in_data = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(in_data)
# color = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(color.reshape(-1, 1)).reshape(-1)

#%%
from pylab import rcParams
rcParams['figure.figsize'] = 15, 15
df.hist()

#%%
df = df.drop(df.columns[[0, 1]], axis=1)

# %%
df.duplicated()

# %%
df.describe()
# %%
for n, _ in enumerate(df.columns):
    if n+1 == 58:
        break
    ax = df[df.columns[n]].hist()
    df[df.columns[n+1]].hist()
    fig = ax.get_figure()
    plt.legend(f'{n}', f'{n+1}')
    fig.savefig(f'result/{n}_{n+1}.png')
    plt.cla()
    plt.clf()

# %%
