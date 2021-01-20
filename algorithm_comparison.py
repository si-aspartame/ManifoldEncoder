# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from sklearn import datasets, decomposition, manifold, preprocessing
from sklearn.datasets import make_s_curve, make_swiss_roll
from colorsys import hsv_to_rgb
from coranking import *
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
#%%
import umap
import gc
import random
from model import *
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
mode = 'mnist'
n_samples = 4**7
noise = 0
if mode == 'curve':
    in_data, color = make_s_curve(n_samples, noise)
elif mode == 'roll':
    in_data, color = make_swiss_roll(n_samples, noise)
elif mode == 'mnist':
    in_data, color = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./MNIST/')
sns.set(context="paper", style="white")
in_data = preprocessing.MinMaxScaler().fit_transform(in_data)
# %%
reducers = [
    (decomposition.PCA, {}),#"iterated_power": 1000
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
    (manifold.TSNE, {"perplexity": 50}),
    (manifold.MDS, {}),
]

test_data = [
    (in_data, color),
]
dataset_names = [mode]

n_rows = len(test_data)
n_cols = len(reducers)
n_samples = len(in_data)
print(n_rows, n_cols, n_samples)

# %%
def plot_latent(in_data, color):
    df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Labels':color}).sort_values('Labels')
    if not mode=='mnist':
        df['Labels'] = pd.qcut(df['Labels'], 10).astype(str)
    fig = px.scatter(df, x='X', y='Y', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size_max=5, opacity=0.5)
    fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=False)#縦横比を1:1に
    return fig


# %%
strings=['PCA','UMAP', 'TSNE', 'MDS']
s=0
sampling_num = 1000
n_sampling_iter=70
for np_x, labels in test_data:
    for reducer, args in reducers:
        if strings[s]=='MDS' and mode=='mnist':
            break
        global_score = 0
        local_score = 0
        start_time = time.time()
        result = np.array(reducer(n_components=2, **args).fit_transform(np_x))
        elapsed_time = time.time() - start_time
        print(f'TIME:{str(reducer).split(".")[2]}:{elapsed_time}')
        for n in range(n_sampling_iter):
            rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
            rnd_np_x = np.array([np_x[i] for i in rnd_idx])
            rnd_result = np.array([result[i] for i in rnd_idx])
            global_score += CoRanking(rnd_np_x).evaluate_corank_matrix(rnd_result, sampling_num, 100) / n_sampling_iter
        print(f'GLOBAL_SCORE:{str(reducer).split(".")[2]}:{global_score}')
        for n in range(n_sampling_iter):
            rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
            rnd_np_x = np.array([np_x[i] for i in rnd_idx])
            rnd_result = np.array([result[i] for i in rnd_idx])
            local_score += CoRanking(rnd_np_x).evaluate_corank_matrix(rnd_result, 50, 5) / n_sampling_iter
        print(f'LOCAL_SCORE:{str(reducer).split(".")[2]}:{local_score}')
        fn = f'{strings[s]}_{elapsed_time}'
        plot_num = 3000
        rnd_idx = [random.randint(0, n_samples-1) for i in range(plot_num)]
        rnd_np_x = np.array([np_x[i] for i in rnd_idx])
        rnd_result = np.array([result[i] for i in rnd_idx])
        rnd_color = np.array([color[i] for i in rnd_idx])
        plot_latent(rnd_result, rnd_color).update_layout(title=f"{strings[s]}_{mode}").write_image(f"./comparision/{strings[s]}_{mode}.png")
        s += 1
# %%



