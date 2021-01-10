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
from colorsys import hsv_to_rgb
from coranking import *
import pandas as pd
import plotly.express as px
#%%
import umap
import gc
import random
from model import *
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
in_data, color = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./MNIST/')
in_data = in_data-128
in_data /= 255
sns.set(context="paper", style="white")

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
dataset_names = ["MNIST"]

n_rows = len(test_data)
n_cols = len(reducers)
n_samples = len(in_data)
print(n_rows, n_cols, n_samples)

# %%
def plot_latent(in_data, color):
    if LATENT_DIMENSION == 2:
        df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Labels':color}).sort_values('Labels')
        fig = px.scatter(df, x='X', y='Y', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size_max=5, opacity=0.5)
        fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=True)#縦横比を1:1に
    if LATENT_DIMENSION == 3:
        df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Z':in_data[:, 2], 'Labels':color}).sort_values('Labels')
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size=np.repeat(10, len(in_data)), size_max=5, opacity=0.5)
        fig.update_layout(showlegend=True)#縦横比を1:1に
    return fig


# %%
strings=['PCA','UMAP', 'TSNE', 'MDS']
s=0
sampling_num = 1000
n_sampling_iter=20
for np_x, labels in test_data:
    for reducer, args in reducers:
        if strings[s]=='MDS':
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
            local_score += CoRanking(rnd_np_x).evaluate_corank_matrix(rnd_result, 50, 10) / n_sampling_iter
        print(f'LOCAL_SCORE:{str(reducer).split(".")[2]}:{local_score}')
        fn = f'{strings[s]}_{elapsed_time}'
        plot_latent(result, color).update_layout(title=fn).write_image(f"./comparision/{strings[s]}.png")
        s += 1

# %%



