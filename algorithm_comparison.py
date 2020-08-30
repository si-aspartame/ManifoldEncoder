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
import umap
import gc
import random
get_ipython().run_line_magic('matplotlib', 'inline')
convex=False
rotate=False
n_samples = 3**9
sns.set(context="paper", style="white")
sr, swissroll_labels = datasets.make_swiss_roll(n_samples=n_samples, noise=0)
print('a')
np_x = np.array(sr)
if convex:
    np_x = np.array([[x[0], x[1]+(c*2), x[2]] for c, x in zip(swissroll_labels, list(np_x))])
if rotate:
    np_x = np.array([[x[0]+x[0], (x[2]-x[1]), (x[1]+x[2])] for x in list(np_x)])
# np_x[:, 1] = np_x[:, 1] / 2
swissroll = np_x
print('b')
fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(211, projection='3d')
ax.scatter(swissroll[:, 0], swissroll[:, 1], swissroll[:, 2], c=swissroll_labels, cmap="coolwarm")
plt.show()


# %%
reducers = [
    (decomposition.PCA, {"iterated_power": 1000}),
    (umap.UMAP, {"n_neighbors": 30, "min_dist": 0.3}),
    (manifold.TSNE, {"perplexity": 50}),
    (manifold.MDS, {}),
]

test_data = [
    (swissroll, swissroll_labels),
]
dataset_names = ["Swiss Roll"]

n_rows = len(test_data)
n_cols = len(reducers)


# %%
def plot_swissroll(sr, color, dim):
    if dim == 3:
        plotly.offline.init_notebook_mode()
        fig = go.Figure(data=[go.Scatter3d(x=sr[:, 0], y=sr[:, 1], z=sr[:, 2], mode='markers', marker=dict(size=1, color=color, colorscale="blugrn"))])
    elif dim == 2:
        plotly.offline.init_notebook_mode()
        fig = go.Figure(data=[go.Scatter(x=sr[:, 0], y=sr[:, 1], mode='markers', marker=dict(size=5, color=color, colorscale="blugrn"))])
    fig.update_layout(yaxis=dict(scaleanchor='x'))#縦横比を1:1に
    return fig


# %%
strings=['PCA','UMAP', 'TSNE', 'MDS']
s=0
sampling_num = 1000
n_sampling_iter=20
for np_x, labels in test_data:
    for reducer, args in reducers:
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
        plot_swissroll(result, swissroll_labels, 2).update_layout(title=fn).write_image(f"./comparision/{strings[s]}.png")
        s+=1


# %%



