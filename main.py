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
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import preprocessing
from sklearn.datasets import make_s_curve, make_swiss_roll, fetch_openml
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
from radam import *
from model import *
import time
from coranking import *
plotly.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
#MDSは遠いものが遠くにあることを優先するということ↓
#距離5を距離10と間違えた場合損失は5
#距離2を距離4と間違えた場合の損失は2
#比率は同じ（重要度は同じ）だが、結果的に上が優先される
#遠遠と近近どちらが有効かを評価して重要度を入れ替えられたらすごいよね


# %%
n_samples = 70000-(70000%BATCH_SIZE)#18000
num_epochs = 30
learning_rate = 1e-4
early_stopping = 50
g_distance = torch.Tensor()
g_mse = torch.Tensor()
g_distance_list = []
g_mse_list = []
wd = 0.0
sigma = 2#外れ値の除外に使う

#%%
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%q
in_data, color = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./MNIST/')
in_data = preprocessing.MinMaxScaler().fit_transform(in_data)
#in_data /= 255

#%%

def custom_loss(output, target, in_diff_sum, lat_diff_sum, out_diff_sum):
    global g_distance, g_mse
    KL_divergence = nn.KLDivLoss().cuda()#reduction="sum"
    SM = nn.Softmax(dim=0).cuda()
    MSE = nn.MSELoss(reduction='mean').cuda()
    g_mse = MSE(output, target)
    x2z = (0.5*(KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))+KL_divergence(SM(lat_diff_sum).log(), SM(in_diff_sum))))
    z2y = (0.5*(KL_divergence(SM(lat_diff_sum).log(), SM(out_diff_sum))+KL_divergence(SM(out_diff_sum).log(), SM(lat_diff_sum))))
    g_distance = (x2z+z2y) / 4
    #loss = g_distance
    loss = g_mse + g_distance
    return loss


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
model = autoencoder().cuda()
criterion = custom_loss
optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay
print(model)

# %%
global in_ndarray, color, in_tensor
in_ndarray = np.array(in_data)[:n_samples, :].astype(np.float32)
color = color[:n_samples]
in_tensor = torch.from_numpy(in_ndarray)#in_ndarrayをテンソルにしたもの
print(f"in_tensor.shape:{in_tensor.shape}")
#%%
def next_epoch(model, epoch, g_mse, g_distance):
    result=np.empty((0, INPUT_AXIS))
    lat_result=np.empty((0, LATENT_DIMENSION))
    model.eval()
    for n, data in enumerate(DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=False)):#シャッフルしない
        #print(f'TEST:{n}')
        batch = data
        batch = batch.reshape(BATCH_SIZE, 1, 28, 28)
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, _, _, _, lat_repr = model(batch)
        lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])
    sampling_num = 3000
    rnd_idx = [random.randint(0, len(in_tensor)-1) for i in range(sampling_num)]
    rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
    rnd_color = np.array([color[i] for i in rnd_idx])
    file_name = f"{epoch}_{g_mse}_{g_distance}"
    if LATENT_DIMENSION <= 3: plot_latent(rnd_lat_result, rnd_color).update_layout(title=file_name).write_image(f"./lat/{file_name}.png")
    return
# %%
all_loss=[]
best_loss=99999
es_count=0
frames = []
start_time = time.time()
for epoch in range(1, num_epochs+1):
    temp_mse = 0
    temp_distance = 0
    temp_loss = 0
    model.train()
    data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True)
    for data in data_iter:
        batch = data
        #print(f'batch.shape:{batch.shape}')
        batch = batch.reshape(BATCH_SIZE, 1, 28, 28)
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, in_diff_sum, lat_diff_sum, out_diff_sum, _ = model(batch)
        loss = criterion(output, batch, in_diff_sum, lat_diff_sum, out_diff_sum)#batch = 入力 = 教師データ
        #print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        temp_mse += g_mse.data.sum().item() / (n_samples / BATCH_SIZE)
        temp_distance += g_distance.data.sum().item() / (n_samples / BATCH_SIZE)
        temp_loss += loss.data.sum().item() / (n_samples / BATCH_SIZE)
    if temp_loss < best_loss:
        print('[BEST] ', end='')
        torch.save(model.state_dict(), f'./output/{epoch}.pth')
        best_loss = temp_loss
        es_count = 0
    es_count += 1

    print(f'epoch [{epoch}/{num_epochs}], loss:{temp_loss}, \n g_mse = {temp_mse}, g_distance:{temp_distance}')
    all_loss.append([epoch, temp_loss, temp_mse, temp_distance])
    g_mse_list.append(temp_mse)
    g_distance_list.append(temp_distance)
    next_epoch(model, epoch, temp_mse, temp_distance)
    if es_count == early_stopping or (temp_distance+temp_mse)==0.0:
        print('early stopping!')
        break#early_stopping

# %%
best_iteration=np.argmin([x[1] for x in all_loss])
print(f'best_iteration:{all_loss[best_iteration]}')
# %%
best_model = autoencoder().cuda()
best_model.load_state_dict(torch.load(f'./output/{all_loss[best_iteration][0]}.pth'))
lat_result = np.empty((0, LATENT_DIMENSION))
best_model.eval()
for n, data in enumerate(DataLoader(in_tensor, batch_size = BATCH_SIZE, shuffle = False)):#シャッフルしない
    #print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(BATCH_SIZE, 1, 28, 28)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output, _, _, _, lat_repr = best_model(batch)
    lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])
#%%
elapsed_time = time.time() - start_time
print(f'elapsed_time:{elapsed_time}')

#%%
print(len(in_ndarray[0]))#in_ndarrayはn_samples個
print(len(lat_result[0]))
# %%
# sampling_num = 1000
# rnd_idx = [random.randint(0, len(in_tensor)-1) for i in range(sampling_num)]
# rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
# rnd_color = np.array([color[i] for i in rnd_idx])
# if LATENT_DIMENSION <= 3:
#     plotly.offline.iplot(plot_latent(rnd_lat_result, rnd_color), filename='latent representation')


# %%
# loss_fig = go.Figure()
# num=3#移動平均の個数
# b=np.ones(num)/num
# MA_g_mse_list=np.convolve(g_mse_list, b, mode='same')#移動平均
# MA_g_distance_list=np.convolve(g_distance_list, b, mode='same')#移動平均
# loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_mse_list, name='mse'))
# loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_distance_list, name='distance'))
# plotly.offline.iplot(loss_fig, filename='mse and distance progress')

# %%
s=0
sampling_num = 1000
n_sampling_iter = 20
global_score = 0
local_score = 0
print(f'TIME:{elapsed_time}')
for n in range(n_sampling_iter):
    rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
    rnd_in_data = np.array([in_data[i] for i in rnd_idx])
    rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
    global_score += CoRanking(rnd_in_data).evaluate_corank_matrix(rnd_lat_result, sampling_num, 100) / n_sampling_iter
print(f'GLOBAL_SCORE:{global_score}')
for n in range(n_sampling_iter):
    rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
    rnd_in_data = np.array([in_data[i] for i in rnd_idx])
    rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
    local_score += CoRanking(rnd_in_data).evaluate_corank_matrix(rnd_lat_result, 50, 10) / n_sampling_iter
print(f'LOCAL_SCORE:{local_score}')

#%%