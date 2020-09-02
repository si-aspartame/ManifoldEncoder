# %%
from IPython import get_ipython
from IPython.display import Image
import os
import gc
import numpy as np
import random
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.datasets import make_s_curve, make_swiss_roll
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
from radam import *
from model import *
import time
from coranking import *

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#MDSは遠いものが遠くにあることを優先するということ↓
#距離5を距離10と間違えた場合損失は5
#距離2を距離4と間違えた場合の損失は2
#比率は同じ（重要度は同じ）だが、結果的に上が優先される
#遠遠と近近どちらが有効かを評価して重要度を入れ替えられたらすごいよね


# %%
mode = 'roll'
convex = False
rotate = False
num_epochs = 1
learning_rate = 0.01#0.01
early_stopping = 50
g_distance = torch.Tensor()
g_mse = torch.Tensor()
g_distance_list = []
g_mse_list = []
wd = 0.0#0.00001
#Softmaxでスケールの情報が消えるから正則化が効く
#四角に発散するやつはスケールぐちゃぐちゃ


# %%
#swissroll parameter
n_samples = 4**7#3**9#
noise = 0.1#0.05^\
if mode == 'curve':
    sr, color = make_s_curve(n_samples, noise)
elif mode == 'roll':
    sr, color = make_swiss_roll(n_samples, noise)


# %%
def custom_loss(output, target, in_diff_sum, lat_diff_sum):
    global g_distance, g_mse
    KL_divergence = nn.KLDivLoss().cuda()#reduction="sum"
    SM = nn.Softmax(dim=0).cuda()
    MSE = nn.MSELoss(reduction='mean').cuda()
    g_mse = MSE(output, target)
    #distance = KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))
    g_distance = KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))
    #loss = g_mse+(g_distance/g_mse)
    loss = g_mse+(20*g_distance)
    #loss = ((1/20)*g_mse)+g_distance
    return loss


# %%
def scaling(x, n_samples, dimension, axis = None):
    # xmean = x.mean(axis=axis, keepdims=True)
    # xstd = np.std(x, axis=axis, keepdims=True)
    # zscore = (x-xmean)/xstd
    x_min = np.amin(x.reshape(n_samples*dimension), axis=0)
    x_max = np.amax(x.reshape(n_samples*dimension), axis=0)
    result = (x-x_min)/(x_max-x_min)
    return (result*2) - 1, x_max, x_min


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
def do_plot(model, epoch, g_mse, g_distance):
    result=np.empty((0,3))
    lat_result=np.empty((0,2))
    model.eval()
    for n, data in enumerate(DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=False)):#シャッフルしない
        #print(f'TEST:{n}')
        batch = data
        batch = batch.reshape(batch.size(0)*3)
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, _, _, lat_repr = model(batch)
        result=np.vstack([result, output.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS)])
        lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS-1)])
    file_name = f"{epoch}_{g_mse}_{g_distance}"
    plot_swissroll(result, color, 3).update_layout(title=file_name).write_image(f"./result/{file_name}.png")
    plot_swissroll(lat_result, color, 2).update_layout(title=file_name).write_image(f"./lat/{file_name}.png")

# %%
np_x = np.array(sr)
#%%
if convex:
    np_x = np.array([[x[0], x[1]+(c*5), x[2]] for c, x in zip(color, list(np_x))])
if rotate:
    np_x = np.array([[x[0]+x[0], (x[2]-x[1]), (x[1]+x[2])] for x in list(np_x)])
np_x, input_max, input_min = scaling(np_x, n_samples, INPUT_AXIS)#min-max正規化
# plot_swissroll(np_x, color, 3).update_layout(title=f"Original")

# %%
print(f'max:{input_max}, min:{input_min}')

#%%
ae_start_time = time.time()

# %%
model = autoencoder().cuda()
criterion = custom_loss#nn.MSELoss()
#optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
#scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)

# %%
in_tensor = torch.from_numpy(np_x.astype(np.float32))#np_xをテンソルにしたもの
print(f"in_tensor:{in_tensor.size()}")


# %%
all_loss=[]
best_loss=99999
es_count=0
frames = []
for epoch in range(1, num_epochs+1):
    temp_mse = 0
    temp_distance = 0
    temp_loss = 0
    model.train()
    data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True)
    for data in data_iter:
        batch = data
        batch = batch.reshape(batch.size(0)*3)#考える必要あり
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, in_diff_sum, lat_diff_sum, _ = model(batch)
        loss = criterion(output, batch, in_diff_sum, lat_diff_sum)#batch = 入力 = 教師データ
        #print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        #loss.sum().backward()
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
    all_loss.append(
        [epoch, temp_loss, temp_mse, temp_distance]
        )
    g_mse_list.append(temp_mse)
    g_distance_list.append(temp_distance)
    do_plot(model, epoch, temp_mse, temp_distance)
    if es_count == early_stopping or (temp_distance+temp_mse)==0.0:
        print('early stopping!')
        break#early_stopping


# %%
best_iteration=np.argmin([x[1] for x in all_loss])
print(f'best_iteration:{all_loss[best_iteration]}')


# %%
best_model = autoencoder().cuda()
best_model.load_state_dict(torch.load(f'./output/{all_loss[best_iteration][0]}.pth'))
result = np.empty((0,3))
lat_result = np.empty((0,2))
best_model.eval()
for n, data in enumerate(DataLoader(in_tensor, batch_size = BATCH_SIZE, shuffle = False)):#シャッフルしない
    #print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(batch.size(0)*3)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output, _, _, lat_repr = best_model(batch)
    result=np.vstack([result, output.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS)])
    lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS-1)])
ae_end_time = time.time()
ae_total_time = ae_end_time - ae_start_time


# %%
print(ae_total_time)


# %%
sampling_num = 1000
n_sampling_iter = 20
cr_score = 0
global_score = 0
local_score = 0
for n in range(n_sampling_iter):
    print(f'calculation mean corank:{n}')
    rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
    rnd_np_x = np.array([np_x[i] for i in rnd_idx])
    rnd_result = np.array([result[i] for i in rnd_idx])
    global_score += CoRanking(rnd_np_x).evaluate_corank_matrix(rnd_result, sampling_num, 100) / n_sampling_iter
print(f'GLOBAL_SCORE:{global_score}')
for n in range(n_sampling_iter):
    print(f'calculation mean corank:{n}')
    rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
    rnd_np_x = np.array([np_x[i] for i in rnd_idx])
    rnd_result = np.array([result[i] for i in rnd_idx])
    local_score += CoRanking(rnd_np_x).evaluate_corank_matrix(rnd_result, 50, 10) / n_sampling_iter
print(f'LOCAL_SCORE:{local_score}')

rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
rnd_color = np.array([color[i] for i in rnd_idx])


# %%
# matrix = cr.multi_evaluate_corank_matrix(result, range(2, int(n_samples/10)), range(2, 10))
# fig = go.Figure(data=go.Heatmap(z=matrix))
# fig.show()


# %%
plotly.offline.iplot(plot_swissroll(rnd_result, rnd_color, 3), filename='decoded swiss roll')


# %%
plotly.offline.iplot(plot_swissroll(rnd_lat_result, rnd_color, 2), filename='latent representation')


# %%
loss_fig = go.Figure()


# %%
num=3#移動平均の個数
b=np.ones(num)/num
MA_g_mse_list=np.convolve(g_mse_list, b, mode='same')#移動平均
MA_g_distance_list=np.convolve(g_distance_list, b, mode='same')#移動平均
loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_mse_list, name='mse'))
loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_distance_list, name='distance'))
plotly.offline.iplot(loss_fig, filename='mse and distance progress')


# %%



