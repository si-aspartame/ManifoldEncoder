# %%
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
from sklearn.datasets import make_s_curve, make_swiss_roll, fetch_openml
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
from radam import *
from model import *
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
n_samples = 69990#15000#
num_epochs = 30
learning_rate = 1e-3
early_stopping = 50
g_distance = torch.Tensor()
g_mse = torch.Tensor()
g_distance_list = []
g_mse_list = []
wd = 0.0#000001#00001
sigma = 2#外れ値の除外に使う

#%%
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# %%
in_data, color = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./MNIST/')
# in_data = (-1*in_data)+255
in_data /= 255

#%%
def custom_loss(output, target, in_diff_sum, lat_diff_sum):
    global g_distance, g_mse
    KL_divergence = nn.KLDivLoss().cuda()#reduction="sum"
    SM = nn.Softmax(dim=0).cuda()
    MSE = nn.MSELoss(reduction='mean').cuda()
    g_mse = MSE(output, target)
    g_distance = KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))
    loss = g_mse + ((1e+05)*g_distance)
    return loss


# %%
def plot_latent(in_data, color):
    if LATENT_DIMENSION == 2:
        df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Labels':color}).sort_values('Labels')
        fig = px.scatter(df, x='X', y='Y', color='Labels', color_discrete_sequence=px.colors.qualitative.D3)
        fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=True)#縦横比を1:1に
    if LATENT_DIMENSION == 3:
        df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Z':in_data[:, 2], 'Labels':color}).sort_values('Labels')
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size=np.repeat(10, len(in_data)), size_max=10, opacity=1.0)
        fig.update_layout(showlegend=True)#縦横比を1:1に
    return fig



# %%
model = autoencoder().cuda()
criterion = custom_loss#nn.MSELoss()
optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay
#optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
#scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)


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
        output, _, _, lat_repr = model(batch)
        lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])
    sampling_num = 3000
    rnd_idx = [random.randint(0, len(in_tensor)-1) for i in range(sampling_num)]
    rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
    rnd_color = np.array([color[i] for i in rnd_idx])
    file_name = f"{epoch}_{g_mse}_{g_distance}"
    if LATENT_DIMENSION <= 3: plot_latent(rnd_lat_result, rnd_color).update_layout(title=file_name).write_image(f"./lat/{file_name}.png")
    #new_in_tensor, new_color = exclude_outlier(lat_result)
    return #new_in_tensor, new_color

def exclude_outlier(encoded_data, sigma=sigma):
    #print(encoded_data)
    means = np.array([np.mean(encoded_data[:, x]) for x in range(LATENT_DIMENSION)])
    stds = np.array([np.std(encoded_data[:, x]) for x in range(LATENT_DIMENSION)])
    uppers = np.array([m+(s*sigma) for m, s in zip(means, stds)])
    lowers = np.array([m-(s*sigma) for m, s in zip(means, stds)])
    outlier_flags = [np.where((uppers[x] < encoded_data[:, x]) | (encoded_data[:, x] < lowers[x]))[0].tolist() for x in range(LATENT_DIMENSION)]#[outflag_x, outflag_y ....]

    if LATENT_DIMENSION == 2:
        duplicated_index = set(outlier_flags[0]) & set(outlier_flags[1])#全次元が外れ値になっている行のindex一覧
    elif LATENT_DIMENSION ==3:
        duplicated_index = set(outlier_flags[0]) & set(outlier_flags[1]) & set(outlier_flags[2])
    else:
        print('LATENT_DIMENSION is too big')
        exit()

    #print(f'外れ値:{list(duplicated_index)}')
    new_color = np.delete(color, list(duplicated_index), axis=0)
    new_in_tensor = torch.from_numpy(np.delete(in_ndarray, list(duplicated_index), axis=0))
    return new_in_tensor, new_color


# %%
all_loss=[]
best_loss=99999
es_count=0
frames = []
for epoch in range(1, num_epochs+1):
    #BATCH_SIZEでデータ数を割り切れない時は余りを引く
    print(in_tensor.shape)
    if not (in_tensor.shape[0] % BATCH_SIZE) == 0:########できれば乖離している順に無駄なく引いた方が良い
        in_ndarray = in_ndarray[:in_tensor.shape[0]-(in_tensor.shape[0] % BATCH_SIZE)]
        color = color[:in_tensor.shape[0]-(in_tensor.shape[0] % BATCH_SIZE)]
        in_tensor = in_tensor[:in_tensor.shape[0]-(in_tensor.shape[0] % BATCH_SIZE), :]
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
        output, in_diff_sum, lat_diff_sum, _ = model(batch)
        loss = criterion(output, batch, in_diff_sum, lat_diff_sum)#batch = 入力 = 教師データ
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

    #in_tensor, color = next_epoch(model, epoch, temp_mse, temp_distance)
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
    output, _, _, lat_repr = best_model(batch)
    lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])


# %%
sampling_num = 1000
rnd_idx = [random.randint(0, len(in_tensor)-1) for i in range(sampling_num)]
rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
rnd_color = np.array([color[i] for i in rnd_idx])

# %%
if LATENT_DIMENSION <= 3:
    plotly.offline.iplot(plot_latent(rnd_lat_result, rnd_color, LATENT_DIMENSION), filename='latent representation')


# %%
loss_fig = go.Figure()

num=3#移動平均の個数
b=np.ones(num)/num
MA_g_mse_list=np.convolve(g_mse_list, b, mode='same')#移動平均
MA_g_distance_list=np.convolve(g_distance_list, b, mode='same')#移動平均
loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_mse_list, name='mse'))
loss_fig.add_trace(go.Scatter(x = list(range(0, len(all_loss))), y = MA_g_distance_list, name='distance'))
plotly.offline.iplot(loss_fig, filename='mse and distance progress')

# %%
