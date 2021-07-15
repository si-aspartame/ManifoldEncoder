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
#plotly.offline.init_notebook_mode()
plotly.io.kaleido.scope.default_format = "png"
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# %%
#ハイパーパラメータ
n_samples = 70000
s_num = 3000
num_epochs = 50
learning_rate = 1e-4
early_stopping = 50
g_distance = torch.Tensor()
g_mse = torch.Tensor()
g_distance_list = []
g_mse_list = []
wd = 0.01
LAMBDA = 10


# %%
#実験用に乱数を固定
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# %%
#MNISTを読み込み
in_data, color = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./MNIST/')
in_data = preprocessing.MinMaxScaler().fit_transform(in_data)


# %%
#損失関数
def custom_loss(output, target, in_diff_sum, lat_diff_sum, out_diff_sum):
    global g_distance, g_mse
    KL_divergence = nn.KLDivLoss().cuda()
    SM = nn.Softmax(dim=0).cuda()
    MSE = nn.MSELoss().cuda()
    g_mse = MSE(output, target)
    # x2z = ((KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))+KL_divergence(SM(lat_diff_sum).log(), SM(in_diff_sum))) / 2)
    # z2y = ((KL_divergence(SM(lat_diff_sum).log(), SM(out_diff_sum))+KL_divergence(SM(out_diff_sum).log(), SM(lat_diff_sum))) / 2)
    x2z = torch.abs(torch.sum(lat_diff_sum / in_diff_sum) - BATCH_SIZE)
    z2y = torch.abs(torch.sum(lat_diff_sum / out_diff_sum) - BATCH_SIZE)
    g_distance = (x2z+z2y) / 2
    loss = g_mse + (LAMBDA*g_distance)
    return loss


# %%
#図の出力
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
#モデルを宣言
model = autoencoder().cuda()
criterion = custom_loss
optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=wd)
print(model)


# %%
#実験用にデータを小さくした場合バッチサイズと齟齬がないように削減
global in_ndarray, color, in_tensor
n_samples = n_samples - (n_samples%BATCH_SIZE)
in_ndarray = np.array(in_data)[:n_samples, :].astype(np.float32)
in_tensor = torch.from_numpy(in_ndarray)#in_ndarrayをテンソルにしたもの
color = color[:n_samples]
print(f"in_tensor.shape:{in_tensor.shape}")


# %%
#ランダムに抽出してCoRankingMatrixを計算
def get_crm_score(in_data, lat_result, n_sampling_iter = 70):
    s=0
    sampling_num = 1000
    G_cutoff = sampling_num
    G_error = G_cutoff / 10
    L_cutoff = 50
    L_error = L_cutoff / 10
    global_score = 0
    local_score = 0
    for n in range(n_sampling_iter):
        rnd_idx = [random.randint(0, len(in_data)-1) for i in range(sampling_num)]
        rnd_in_data = np.array([in_data[i] for i in rnd_idx])
        rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
        global_score += CRM(rnd_in_data).evaluate_crm(rnd_lat_result, G_cutoff, G_error) / n_sampling_iter
    print(f'GLOBAL_SCORE:{global_score}')
    for n in range(n_sampling_iter):
        rnd_idx = [random.randint(0, len(in_data)-1) for i in range(sampling_num)]
        rnd_in_data = np.array([in_data[i] for i in rnd_idx])
        rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
        local_score += CRM(rnd_in_data).evaluate_crm(rnd_lat_result, L_cutoff, L_error) / n_sampling_iter
    print(f'LOCAL_SCORE:{local_score}')
    return

#epochごとに潜在表現の予測結果を出力し、スコアを表示
# %%
def next_epoch(model, epoch, g_mse, g_distance, s_num=s_num):
    sampled_lat_result=np.empty((0, LATENT_DIMENSION))
    model.eval()
    s_num = s_num - (s_num%BATCH_SIZE)
    sampled_tensor = torch.from_numpy(np.array(in_data)[:s_num, :].astype(np.float32))
    sampled_color = color[:s_num]
    for n, data in enumerate(DataLoader(sampled_tensor, batch_size=BATCH_SIZE, shuffle=False)):#シャッフルしない
        batch = Variable(data.reshape(BATCH_SIZE, 1, 28, 28)).cuda()
        output, _, _, _, lat_repr = model(batch)
        sampled_lat_result = np.vstack([sampled_lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])
    file_name = f"{epoch}_{g_mse}_{g_distance}"
    if LATENT_DIMENSION <= 3: plot_latent(sampled_lat_result, sampled_color).update_layout(title=file_name).write_image(f"./lat/{file_name}.png")
    get_crm_score(sampled_tensor.cpu().numpy(), sampled_lat_result, n_sampling_iter = 10)
    return


# %%
#学習
all_loss=[]
best_loss=99999
es_count=0
start_time = time.time()
for epoch in range(1, num_epochs+1):
    temp_mse = 0
    temp_distance = 0
    temp_loss = 0
    model.train()
    data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    for data in data_iter:
        batch = Variable(data.reshape(BATCH_SIZE, 1, 28, 28)).cuda()
        output, in_diff_sum, lat_diff_sum, out_diff_sum, _ = model(batch)
        loss = criterion(output, batch, in_diff_sum, lat_diff_sum, out_diff_sum)#batch = 入力 = 教師データ
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        #損失関数の値はバッチごとに平均を取ってepochごとに表示
        temp_mse += g_mse.data.sum().item() / (n_samples / BATCH_SIZE)
        temp_distance += g_distance.data.sum().item() / (n_samples / BATCH_SIZE)
        temp_loss += loss.data.sum().item() / (n_samples / BATCH_SIZE)
    #損失関数の値が改善した場合はモデルを保存
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
    n = min(data.size(0), 8)
    #epochごとにreconstructionの結果を保存
    comparison = torch.cat([batch[:n], output.view(BATCH_SIZE, 1, 28, 28)[:n]])
    save_image(comparison.cpu(), 'rec/' + str(epoch) + '.png', nrow=n)
    if es_count == early_stopping or (temp_distance+temp_mse)==0.0:
        print('early stopping!')
        break


# %%
#最も損失関数の値が小さかったイテレーションのモデルを読み込んで全データを入力
best_iteration=np.argmin([x[1] for x in all_loss])
print(f'best_iteration:{all_loss[best_iteration]}')
best_model = autoencoder().cuda()
best_model.load_state_dict(torch.load(f'./output/{all_loss[best_iteration][0]}.pth'))
lat_result = np.empty((0, LATENT_DIMENSION))
best_model.eval()
for n, data in enumerate(DataLoader(in_tensor, batch_size = BATCH_SIZE, shuffle = False)):#シャッフルしない
    batch = Variable(data.reshape(BATCH_SIZE, 1, 28, 28)).cuda()
    output, _, _, _, lat_repr = best_model(batch)
    lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, LATENT_DIMENSION)])


# %%
#ここまでにかかった時間を算出し、CRMのスコアを計算する
elapsed_time = time.time() - start_time
print(f'elapsed_time:{elapsed_time}')
get_crm_score(in_tensor.cpu().numpy(), lat_result)


