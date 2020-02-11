# %%
from IPython import get_ipython
from IPython.display import Image
import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.datasets import make_swiss_roll
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly
from bayes_opt import BayesianOptimization
from radam import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from maguro import *
from model import *

# %%
#learning parameter
num_epochs = 1
learning_rate = 0.0001#1e-4
early_stopping = 50
g_distance = torch.Tensor()
g_mse = torch.Tensor()
g_distance_list = []
g_mse_list = []
wd=0.00005

# %%
#swissroll parameter
n_samples = 32768
noise = 0.05#0.05
sr, color = make_swiss_roll(n_samples, noise)#sr=swissroll
#動的にlossとdistanceに係数をかけて、どちらかに偏重しないようにする
#最初のイテレーションのlossとdistanceを1として、前回下がっていない方の損失を増やす
#どちらも出来ないことは許すが、どっちかが出来ることは罰する
def custom_loss(output, target, in_diff_sum, lat_diff_sum):
    global g_distance, g_mse
    
    g_mse = torch.mean((output - target)**2)
    KL_divergence = nn.KLDivLoss(reduction="sum")
    SM = nn.Softmax(dim=0)
    #全組み合わせの距離について、入力・潜在表現間のダイバージェンスを求める
    #その後、BATCH_SIZE個の全ダイバージェンスについて平均を取る
    g_distance = 0
    # for n in range(BATCH_SIZE):
    #     #BATCH_SIZE*BATCH_SIZEの組み合わせに対して一次元ユークリッド距離を取り、分布を求める
    #     in_euclid_comb = [torch.abs((in_diff_sum[n] - in_diff_sum[m])) for m in range(BATCH_SIZE)]
    #     in_distance = torch.stack(in_euclid_comb, dim=0)
    #     lat_euclid_comb = [torch.abs((lat_diff_sum[n] - lat_diff_sum[m])) for m in range(BATCH_SIZE)]
    #     lat_distance = torch.stack(lat_euclid_comb, dim=0)
    #     g_distance += KL_divergence(SM(in_distance).log(), SM(lat_distance)) / BATCH_SIZE
    g_distance = KL_divergence(SM(in_diff_sum).log(), SM(lat_diff_sum))
    loss = (g_mse+g_distance)*(1+torch.abs(g_mse-g_distance))
    return loss


# %%
def z_score(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore, xmean, xstd

def reverse_z_score(x, input_mean, input_std, axis = None):
    reverse_zscore = (x+input_mean)*input_std
    return reverse_zscore


# %%
def plot_swissroll(sr, color, dim):
    if dim == 3:
        plotly.offline.init_notebook_mode()
        fig = go.Figure(data=[go.Scatter3d(x=sr[:, 0], y=sr[:, 1], z=sr[:, 2], mode='markers', marker=dict(size=1, color=color,))])# colorscale="burg"))])
    elif dim == 2:
        plotly.offline.init_notebook_mode()
        fig = go.Figure(data=[go.Scatter(x=sr[:, 0], y=sr[:, 1], mode='markers', marker=dict(size=5, color=color))])#, colorscale="burg"))])
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
np_sr = np.array(sr)
# sr[:, 0] = sr[:, 0] * 4
# sr[:, 2] = sr[:, 2] * 4
plot_swissroll(sr, color, 3).update_layout(title=f"Original").write_image(f"./result/{0}.png")
np_sr, input_mean, input_std = z_score(np_sr)#zスコアで標準化


# %%
dimension=3#スイスロールは3次元のため一応明示的に書いておく
sr_min=np.amin(np_sr.reshape(n_samples*dimension), axis=0)#clampで標準化してから学習するが、
sr_max=np.amax(np_sr.reshape(n_samples*dimension), axis=0)#最終的に復元するときのために最大最小を保存
print(f'min:{sr_min}, max:{sr_max}')


# %%
model = autoencoder().cuda()
criterion = custom_loss#nn.MSELoss()
optimizer = RAdam(
    model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay


# %%
in_tensor = torch.from_numpy(np_sr.astype(np.float32))#np_srをテンソルにしたもの
print(f"in_tensor:{in_tensor.size()}")


# %%
all_loss=[]
best_loss=99999
es_count=0
frames = []
for epoch in range(1, num_epochs+1):
    model.train()
    for data in DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True):
        batch = data
        batch = batch.reshape(batch.size(0)*3)#考える必要あり
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, in_diff_sum, lat_diff_sum, _ = model(batch)
        loss = criterion(output, batch, in_diff_sum, lat_diff_sum)
        #print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if loss.data.item() < best_loss:
        print('[BEST] ', end='')
        torch.save(model.state_dict(), f'./output/{epoch}.pth')
        best_loss = loss.data.item()
        es_count = 0
    es_count += 1
    print(f'epoch [{epoch}/{num_epochs}], loss:{loss.data.item()},         \n g_mse = {g_mse}, g_distance:{g_distance}'
        )
    all_loss.append(
        [epoch, loss.data.item(), g_mse.data.item(), g_distance.data.item()]
        )
    g_mse_list.append(g_mse.data.item())
    g_distance_list.append(g_distance.data.item())
    do_plot(model, epoch, g_mse, g_distance)
    #######################################################################ここに点の密集度の罰則をつける
    if es_count == early_stopping:
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
    print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(batch.size(0)*3)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output, _, _, lat_repr = best_model(batch)
    result=np.vstack([result, output.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS)])
    lat_result=np.vstack([lat_result, lat_repr.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS-1)])


# %%
plotly.offline.iplot(plot_swissroll(result, color, 3), filename='decoded swiss roll')


# %%
plotly.offline.iplot(plot_swissroll(lat_result, color, 2), filename='latent representation')


# %%
loss_fig = go.Figure()

num=1#移動平均の個数
b=np.ones(num)/num
MA_g_mse_list=np.convolve(g_mse_list, b, mode='same')#移動平均
MA_g_distance_list=np.convolve(g_distance_list, b, mode='same')#移動平均
loss_fig.add_trace(go.Scatter(x = list(range(0, es_count)), y = MA_g_mse_list, name='mse'))
loss_fig.add_trace(go.Scatter(x = list(range(0, es_count)), y = MA_g_distance_list, name='distance'))
plotly.offline.iplot(loss_fig, filename='mse and distance progress')


