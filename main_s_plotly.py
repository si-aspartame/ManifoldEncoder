# %%
from IPython import get_ipython
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
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from maguro import *
from model import *
#次元落とさずに3次元でできないか(1次元にせずチャンネルを保持)

# %%
#learning parameter
num_epochs = 99999
learning_rate = 0.0001#1e-4
early_stopping=100
g_distance = torch.Tensor()
g_mse = torch.Tensor()
wd=0.00001
#{'target': -0.4959750175476074, 
# 'params': {'learning_rate': 4.9381810919946176e-05,
# 'p1': 0.2831057892577199,
# 'p2': 0.3855041854229957,
# 'wd': 3.914557171667492e-05}}
# %%
#swissroll parameter
n_samples = 32768#32768
noise = 0#0.05
sr, color = make_swiss_roll(n_samples, noise)#sr=swissroll
#動的にlossとdistanceに係数をかけて、どちらかに偏重しないようにする
#最初のイテレーションのlossとdistanceを1として、前回下がっていない方の損失を増やす
#どちらも出来ないことは許すが、どっちかが出来ることは罰する
def custom_loss(output, target, distance):
    global g_distance, g_mse
    g_mse = torch.mean((output - target)**2)
    g_distance = distance
    loss = (g_mse+g_distance)*((1+torch.abs(g_mse-g_distance))**2)
    return loss
#(m+d)*(1+(d-m))
#(m+d)-(m+d)(m-d)
#(m+d)-(m^2-d^2)
#
#

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
def plot_swissroll(sr, color):
    plotly.offline.init_notebook_mode()
    fig = go.Figure(data=[go.Scatter3d(x=sr[:, 0], y=sr[:, 1], z=sr[:, 2], mode='markers', marker=dict(size=1, color=color))])
    plotly.offline.iplot(fig, filename='jupyter-parametric_plot')
#         fig = plt.figure()#図の宣言
#         ax = fig.add_subplot(111, projection='3d')#三次元で[1,1,1]の位置にプロット
#         ax.scatter(sr[:, 0], sr[:, 1], sr[:, 2], c=color, cmap=plt.cm.Spectral)


# %%
np_sr = np.array(sr)
# plot_swissroll(sr, color)
np_sr, input_mean, input_std = z_score(np_sr)#zスコアで標準化

#%%
dimension=3#スイスロールは3次元のため一応明示的に書いておく
sr_min=np.amin(np_sr.reshape(n_samples*dimension), axis=0)#clampで標準化してから学習するが、
sr_max=np.amax(np_sr.reshape(n_samples*dimension), axis=0)#最終的に復元するときのために最大最小を保存
print(f'min:{sr_min}, max:{sr_max}')


# %%
model = autoencoder().cuda()
criterion = custom_loss#nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=wd)#weight_decay


# %%
in_tensor = torch.from_numpy(np_sr.astype(np.float32))#np_srをテンソルにしたもの
print(f"in_tensor:{in_tensor.size()}")


# %%
all_loss=[]
best_loss=99999
es_count=0
model.train()
for epoch in range(1, num_epochs+1):
    for data in DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True):
        batch = data
        batch = batch.reshape(batch.size(0)*3)#考える必要あり
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, distance = model(batch)
        loss = criterion(output, batch, distance)
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
    print(f'epoch [{epoch}/{num_epochs}], loss:{loss.data.item()}, \
        \n g_mse = {g_mse}, g_distance:{g_distance}'
        )
    all_loss.append(
        [epoch, loss.data.item(), g_mse.data.item(), \
        g_distance.data.item()]
        )
    if es_count == early_stopping:
        print('early stopping!')
        break#early_stopping

#%%
best_iteration=np.argmin([x[1] for x in all_loss])
print(f'best_iteration:{all_loss[best_iteration]}')

# %%
best_model = autoencoder().cuda()
best_model.load_state_dict(torch.load(f'./output/{all_loss[best_iteration][0]}.pth'))
result=np.empty((0,3))
model.eval()
for n, data in enumerate(DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=False)):#シャッフルしない
    print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(batch.size(0)*3)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output, _, _, = best_model(batch)
    result=np.vstack([result, output.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS)])
    
#plot_swissroll(reverse_z_score(result, input_mean, input_std), color)
plot_swissroll(result, color)


# %%



