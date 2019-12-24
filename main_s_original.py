#%%
import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



%matplotlib inline
%load_ext autoreload
%autoreload 2

from model import *
#%%
#learning parameter
num_epochs = 50
learning_rate = 1e-5

#%%
#swissroll parameter
n_samples = 25600
noise = 0.05#0.05
sr, color = make_swiss_roll(n_samples, noise)#sr=swissroll
#動的にlossとdistanceに係数をかけて、どちらかに偏重しないようにする
#最初のイテレーションのlossとdistanceを1として、前回下がっていない方の損失を増やす
#どちらも出来ないことは許すが、どっちかが出来ることは罰する
def custom_loss(output, target, distance):
    loss = (torch.mean((output - target)**2)+(distance/100000))/2
    _loss = torch.mean((output - target)**2).data.item()
    _distance = distance/1000
    print(f'{_distance - _loss}|{_loss}|{_distance}')
    return loss

#%%
def z_score(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore, xmean, xstd

def reverse_z_score(x, input_mean, input_std, axis = None):
    reverse_zscore = (x+input_mean)*input_std
    return reverse_zscore

#%%
def plot_swissroll(sr, color):
    if np_sr.shape[1] == 3:
        fig = plt.figure()#図の宣言
        ax = fig.add_subplot(111, projection='3d')#三次元で[1,1,1]の位置にプロット
        ax.scatter(sr[:, 0], sr[:, 1], sr[:, 2], c=color, cmap=plt.cm.Spectral)
    elif np_sr.shape[1] == 2:
        plt.scatter(sr[:, 0], sr[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.xticks([]), plt.yticks([])
    plt.show()
#%%
np_sr = np.array(sr)
#plot_swissroll(sr, color)
# %%
np_sr, input_mean, input_std = z_score(np_sr)#zスコアで標準化
dimension=3#スイスロールは3次元のため一応明示的に書いておく
sr_min=np.amin(np_sr.reshape(n_samples*dimension), axis=0)#clampで標準化してから学習するが、
sr_max=np.amax(np_sr.reshape(n_samples*dimension), axis=0)#最終的に復元するときのために最大最小を保存
print(f'min:{sr_min}, max:{sr_max}')

# %%
model = autoencoder().cuda()
criterion = custom_loss#nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-6)

#%%
in_tensor = torch.from_numpy(np_sr.astype(np.float32))#np_srをテンソルにしたもの
print(f"in_tensor:{in_tensor.size()}")

#%%
for epoch in range(num_epochs):
    for data in DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True):
        batch = data
        batch = batch.reshape(batch.size(0)*3)
        batch = Variable(batch).cuda()
        # ===================forward=====================
        output, distance = model(batch)
        loss = criterion(output, batch, distance)
        #print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.data.item()}')
    if epoch % 10 == 0:
        pass#あとで復元がどれくらいできているかここに書いてもいいかも

#%%
result=np.empty((0,3))
for n, data in enumerate(DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=False)):#シャッフルしない
    print(f'TEST:{n}')
    batch = data
    batch = batch.reshape(batch.size(0)*3)
    batch = Variable(batch).cuda()
    # ===================forward=====================
    output, _ = model(batch)
    result=np.vstack([result, output.data.cpu().numpy().reshape(BATCH_SIZE, INPUT_AXIS)])
    
plot_swissroll(reverse_z_score(result, input_mean, input_std), color)

# loss the Variable,
# loss.data the (presumably size 1) Tensor,
# loss.data[0] the (python) float at position 0 in the tensor.
torch.save(model.state_dict(), './autoencoder.pth')

# %%
