#%%
#https://blog.amedama.jp/entry/2017/03/19/160121
#https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import math
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from RBFN_model import *
from radam import *
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
################################
####NeuroScaleは教師データ使うのか？
################################
#%%
# seed = 5#乱数を固定しておく
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

#%%
#パラメータ
num_epochs = 20
learning_rate = 0.0000001
wd = 0
#%%
#Blob データを生成する
dataset = datasets.make_blobs(n_samples=1000, centers=N_LABELS)
features = dataset[0]#入力データ
targets = dataset[1]#教師データ

#%%
#K-meansでprototypeにするcenterを取得
cls = KMeans(n_clusters=N_CENTROIDS) 
pred = cls.fit_predict(features)

#%%
#各要素をラベルごとに色付けして表示
# for n_label in range(N_LABELS):
#     #ラベルごとにfeatureを抽出、描画
#     points_index = [idx for idx, value in enumerate(targets) if value == n_label]
#     points = features[points_index]
#     plt.scatter(points[:, 0], points[:, 1])
plt.scatter(features[:, 0], features[:, 1])

clusterd_labels = cls.labels_
centers = cls.cluster_centers_# クラスタのセントロイド (重心) を描く
#centers[n]はラベルがnのセントロイド
#print(centers)
plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')
plt.show()

#%%
centers_beta = []#それぞれのクラスタに対するBeta coefficientを格納する配列
for i in range(N_CENTROIDS):
    n_points = features[pred == i]#それぞれのクラスタに属する値を抽出
    #centerがlen(n_points)個並ぶ配列を作って下で使う
    n_center = [centers[i] for n, p in enumerate(n_points)]#回すだけでnとpは使わない
    sigma = 0
    beta = 0
    #Kmeansでcentroidを決めたとき、σはセントロイドと、
    #セントロイドが代表するクラスタ内の全データポイントとの距離の平均
    for p, c in zip(n_points, n_center):
        sigma += np.linalg.norm(p-c, ord=2)#2乗ノルムで良いのか？
    sigma /= len(n_points)
    beta = (1 / (2*(sigma**2)))
    centers_beta.append(beta)
print(centers_beta)

#%%
# print(centers)
# print(features)

#%%
def GetActivatedValue(c_beta, input_data, prototype):
    """
    RBFNの活性化関数適応後の値を出力し単層ネットワークの入力とする
    """
    exp = math.e
    distance = np.linalg.norm(input_data-prototype, ord=2)#L2ノルム
    return exp**(-c_beta*(distance**2))#簡略化した活性化関数

#%%
model = RadialBasisFunctionNetwork().cuda()
criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
#optimizer = RAdam(model.parameters(), lr=learning_rate, weight_decay=wd)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)

#%%
print('learning...')
in_tensor = torch.from_numpy(np.array(features).astype(np.float32))
list_targets = targets.tolist()#targetを配列操作の都合上listにしておく
for epoch in range(1, num_epochs+1):
    model.train() #モデルを学習モードに
    data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=True)#batch_sizeは1
    all_loss = 0
    for data in data_iter:
        input_data = data.numpy().tolist()#取り出したデータをリストに変換
        batch_target = [list_targets[in_tensor.tolist().index(i)] for i in input_data]
        batch_target_tensor_onehot = torch.eye(N_LABELS)[batch_target].cuda()
        #print(batch_target_tensor_onehot)
        rbf_output=[]
        for n, prototype in enumerate(centers):#中間層の終わりまでを計算する
            rbf_output.append(GetActivatedValue(centers_beta[n], input_data, prototype))
        network_input = torch.FloatTensor(rbf_output).cuda()
        batch = network_input#RBFニューロンの出力を最終層へ
        output = model(batch)
        loss = criterion(output, batch_target_tensor_onehot)#one-hotじゃないと比較できない？
        loss.backward()
        optimizer.step()
        all_loss += loss.item()
    print('{}/{:.6f}'.format(epoch, all_loss))

#%%
model.eval()#モデルを評価モードに
data_iter = DataLoader(in_tensor, batch_size=BATCH_SIZE, shuffle=False)#batch_sizeは1
all_loss = 0
result = []
for data in data_iter:
    input_data = data.numpy().tolist()#取り出したデータをリストに変換
    batch_target = [list_targets[in_tensor.tolist().index(i)] for i in input_data]
    batch_target_tensor_onehot = torch.eye(N_LABELS)[batch_target].cuda()
    #print(batch_target_tensor_onehot)
    rbf_output=[]
    for n, prototype in enumerate(centers):#中間層の終わりまでを計算する
        rbf_output.append(GetActivatedValue(centers_beta[n], input_data, prototype))
    network_input = torch.FloatTensor(rbf_output).cuda()
    batch = network_input#RBFニューロンの出力を最終層へ
    output = model(batch).data.cpu().numpy()
    #outputをone-hotのndarrayに変換しておく
    output_max_index = np.argmax(output)
    output_onehot = np.zeros(shape=N_LABELS)
    output_onehot[output_max_index] = 1
    # print(output_onehot)
    result.append(list(output_onehot))

#one-hot2scalar
scalar_result_list=[]
for x in result:
    scalar_result = x.index(1)#onehot配列中の1の場所を取得
    scalar_result_list.append(scalar_result)

for n_label in range(N_LABELS):
    points_index = [idx for idx, result in enumerate(scalar_result_list) if result == n_label]
    points = features[points_index]
    plt.scatter(points[:, 0], points[:, 1])

centers = cls.cluster_centers_# クラスタのセントロイド (重心) を描く
#centers[n]はラベルがnのセントロイド
# print(centers)
plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')
plt.show()

# %%
