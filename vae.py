#%%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time
import random
import numpy as np
from coranking import *
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import pandas as pd
import plotly
plotly.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=args.batch_size, num_workers=0, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, num_workers=0, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21A = nn.Linear(400, 200)
        self.fc21B = nn.Linear(200, 2)
        self.fc22A = nn.Linear(400, 200)
        self.fc22B = nn.Linear(200, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21B(self.fc21A(h1)), self.fc22B(self.fc22A(h1))#20個ずつのμとσ

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #0に平均を足してから、-x < σ < +xまでの乱数で分散を足すと正規分布の形になる
        eps = torch.randn_like(std)#平均0分散1の乱数をstdと同じテンソルの形で生成 -1~+1
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def plot_latent(in_data, color):
    df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Labels':color}).sort_values('Labels')
    fig = px.scatter(df, x='X', y='Y', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size_max=5, opacity=0.5)
    fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=False)#縦横比を1:1に
    return fig


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #N(0,1)と比較する場合は厳密なKLダイバージェンスではなく以下の計算で近似できる
    #logvarが1、muが0に近づくと全体が0になる
    #-0.5 * sum(1+ loge(1) -0^2 -1)
    #このときloge(1)は0なので-0.5*sum(0)=0になる
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#logvar.exp()で自然対数を外してvar=sigma**2にする

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'comparision/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#%%
if __name__ == "__main__":
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 2).to(device)
        sample = model.decode(sample)
        save_image(sample.view(64, 1, 28, 28), 'comparision/sample_' + str(epoch) + '.png')
        encoded_tr, encoded_te, lat_result, in_data, color = list(), list(), list(), list(), list()
        for train_batch_idx, (x_train, y_train) in enumerate(train_loader, start=0):
            encoded_train = model.encode(x_train.to('cuda:0').view(-1, 784))
            reparametized_train = model.reparameterize(encoded_train[0], encoded_train[1])
            for idx, tr in enumerate(x_train):
                in_data.append(tr.tolist())
            for idx, entr in enumerate(reparametized_train):
                lat_result.append(entr.tolist())
            for idx, cltr in enumerate(y_train):
                color.append(cltr.tolist())
        for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
            encoded_test = model.encode(x_test.to('cuda:0').view(-1, 784))
            reparametized_test = model.reparameterize(encoded_test[0], encoded_test[1])
            for idx, te in enumerate(x_test):
                in_data.append(te.tolist())
            for idx, ente in enumerate(reparametized_test):
                lat_result.append(ente.tolist())
            for idx, clte in enumerate(y_test):
                color.append(clte.tolist())
        n_samples = 70000
        print(len(in_data[0]))#len(in_data[0][0])=28, len(in_data[0][1])=28
        print(len(lat_result[0]))
        in_data = np.array(in_data).reshape(n_samples, 784)
        
        elapsed_time = time.time() - start_time
        s=0
        sampling_num = 1000
        n_sampling_iter = 70
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
            local_score += CoRanking(rnd_in_data).evaluate_corank_matrix(rnd_lat_result, 50, 5) / n_sampling_iter
        print(f'LOCAL_SCORE:{local_score}')
#%%
        sampling_num = 3000
        rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
        rnd_in_data = np.array([in_data[i] for i in rnd_idx])
        rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
        rnd_color = np.array([str(color[i]) for i in rnd_idx])
        plot_latent(rnd_lat_result, rnd_color).update_layout(title='VAE').write_image(f"./comparision/VAE.png")
    
# %%
