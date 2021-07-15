#%%
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import random
from coranking import *
import plotly.express as px
import plotly.graph_objects as go
import chart_studio.plotly as py
import pandas as pd
import plotly
torch.manual_seed(123)
plotly.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=2, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
args = parser.parse_args(args=[])

trainset = MNIST(root='./data/',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=True)

testset = MNIST(root='./data/',
                 train=False,
                 transform=transforms.ToTensor(),
                 download=True)

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0)

test_loader = DataLoader(dataset=testset,
                         batch_size=104,
                         shuffle=False,
                         num_workers=0)

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def plot_latent(in_data, color):
    df = pd.DataFrame({'X':in_data[:, 0], 'Y':in_data[:, 1], 'Labels':color}).sort_values('Labels')
    fig = px.scatter(df, x='X', y='Y', color='Labels', color_discrete_sequence=px.colors.qualitative.D3, size_max=5, opacity=0.5)
    fig.update_layout(yaxis=dict(scaleanchor='x'), showlegend=False)#縦横比を1:1に
    return fig

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
        )
        self.fc = nn.Linear(50, self.n_z)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.main(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, 50),
            nn.ReLU(),
        )

        self.main = nn.Sequential(
            nn.Linear(50, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.main(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, self.dim_h * 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h * 4, 1),
            nn.Sigmoid()#出力は0~1でサイズは[batch_size, 1]
        )

    def forward(self, x):
        x = self.main(x)
        return x

encoder, decoder, discriminator = Encoder(args), Decoder(args), Discriminator(args)
criterion = nn.MSELoss()

encoder.train()
decoder.train()
discriminator.train()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr = args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr = args.lr)
dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * args.lr)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)


encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

one = torch.tensor(1, dtype=torch.float)#1
mone = one * -1#-1

one = one.cuda()
mone = mone.cuda()
#TIME:153.0499987602234
# GLOBAL_SCORE:0.32347360000000003
# LOCAL_SCORE:0.20174399999999998
start_time=time.time()
for epoch in range(args.epochs):
    step = 0

    for images, _ in train_loader:
        images = images.cuda()

        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()

        # ======== Train Discriminator ======== #

        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        ############################################################################
        z_real = torch.randn(images.size()[0], args.n_z) * args.sigma#潜在表現と同じテンソルの形のノイズ、N(0,1)からサンプリングしたのちに分散を乗じる
        ############################################################################
        z_real = z_real.cuda()
        d_real = discriminator(z_real)#discriminated_real

        z_fake = encoder(images)
        d_fake = discriminator(z_fake)#discriminated_fake

        ##############################################################################
        #out.backward()は，out.backward(torch.tensor([1.0]))と等価＝正の値の最小化
        #おそらく、backward(-1)をすると、正の値の最小化を「負の値の最小化＝正の値の最大化」に置き換えられる
        ##############################################################################
        #0に漸近する = d_realの出力値が限りなく1にに近づく
        torch.log(d_real).mean().backward(mone)#d_realを最大化
        temp_real_mean = d_real.var().cpu().detach().numpy()
        #-∞に漸近する = d_fakeの出力値が限りなく0に近づく
        torch.log(1-d_fake).mean().backward(mone)#log(1-d_fake)を最大化＝d_fakeを最小化
        temp_fake_mean = d_fake.var().cpu().detach().numpy()
        #Discriminatorはd_realとd_fakeの平均を離す
        ##############################################################################
        dis_optim.step()

        # ======== Train Generator(encoder & decoder) ======== #

        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        batch_size = images.size()[0]

        z_fake = encoder(images)
        x_recon = decoder(z_fake)
        d_fake = discriminator(encoder(Variable(images.data)))#discriminatorを凍結してgeneratorを更新

        recon_loss = criterion(x_recon, images)#MSE
        ###################################################################
        d_loss = args.LAMBDA * (torch.log(d_fake)).mean()#discriminater's loss
        ###################################################################

        recon_loss.backward(one)#再構成誤差を最小化
        d_loss.backward(mone)#d_fakeを最大化＝同じく最大化しているd_realに近づける

        #encoder & decoderは再構成誤差を減らしながら、fakeの平均をrealに近づける
        enc_optim.step()
        dec_optim.step()

        step += 1
        #print(f'{temp_real_mean}|{temp_fake_mean}')
        if (step + 1) % 300 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, args.epochs, step + 1, len(train_loader), recon_loss.data.item()))

    batch_size = 104
    test_iter = iter(test_loader)
    test_data = next(test_iter)

    z_fake = encoder(Variable(test_data[0]).cuda())
    reconst = decoder(torch.randn_like(z_fake)).cpu().view(-1, 784)

    if not os.path.isdir('./data/reconst_images'):
        os.makedirs('data/reconst_images')

    save_image(test_data[0].view(batch_size, 1, 28, 28), './data/reconst_images/wae_gan_input.png')
    save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch + 1))
with torch.no_grad():
    encoded_tr, encoded_te, lat_result, in_data, color = list(), list(), list(), list(), list()
    for train_batch_idx, (x_train, y_train) in enumerate(train_loader, start=0):
        encoded_train = encoder(Variable(x_train).cuda())
        for idx, tr in enumerate(x_train):
            in_data.append(tr.tolist())
        for idx, entr in enumerate(encoded_train):
            lat_result.append(entr.tolist())
        for idx, cltr in enumerate(y_train):
            color.append(cltr.tolist())
    for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
        encoded_test = encoder(Variable(x_test).cuda())
        for idx, te in enumerate(x_test):
            in_data.append(te.tolist())
        for idx, ente in enumerate(encoded_test):
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

    sampling_num = 3000
    rnd_idx = [random.randint(0, n_samples-1) for i in range(sampling_num)]
    rnd_in_data = np.array([in_data[i] for i in rnd_idx])
    rnd_lat_result = np.array([lat_result[i] for i in rnd_idx])
    rnd_color = np.array([str(color[i]) for i in rnd_idx])
    plot_latent(rnd_lat_result, rnd_color).update_layout(title='WAE').write_image(f"./comparision/WAE.png")
