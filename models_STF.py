# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from Slip2Stress import *
from compute_CL import *


import stf_Yoffe
from stf_Yoffe import *

import Yoffe_triangle
from Yoffe_triangle import *

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

   
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)


        out_ft[:, :, :self.modes1] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

def embed(in_dim,up_dim):
    layers = nn.Sequential(
        nn.Linear(in_dim,128, bias=True), torch.nn.ReLU(),
        nn.Linear(128, 128, bias=True), torch.nn.ReLU(),
        nn.Linear(128, up_dim, bias=True)
    )
    return layers

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic

        self.embed = embed(5,501)  # Take meta data with size BATCH_SIZE x 7 and outputs BATCH_SIZE x 68 size..
        self.fc0 = nn.Linear(2, self.width)   ### changed from 12 to 6...now 7 after using 6 inputs (logscale)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 1000)
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x, meta_data):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        #print('x shape afer concat',x.shape)
        m1 = self.embed(meta_data)
        m1 = m1.unsqueeze(-1)
       
        x =torch.cat((x, m1), dim=-1)

        x = self.fc0(x)
        #print('x after first LT',x.shape)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x



def get_predictions_STF(path_model, batch_size, test_a_norm, test_meta_all, rise_time_factor):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              
    path_model = './Trained_models/FNO_stf_norm_all_goodfit'
    if device.type == 'cpu':
        model = torch.load(path_model,map_location=torch.device('cpu'))
        model.eval()
    elif device.type == 'cuda':
        model = torch.load(path_model).cuda()
        model.eval()

    test_a1 = test_a_norm[:, :]
    test_meta1 = test_meta_all[:, :5]

    # Onset times, rise times and slip that needs to be scaled..
    t_onset = test_meta_all[:,5]
    t_r = test_meta_all[:,6]*rise_time_factor
    slip = test_meta_all[:,7]
    
    test_a1 = torch.from_numpy(test_a1)
    test_meta1 = torch.from_numpy(test_meta1)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a1, test_meta1),
        batch_size=batch_size, shuffle=False)
                

    i = 0
    pred_norm = torch.zeros(test_a1.shape)
    with torch.no_grad():
        for xx, meta in test_loader:
            xx = xx.to(device)
            meta = meta.to(device)

            im = model(xx.unsqueeze(-1).float(), meta.float())
            pred_norm[i:i + batch_size, :] = torch.squeeze(im).cpu()
            i += batch_size

    # Making area inside equal to slip
    time = torch.from_numpy(np.linspace(0,20,501))
    pred1_norm = torch.zeros(pred_norm.shape).detach().numpy()
    input1 = torch.zeros(pred_norm.shape).detach().numpy()

    for i in range(0,len(test_a1)):
                    
        pred1_norm[i,:] = pred_norm[i,:].detach().numpy()
        pred1_norm[i,:] = np.abs(pred1_norm[i,:])

        k1 = np.argwhere(time>t_onset[i])
        k2 = np.argwhere(time>(t_onset[i] + t_r[i]))
        if k2.nelement()>0:
            pred1_norm[i,:k1[0][0]] = 0
            pred1_norm[i,k2[0][0]:] = 0

            pred1_norm[i,:] = pred1_norm[i,:]/np.trapz(pred1_norm[i,:], dx=0.04)           # first making it unit area
            pred1_norm[i,:] = pred1_norm[i,:]*(slip[i])
      
            # input model... 
            input1[i,:] = (test_a1[i,:]/np.trapz(test_a1[i,:], dx=0.04)).detach().numpy()
            input1[i,:] = input1[i,:]*(slip[i])

            # In some cases ML can predict all NaNs, especailly when Tr is very small..in that case don't modify anything.
            if np.nansum(pred1_norm[i,:])==0:
                pred1_norm[i,:]  = input1[i,:]
                
    return pred1_norm, input1



def compute_STF_new_points(test_meta,rise_time_factor,tacc_ratio=0.1):

    t_end = 20
    num_pts = 501

    idx = test_meta.shape[0]

    slip_time = np.linspace(0,t_end,num_pts)
    slip_rate = np.zeros(shape=(idx,501))
    slip_rate_norm = np.zeros_like(slip_rate)


    slip = test_meta[:,7]
    Tr_eff = test_meta[:,6]
    t_onset = test_meta[:,5]
    Tacc = Tr_eff*tacc_ratio

    for i in range(0,slip.shape[0]):
        slip_rate[i,:] = stf_Yoffe(slip_time,t_onset[i],Tr_eff[i]*rise_time_factor,slip[i]/100,Tacc[i])  # Rise time is increased; Tacc is kept same.!!!
        slip_rate_norm[i,:] = slip_rate[i,:]/np.max(np.abs((slip_rate[i,:]))) 

    test_meta_new = test_meta
    test_meta_new[:,6] = test_meta[:,6]*rise_time_factor    # Increased rise time

    return slip_rate_norm, test_meta_new
