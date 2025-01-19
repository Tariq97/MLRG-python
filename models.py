# Created by Tariq Anwar Aquib, 2024, KAUST
# tariqanwar.aquib@kaust.edu.sa
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()



        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_Vr2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO_Vr2d, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 

        self.fc0 = nn.Linear(5, self.width)  

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.permute(x,[0,2,3,1])
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.permute(0,3,1,2)
        return x


# getting estimation  of Vr model....

def get_vr_estimations(path_model,test_inputs,modes,width):

    model = FNO_Vr2d(modes, modes, width).cuda()
    model.load_state_dict(torch.load(path_model))
    model = model.float()


    from monai.inferers import SlidingWindowInferer

    input_tensor = torch.zeros([5,test_inputs.shape[0],test_inputs.shape[1]])

    input_tensor[0,:,:] = test_inputs[:,:,0]
    input_tensor[1,:,:] = test_inputs[:,:,1]
    input_tensor[2,:,:] = test_inputs[:,:,2]
    input_tensor[3,:,:] = test_inputs[:,:,3]
    input_tensor[4,:,:] = test_inputs[:,:,4]

    input_tensor = input_tensor.unsqueeze(0).cuda()
    

    inferer = SlidingWindowInferer(roi_size=(32,32),sw_batch_size=1,overlap=0.75,mode='gaussian',sigma_scale=0.2)
    
    with torch.no_grad():
        pred_gau = inferer(inputs=input_tensor, network=model)

    vr_org = torch.abs(pred_gau.squeeze(0).squeeze(0).cpu())

    a_max = 6599.7611; a_min = 0.0003; vr_org = vr_org*(a_max-a_min) + a_min; 
    
    vr_org = vr_org/1000      
    return vr_org


### PSV model......


class FNO2d_PSV(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_PSV, self).__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 

        self.fc0 = nn.Linear(4, self.width)   

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.permute(x,[0,2,3,1])
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = F.relu(x)

        x = x.permute(0,3,1,2)
        return x


def get_psv_estimations(path_model,test_inputs,modes,width):

    model = FNO2d_PSV(modes, modes, width).cuda()
    model.load_state_dict(torch.load(path_model))
    model = model.float()


    from monai.inferers import SlidingWindowInferer

    input_tensor = torch.zeros([4,test_inputs.shape[0],test_inputs.shape[1]])

    input_tensor[0,:,:] = test_inputs[:,:,0]
    input_tensor[1,:,:] = test_inputs[:,:,1]
    input_tensor[2,:,:] = test_inputs[:,:,2]
    input_tensor[3,:,:] = test_inputs[:,:,3]

    input_tensor = input_tensor.unsqueeze(0).cuda()

    inferer = SlidingWindowInferer(roi_size=(32,32),sw_batch_size=1,overlap=0.75,mode='gaussian',sigma_scale=0.1)
    with torch.no_grad():
        pred_gau = inferer(inputs=input_tensor, network=model)
    a_max = 2.133
    a_min = 0.0283
    pred_gau = pred_gau*(a_max-a_min) + a_min
    return pred_gau

