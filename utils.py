import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
import itertools
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import time

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(SimpleCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

# Proximal map of regularizer    
class istalayer(nn.Module):
    def __init__(self, wavefamily = "db4", levels=5, grad_in=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))
        self.lamda0 = nn.Parameter(torch.ones((1,1,8,8), dtype=torch.float32))
        self.lamda1 = nn.Parameter(torch.ones((1,1,3,32,32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(torch.ones((1,1,3,16,16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(torch.ones((1,1,3,8,8), dtype=torch.float32))
        self.levels = levels
        self.wavefamily = wavefamily
    def forward(self, x, y):
        xfwt =  DWTForward(J=self.levels, wave=pywt.Wavelet(self.wavefamily), mode="periodization").cuda()
        xinvfwt = DWTInverse(wave=pywt.Wavelet(self.wavefamily), mode="periodization").cuda()
        thresh = nn.ReLU()
        x_inter = (1 - self.alpha)*x + self.alpha*y
        (yl, yh) = xfwt(x_inter)
        ylnew = thresh(yl-self.lamda0) - thresh(-yl-self.lamda0)
        yhnew = []
        yhnew.append(thresh(yh[0]-self.lamda1) - thresh(-yh[0]-self.lamda1))
        yhnew.append(thresh(yh[1]-self.lamda2) - thresh(-yh[1]-self.lamda2))
        yhnew.append(thresh(yh[2]-self.lamda3) - thresh(-yh[2]-self.lamda3))
        return xinvfwt((ylnew, yhnew))

# Neural network    
class nonexpansivenn_ISTA(nn.Module):
    def __init__(self, levels=3, grad_in = 0.1, num_of_layers = 10):
        super(nonexpansivenn_ISTA,self).__init__()
        wavefamilyset = ["db4", "haar", "sym4"]
        self.layers = nn.ModuleList([istalayer(wavefamilyset[0], levels, grad_in)])
        for i in range(1,len(wavefamilyset)):
            self.layers.append(istalayer(wavefamilyset[i], levels, grad_in))
        for i in range(num_of_layers-1):
            for wave in wavefamilyset:
                self.layers.append(istalayer(wave, levels, grad_in))
    def forward(self, x):
        out = self.layers[0](x,x)
        for layer in self.layers:
            out = layer(out,x)
        return out

class drslayer(nn.Module):
    def __init__(self, wavefamily = "db4", levels=5, grad_in=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))
        self.lamda0 = nn.Parameter(torch.ones((1,1,8,8), dtype=torch.float32))
        self.lamda1 = nn.Parameter(torch.ones((1,1,3,32,32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(torch.ones((1,1,3,16,16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(torch.ones((1,1,3,8,8), dtype=torch.float32))
        self.levels = levels
        self.wavefamily = wavefamily
    def forward(self, x, y):
        xfwt =  DWTForward(J=self.levels, wave=pywt.Wavelet(self.wavefamily), mode="periodization").cuda()
        xinvfwt = DWTInverse(wave=pywt.Wavelet(self.wavefamily), mode="periodization").cuda()
        thresh = nn.ReLU()
        x_inter = (1/(1 + self.alpha))*x + (self.alpha/(1 + self.alpha))*y
        x_inter1 = 2*x_inter - x
        (yl, yh) = xfwt(x_inter1)
        ylnew = thresh(yl-self.lamda0) - thresh(-yl-self.lamda0)
        yhnew = []
        yhnew.append(thresh(yh[0]-self.lamda1) - thresh(-yh[0]-self.lamda1))
        yhnew.append(thresh(yh[1]-self.lamda2) - thresh(-yh[1]-self.lamda2))
        yhnew.append(thresh(yh[2]-self.lamda3) - thresh(-yh[2]-self.lamda3))
        x_inter2 = 2*xinvfwt((ylnew, yhnew)) - x_inter1
        return 0.5*x_inter2 + 0.5*x

# Neural network
class nonexpansivenn_DRS(nn.Module):
    def __init__(self, levels=3, grad_in = 0.1, num_of_layers = 10):
        super(nonexpansivenn_DRS,self).__init__()
        wavefamilyset = ["db4", "haar", "sym4"]
        self.layers = nn.ModuleList([drslayer(wavefamilyset[0], levels, grad_in)])
        for i in range(1, len(wavefamilyset)):
            self.layers.append(drslayer(wavefamilyset[i], levels, grad_in))
        for i in range(num_of_layers-1):
            for wave in wavefamilyset:
                self.layers.append(drslayer(wave, levels, grad_in))

    def forward(self, x):
        out = self.layers[0](x,x)
        for layer in self.layers:
            out = layer(out,x)
        return out


def load_model_CNN(model_type, sigma, path):
    print(path)
    if model_type == "DnCNN" or model_type == "RealSN_DnCNN":
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).cuda()
    else:
        model = SimpleCNN(channels=1, num_of_layers=4).cuda()
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()
    return model

def load_model_CNN_2(model_type, sigma, path):
    print(path)
    if model_type == "DnCNN" or model_type == "RealSN_DnCNN":
        model = DnCNN(channels=1, num_of_layers=17).cuda()
    else:
        model = SimpleCNN(channels=1, num_of_layers=4).cuda()
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()
    return model

def load_model_custom_CNN(model_type, sigma, path):
    print(path)
    if model_type == "DnCNN" or model_type == "RealSN_DnCNN":
        model = DnCNN(channels=1, num_of_layers=17).cuda()
    elif model_type == "DnCNN_nobn" or model_type == "RealSN_DnCNN_nobn":
        model = SimpleCNN(channels=1, num_of_layers=17).cuda()
    else:
        model = SimpleCNN(channels=1, num_of_layers=4).cuda()
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()
    return model

def load_model_ISTA(sigma, path):
    net = nonexpansivenn_ISTA()
    model = net.cuda()
    new_dict = torch.load(path)
    model.load_state_dict(new_dict)
    model.eval()
    return model

def load_model_DRS(sigma, path):
    net = nonexpansivenn_DRS()
    model = net.cuda()
    new_dict = torch.load(path)
    model.load_state_dict(new_dict)
    model.eval()
    return model

def find_test_image_circular_pad(imgn_val, model, stride, patch_width, batch_size):
    patch_radius = int(patch_width/2)
    paddim = (patch_radius, patch_radius, patch_radius, patch_radius)
    imgn_pad_val = F.pad(imgn_val, paddim, 'circular')
    rows = imgn_pad_val.shape[2]
    cols = imgn_pad_val.shape[3]
    out_pad_val = torch.zeros(imgn_pad_val.shape)
    rowval = int((rows - patch_width)/stride)
    colval = int((cols - patch_width)/stride)
    arr=list(itertools.product(list(stride*element for element in range(rowval)),list(stride*element for element in range(colval))))
    
    check = torch.zeros([batch_size,1,patch_width,patch_width]).cuda()
    with torch.no_grad():
        for index in range(0,len(arr),batch_size):
            values = min(len(arr)-index, batch_size) 
            check.zero_()
            for index2 in range(values):
                check[index2,:,:,:] = imgn_pad_val[:,:,arr[index+index2][0]:arr[index+index2][0]+patch_width,arr[index+index2][1]:arr[index+index2][1]+patch_width].cuda()
            check =  torch.clamp(model(check), 0., 1.)
            for index2 in range(values):
                out_pad_val[:,:,arr[index+index2][0]:arr[index+index2][0]+patch_width,arr[index+index2][1]:arr[index+index2][1]+patch_width] += check[index2,:,:,:].cpu()

        
    out_val = out_pad_val[:,:,patch_radius:-patch_radius,patch_radius:-patch_radius]
    rows_orig = out_val.shape[2]
    cols_orig = out_val.shape[3]
    out_val[:,:,0:patch_radius,0:patch_radius] += out_pad_val[:,:,rows-patch_radius:rows,cols-patch_radius:cols]
    out_val[:,:,0:patch_radius,cols_orig-patch_radius:cols_orig] += out_pad_val[:,:,rows-patch_radius:rows,0:patch_radius]
    out_val[:,:,rows_orig-patch_radius:rows_orig,0:patch_radius] += out_pad_val[:,:,0:patch_radius,cols-patch_radius:cols]
    out_val[:,:,rows_orig-patch_radius:rows_orig,cols_orig-patch_radius:cols_orig] += out_pad_val[:,:,0:patch_radius,0:patch_radius]
    out_val[:,:,:,0:patch_radius] += out_pad_val[:,:,patch_radius:rows_orig+patch_radius,cols-patch_radius:cols]
    out_val[:,:,:,cols_orig-patch_radius:cols_orig] += out_pad_val[:,:,patch_radius:rows_orig+patch_radius,0:patch_radius]
    out_val[:,:,0:patch_radius,:] += out_pad_val[:,:,rows-patch_radius:rows,patch_radius:cols_orig+patch_radius]
    out_val[:,:,rows_orig-patch_radius:rows_orig,:] += out_pad_val[:,:,0:patch_radius,patch_radius:cols_orig+patch_radius]

    out_val = out_val / ((patch_width/stride)**2)
    return out_val
