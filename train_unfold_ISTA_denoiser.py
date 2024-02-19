import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader
import pywt
import h5py
import random
from pytorch_wavelets import DWTForward, DWTInverse
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean(dim=(2, 3), keepdim=True)
    mu2 = img2.mean(dim=(2, 3), keepdim=True)

    sigma1_sq = ((img1 - mu1)**2).mean(dim=(2, 3), keepdim=True)
    sigma2_sq = ((img2 - mu2)**2).mean(dim=(2, 3), keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=(2, 3), keepdim=True)

    SSIM_num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    SSIM_den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    SSIM = (SSIM_num / SSIM_den).mean()

    return 1 - SSIM  # To make it a loss, we use 1 - SSIM

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        l1_loss = F.l1_loss(output, target)
        ssim_loss = ssim(output, target)

        combined_loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss
        return combined_loss

    
def constrain_parameters(model):
    with torch.no_grad():
        for layer in model.layers:
            if isinstance(layer, proxlayer):
                layer.lamda0[layer.lamda0 < 0] = 0.001
                layer.lamda1[layer.lamda1 < 0] = 0.001
                layer.lamda2[layer.lamda2 < 0] = 0.001
                layer.lamda3[layer.lamda3 < 0] = 0.001
            if isinstance(layer, proxflayer):
                if layer.alpha < 0.: #projecting alpha to [0,2]
                    layer.alpha = nn.Parameter(torch.tensor(0.001))


# Create custom dataset to yield patches
class PatchDataset(Dataset):
    def __init__(self, files, patch_size, stride, transform=None):
        self.files = files
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.patches = []
        self.calculate_patches()

    def calculate_patches(self):
        for file in self.files:
            image = Image.open(file)
            width, height = image.size
            for i in range(0, height, self.stride):
                for j in range(0, width, self.stride):
                    if i + self.patch_size <= height and j + self.patch_size <= width:
                        self.patches.append((file, i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        file, i, j = self.patches[idx]
        image = Image.open(file)
        patch = image.crop((j, i, j + self.patch_size, i + self.patch_size))

        if self.transform:
            patch = self.transform(patch)

        return patch


# create training and validation dataloader
def create_dataloaders(train_dir, batch_size, patch_size, stride, val_split, seed=42):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    files = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    train_files, val_files = train_test_split(files, test_size=val_split, random_state=seed)

    train_dataset = PatchDataset(train_files, patch_size, stride, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PatchDataset(val_files, patch_size, stride, transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


# Gradient layer
class gradientlayer(nn.Module):
    def __init__(self, grad_in):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))
    def forward(self, x, y):
        return (1 - self.alpha)*x + self.alpha*y

# Proximal map of data-fidelity term for denoising
class proxflayer(nn.Module):
    def __init__(self, grad_in):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(grad_in))
    def forward(self, x, y):
        return (1 / (1 + self.alpha))*x + (self.alpha/(1 + self.alpha))*y

# Proximal map of regularizer
class proxlayer(nn.Module):
    def __init__(self, prox_in, wavefamily = "db4", levels=5):
        super().__init__()
        self.lamda0 = nn.Parameter(prox_in[0]*torch.ones((1,1,8,8), dtype=torch.float32))
        self.lamda1 = nn.Parameter(prox_in[1]*torch.ones((1,1,3,32,32), dtype=torch.float32))
        self.lamda2 = nn.Parameter(prox_in[2]*torch.ones((1,1,3,16,16), dtype=torch.float32))
        self.lamda3 = nn.Parameter(prox_in[3]*torch.ones((1,1,3,8,8), dtype=torch.float32))
        self.xfwt =  DWTForward(J=levels, wave=pywt.Wavelet(wavefamily), mode="periodization").cuda()
        self.xinvfwt = DWTInverse(wave=pywt.Wavelet(wavefamily), mode="periodization").cuda()
        self.thresh = nn.ReLU()
        
    def forward(self, x, y):
        (yl, yh) = self.xfwt(x)
        ylnew = self.thresh(yl-self.lamda0) - self.thresh(-yl-self.lamda0)
        yhnew = []
        yhnew.append(self.thresh(yh[0]-self.lamda1) - self.thresh(-yh[0]-self.lamda1))
        yhnew.append(self.thresh(yh[1]-self.lamda2) - self.thresh(-yh[1]-self.lamda2))
        yhnew.append(self.thresh(yh[2]-self.lamda3) - self.thresh(-yh[2]-self.lamda3))
        return self.xinvfwt((ylnew, yhnew))


# Neural network
class nonexpansivenn(nn.Module):
    def __init__(self, prox_in, wavefamilyset, levels=5, grad_in = 0.1, num_of_layers = 10):
        super(nonexpansivenn,self).__init__()
        self.layers = nn.ModuleList([gradientlayer(grad_in)])
        for wave in wavefamilyset:
            self.layers.append(proxlayer(prox_in, wave, levels))
        for i in range(num_of_layers-1):
            self.layers.append(proxflayer(grad_in))
            for wave in wavefamilyset:
                self.layers.append(proxlayer(prox_in, wave, levels))
    def forward(self, x):
        out = self.layers[0](x,x)
        for layer in self.layers:
            out = layer(out,x)
        return out

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

if __name__ == '__main__':


    train_dir = '/data/Pravin/PnP_unfold_final_denoiser_trial/data/train'
    batch_size = 64
    patch_size = 64
    sigma = 5.

    # Creating dataloader
    stride=4
    val_split = 0.1
    loader_train, loader_val = create_dataloaders(train_dir, batch_size, patch_size, stride, val_split)

    # Initializing model
    no_layers = 10
    grad_in = 1.
    levels = 3
    prox_in = [0.0001, 0.001, 0.001, 0.01, 0.1, 0.1]
    wavefamilyset=["db4", "haar", "sym4"]
    lr = 0.0001
    num_epochs = 50
    model = nonexpansivenn(prox_in, wavefamilyset, levels, grad_in, no_layers).cuda()
    saved_model_path = "/home/lisa/pravin/JMIV_revision/new_models/"

    # Optimizer and loss function    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CombinedLoss(alpha=0.7)
    criterion.cuda()

    print("Training started !!!")

    # Training over epochs
    for epoch in range(num_epochs):
        if epoch <= 10:
            current_lr = lr
        elif epoch > 10 and epoch <= 20:
            current_lr = lr / 10.
        else:
            current_lr = lr / 100.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=sigma/255.) #Generating noise
            imgn_train = img_train + noise
            imgn_train, img_train = imgn_train.to(device), img_train.to(device)
            out_train = model(imgn_train)
            loss = criterion(out_train, img_train) 
            loss.backward()
            optimizer.step()
            constrain_parameters(model)
            model.eval()
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            if (i + 1) % 100 == 0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            ## the end of each epoch

       # validate after every 5 epochs
        torch.cuda.empty_cache()
        model.eval()
        if (epoch + 1) % 5 == 0:
            val_loss = 0.
            psnr_val = 0.
            for j, val_data in enumerate(loader_val, 0):
                val_noise = torch.FloatTensor(val_data.size()).normal_(mean=0, std=sigma/255.) #Generating noise
                val_noisydata = val_data + val_noise
                val_noisydata, val_data = val_noisydata.to(device), val_data.to(device)
                val_denoised = model(val_noisydata)
                val_loss_inter = criterion(val_denoised, val_data) 
                psnr_val_inter = batch_PSNR(val_denoised, val_data, 1.)
                val_loss += val_loss_inter.item()
                psnr_val += psnr_val_inter
            print("[epoch %d] validation_loss: %.4f PSNR_validation: %.4f" %
                  (epoch + 1, val_loss / (j+1), psnr_val / (j+1)))
            act_str = saved_model_path + "denoiser_ISTA" + "_sigma_" + str(sigma) + "_epoch_" + str(epoch+1) +".pth"
            torch.save(model.state_dict(), act_str)



