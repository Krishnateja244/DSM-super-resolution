import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from glob import glob
import torchvision.transforms as transforms
from rasterio.plot import reshape_as_image, reshape_as_raster, show
import rasterio
from rasterio import Affine
from rasterio.windows import Window
import data_normalization
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def compute_local_dsm_std_per_centered_patch(dataloader, raster_identifier='DSM'):
    """
    Computes a single, robust scale factor across all DSM training data samples, so as to preserve a (relative) notion
    of scale. The function first centers each training DSM patch to its mean height. Then, it computes the standard
    deviation of the height within each patch, discards standard deviations below the 5th percentile and above the 95th
    percentile to ensure robustness, and averages the remaining ones to obtain a single, robust estimate of the
    standard deviation.
    :param dataloader:          torch.utils.data.DataLoader instance
    :param raster_identifier:   str, identifier to select the DSM data source
                                (default: 'raster_in' to use the initial DSM)
    :return:                    float, standard deviation of the zero-centered DSM training patches
    """
    
    
    # Extract the number of batches
    n_batches = dataloader.__len__()
    print ("n_batches", n_batches)

    # Initialize buffers
    stds = np.zeros(n_batches, dtype=float)
    means = np.zeros(n_batches, dtype=float)

    #pdb.set_trace()
    # Compute the standard deviation over all training pixels per batch (= per single sample)
    for i, batch in enumerate(dataloader):

        if raster_identifier == 'DSM':
            
            x = batch[0][:, 0, :, :].numpy().astype(np.float128)
        else:
            x = batch[1][:, 0, :, :].numpy().astype(np.float128)

        mean_per_sample = x.mean(axis=(1, 2), keepdims=True)
        
        stds[i] = np.sqrt(((x - mean_per_sample) ** 2).sum() / (x.size - 1))
        means[i] = np.mean(mean_per_sample)
        
        # print ("mean_per_sample:", mean_per_sample, "stds: ", stds[i])

    # Discard standard deviations below the 5th percentile and above the 95th percentile to ensure robustness and
    # average the remaining ones to obtain a single, robust estimate of the standard deviation
    perc95 = np.percentile(stds, 95)
    perc5 = np.percentile(stds, 5)
    std = stds[np.logical_and(stds >= perc5, stds <= perc95)].mean().item()

    perc95 = np.percentile(means, 95)
    perc5 = np.percentile(means, 5)
    mean = means[np.logical_and(means >= perc5, means <= perc95)].mean().item()
    

    return mean,std



class CustomDataset(Dataset):
    def __init__(self,hr_dir,rgb_hr_dir,num_samples,tile_size,scale,model=None,dsm_mean=None,dsm_std=None,transform=None,test_set=False): #lr_dir
        # self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        self.num_samples = num_samples
        self.tile_size = tile_size
        self.scale = scale
        self.model = model
        
        with open(self.hr_dir,"r") as hr:
            if not test_set:
                self.hr_filepath = hr.readlines()[:self.num_samples]
            else:
                self.num_samples = -num_samples
                self.hr_filepath = hr.readlines()[self.num_samples:]
        self.rgb_hr_dir = rgb_hr_dir
        with open(self.rgb_hr_dir,"r") as rgb_hr:
            if not test_set:
                self.rgb_hr_filepath = rgb_hr.readlines()[:self.num_samples]
            else:
                self.num_samples = -num_samples
                self.rgb_hr_filepath = rgb_hr.readlines()[self.num_samples:]
        
        self.mean = dsm_mean
        self.std = dsm_std

        self.torch_transform = transforms.Compose([transforms.ToTensor()])

        self.rgb_img_transform = transforms.Compose([
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.hr_filepath)
    
    def __getitem__(self,index):

        xoffset = int(self.hr_filepath[index].split(",")[1])
        yoffset = int(self.hr_filepath[index].split(",")[2])
        
        file_name = self.hr_filepath[index].split(",")[0]
        hr = rasterio.open(file_name) #self.hr_filepath[index])
        
        window = Window(xoffset,yoffset,self.tile_size,self.tile_size)
        
        hr_array = hr.read(1, window = window) 

        rgb_hr_file_name = self.rgb_hr_filepath[index].split(",")[0]
        rgb_hr = rasterio.open(rgb_hr_file_name)
        rgb_hr_window = Window(xoffset,yoffset,self.tile_size,self.tile_size)
        rgb_hr_array = rgb_hr.read(window = rgb_hr_window)
        
        
        lr_array = cv2.resize(hr_array,(int(self.tile_size*self.scale),int(self.tile_size*self.scale)),cv2.INTER_CUBIC)
        lr_2 = cv2.resize(hr_array,(int(self.tile_size*1/2),int(self.tile_size*1/2)),cv2.INTER_CUBIC) 

        if self.model == "pix2pix" or self.model == "pix2pixhd" or self.model == "srvae" or self.model =="enc_srgan":
            lr_array = cv2.resize(lr_array,(int(self.tile_size),int(self.tile_size)),cv2.INTER_CUBIC)
            lr_2 = cv2.resize(lr_2,(int(self.tile_size),int(self.tile_size)),cv2.INTER_CUBIC)
        

        if self.transform:

            if self.mean:
                mean = self.mean
            else:
                
                mean = np.mean(lr_array)

            if self.std:
                std = self.std
            
            trans = data_normalization.get_transform(mean,std)
            
            lr_array = trans(lr_array)
            lr_2 = trans(lr_2)
            hr_array = trans(hr_array)

            rgb_hr_array = self.rgb_img_transform(torch.FloatTensor(rgb_hr_array))

        else:
            mean = 0 
            std = 0
            
            lr_array = torch.from_numpy(lr_array).unsqueeze(0)
            lr_2 = torch.from_numpy(lr_2).unsqueeze(0)
            hr_array = torch.from_numpy(hr_array).unsqueeze(0)
            rgb_hr_array = torch.from_numpy(rgb_hr_array)

        return lr_array,lr_2,hr_array,rgb_hr_array,mean,std,file_name


if __name__=="__main__":
    train_dataset = CustomDataset(hr_dir='../datasets/swiss_dsm/trainset/hr_256_files.txt', \
                                num_samples=10,tile_size=256,model="multisrgan",scale=1/4,transform=None)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    
    mean,std = compute_local_dsm_std_per_centered_patch(train_loader)
    
    # param_file = open("train_norm_parameters.txt","w")
    # param_file.write(f"Mean,{mean}\n")
    # param_file.write(f"Std,{std}")      

    test_dataset = CustomDataset(hr_dir='../datasets/swiss_dsm/testset/hr_256_files.txt',rgb_hr_dir="../datasets/swiss_dsm/rgb_testset/hr_256_files.txt",  \
                                num_samples=10,tile_size=256,model="srgan",scale=1/4,transform=True,dsm_mean=None,dsm_std=std)

    train_loader = DataLoader(test_dataset,batch_size=4,shuffle=True)

    print(next(iter(train_loader))[0].shape)
    print(next(iter(train_loader))[1].shape)
    print(next(iter(train_loader))[2].shape)
    print(next(iter(train_loader))[3].shape)
    print(next(iter(train_loader))[5])