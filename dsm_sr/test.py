import torch 
import numpy as np

from discriminator import Discriminator
from dataset_loader import CustomDataset,compute_local_dsm_std_per_centered_patch
from torch.utils.data import DataLoader
import data_normalization
import rasterio
import os 
from rasterio import Affine
from effnetv2 import effnetv2_s
import argparse
from pix2pix import UnetGenerator
from rasterio.windows import Window
from generator import SrganGenerator, Encoder 
import cv2
from ResDepth.lib import utils
import logging
import torchvision.transforms as T
import rasterio
from rasterio import Affine
from rasterio.windows import Window
import torch.nn as nn 
from srgan_resca import SRCAGAN
from ESRGAN import ESRGAN

from basicsr.archs.rrdbnet_arch import RRDBNet
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
logger = utils.setup_logger('root_logger', level=logging.INFO, log_to_console=True, log_file=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_eval:

    def __init__(self,args):
        self.args = args
        self.tile_size = 256
        self.batch_size = 4
        self.scale = 1/self.args.down_scale
        
        
        if self.args.netG == "srgan":
            print("using SRGAN model")
            self.model =  SrganGenerator(1, 128,self.args.down_scale).to(device)   

        elif self.args.netG == "enc_srgan":
            print("using the encoder sran")
            self.model = Encoder(1).to(device)

        elif self.args.netG == "srgan_atten":
            print("using SRGAN RESIDUAL CHANNEL ATTENTION")
            self.model = SRCAGAN(1,128,10,10,16).to(device)         

        elif self.args.netG == "netv2":
            print("using EfficientNet V2 model")
            self.model =  effnetv2_s().to(device)

        elif self.args.netG == "pix2pix":
            print("using pix2pix (U-net) model")
            self.model = UnetGenerator(1,1,8).to(device)

        elif self.args.netG =="esrgan":
            print("using ESRGAN model")
            self.model = ESRGAN(1,1,scale_factor=self.args.down_scale).to(device)

        if self.args.gen_weights.split("/")[-1].split(".")[-1] == "pt":
            we = torch.load(self.args.gen_weights)
            print(we["epoch"])
            print(we["rmse"])
            print(we["mae"])
            # if 'params_ema' in we:
            #     keyname = 'params_ema'
            # else:
            #     keyname = 'params'
            # self.model.load_state_dict(we[keyname], strict=True)
            self.model.load_state_dict(torch.load(self.args.gen_weights)["g_model"])

        else:
            self.model.load_state_dict(torch.load(self.args.gen_weights))
        

        self.test_loader = self.get_dataloader()
    
    def load_model(self,model, pretrained):
        weights = torch.load(pretrained)
        pretrained_dict = weights['model'].state_dict()  
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        

    def get_dataloader(self):

        param_file = open(self.args.norm_params,"r").readlines()
        self.mean = param_file[0].split(",")[-1]
        self.std = float(param_file[1].split(",")[-1])

        train_dataset = CustomDataset(hr_dir=self.args.train_hr_path, rgb_hr_dir=self.args.train_rgb_hr_path, \
                                  num_samples=1000,tile_size=self.tile_size,scale=self.scale,model=self.args.netG,transform=None)
      

        train_loader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)

        self.mean,self.std = compute_local_dsm_std_per_centered_patch(train_loader)

        print(f"Normalization parameters Mean: {self.mean}, Std :{self.std}")
        
        
        test_dataset = CustomDataset(hr_dir=self.args.test_hr_path,rgb_hr_dir=self.args.test_rgb_hr_path,num_samples=self.args.num_samples, \
                                    tile_size=self.tile_size,scale=self.scale,model=self.args.netG,transform=True,dsm_mean=None,dsm_std=float(self.std),test_set=False)

        test_loader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=False)

        print("shape of input: ",next(iter(test_loader))[0].shape)
        print("shape of lr2 : ",next(iter(test_loader))[1].shape)
        print("shape of rgb_array : ",next(iter(test_loader))[3].shape)
        print("shape of gt : ",next(iter(test_loader))[2].shape)
        return test_loader

    def test(self):
        hr_filepath = self.args.test_hr_path
        num_val_batches = float(len(self.test_loader))
        with torch.inference_mode():
        
            val_mae_sr = 0
            val_rmse_sr = 0 
            val_mae_bc = 0 
            val_rmse_bc = 0 
            val_absolute_median_sr = 0
            val_median_sr = 0 
            val_nmad_sr = 0 
            val_absolute_median_bc =0
            val_median_bc = 0
            val_nmad_bc = 0 
            self.model.eval()
        
            for i,(lr_array,lr_2,hr_array,rgb_hr_array,mean,std,file_name) in enumerate(self.test_loader):

                lr_array = lr_array.to(device)
                hr_array = hr_array.to(device)
                lr_2 = lr_2.to(device)
                rgb_hr_array= rgb_hr_array.to(device)

                if self.args.netG == "esrgan":
                    lr_images = lr_array.cpu().detach().numpy()
                    hr_images = hr_array.cpu().detach().numpy()
                    dummy_RGB_lr = np.ndarray(shape=(lr_images.shape[0],3, lr_images.shape[2], lr_images.shape[3]), dtype= np.float32)
                    dummy_RGB_lr[:,0,:,:] = lr_images[:,0,:,:]
                    dummy_RGB_lr[:,1,:,:] = lr_images[:,0,:,:]
                    dummy_RGB_lr[:,2,:,:] = lr_images[:,0,:,:]
                    dummy_RGB_hr = np.ndarray(shape=(hr_images.shape[0],3, hr_images.shape[2],  hr_images.shape[3]), dtype= np.float32)
                    dummy_RGB_hr[:,0,:,:] = hr_images[:,0,:,:]
                    dummy_RGB_hr[:,1,:,:] = hr_images[:,0,:,:]
                    dummy_RGB_hr[:,2,:,:] = hr_images[:,0,:,:] 

                    lr_array = torch.from_numpy(dummy_RGB_lr).to(device)
                    hr_array = torch.from_numpy(dummy_RGB_hr).to(device)

                with torch.no_grad():
                    predicted_hr = self.model(lr_array)
                
                predicted_hr = data_normalization.denormalize_numpy(predicted_hr,mean,std)

                hr_array = data_normalization.denormalize_numpy(hr_array,mean,std)

                lr_array = data_normalization.denormalize_numpy(lr_array,mean,std)
                
                mae_sr = 0
                rmse_sr = 0 
                mae_b = 0
                rmse_b = 0
                absolute_median_sr = 0
                median_sr = 0 
                nmad_sr = 0 
                absolute_median_b = 0
                median_b = 0 
                nmad_b = 0 

                for i in range(hr_array.shape[0]):
        
                    residuals = np.squeeze(predicted_hr[i],axis=0) - np.squeeze(hr_array[i],axis=0)
                    #residuals = predicted_hr[i] - hr_array[i] # esrgan 3 channel
                
                    # Compute absolute residual errors
                    abs_residuals = np.ma.abs(residuals)
                    
                    # Mean absolute error (MAE)
                    mae = np.ma.mean(abs_residuals)
                    
                    # Root mean square error (RMSE)
                    rmse = np.ma.sqrt(np.ma.mean(abs_residuals ** 2))
                    
                    # Median absolute error
                    absolute_median = np.ma.median(abs_residuals)

                    # Median error
                    median = np.ma.median(residuals)

                    # Normalized median absolute deviation (NMAD)
                    abs_diff_from_med = np.ma.abs(residuals - absolute_median)
                    NMAD = 1.4826 * np.ma.median(abs_diff_from_med)

                    if self.args.netG == "pix2pix" or self.args.netG == "pix2pixhd" or self.args.netG == "enc_srgan":
                        
                        residuals_bc = np.squeeze(hr_array[i],axis=0) - np.squeeze(lr_array[i],axis=0)
                    else:               
                        bicubic = cv2.resize(np.squeeze(lr_array[i],axis=0),(int(self.tile_size),int(self.tile_size)),cv2.INTER_CUBIC)
                        residuals_bc = np.squeeze(hr_array[i],axis=0) - bicubic

                        #esrgan 3 channel
                        # bicubic = cv2.resize(lr_array[i].transpose(1, 2, 0),(int(self.tile_size),int(self.tile_size)),cv2.INTER_CUBIC)
                        
                        # residuals_bc = hr_array[i] - bicubic.transpose(2,1,0)
                
                    # Compute absolute residual errors
                    abs_residuals_bc = np.ma.abs(residuals_bc)
                    
                    # Mean absolute error (MAE)
                    mae_bc = np.ma.mean(abs_residuals_bc)
                    
                    # Root mean square error (RMSE)
                    rmse_bc = np.ma.sqrt(np.ma.mean(abs_residuals_bc ** 2))

                    absolute_median_bc = np.ma.median(abs_residuals_bc)

                    # Median error
                    median_bc = np.ma.median(residuals_bc)

                    # Normalized median absolute deviation (NMAD)
                    abs_diff_from_med_bc = np.ma.abs(residuals_bc - absolute_median_bc)
                    NMAD_bc = 1.4826 * np.ma.median(abs_diff_from_med_bc)

                    mae_b +=mae_bc
                    rmse_b += rmse_bc
                    absolute_median_b +=absolute_median_bc
                    median_b += median_bc
                    nmad_b += NMAD_bc
                    mae_sr += mae
                    rmse_sr += rmse
                    absolute_median_sr += absolute_median
                    median_sr += median
                    nmad_sr += NMAD
    
                val_rmse_sr += rmse_sr/hr_array.shape[0]
                val_mae_sr += mae_sr /hr_array.shape[0]
                val_mae_bc += mae_b /hr_array.shape[0]
                val_rmse_bc += rmse_b/ hr_array.shape[0]
                
                val_absolute_median_sr += absolute_median_sr/hr_array.shape[0]
                val_median_sr += median_sr/hr_array.shape[0]
                val_nmad_sr += nmad_sr/hr_array.shape[0]
                val_absolute_median_bc += absolute_median_b/hr_array.shape[0]
                val_median_bc += median_b/hr_array.shape[0]
                val_nmad_bc += nmad_b/hr_array.shape[0]

            print("SR RMSE: ",val_rmse_sr/num_val_batches)
            print("SR MAE: ",val_mae_sr/num_val_batches)
            print("SR Absolute median: ",val_absolute_median_sr/num_val_batches)
            print("SR Median: ", val_median_sr/num_val_batches)
            print("SR NMAD: ", val_nmad_sr/num_val_batches)
            print("Bicubic RMSE: ",val_rmse_bc/num_val_batches)
            print("Bicubic MAE: ",val_mae_bc/num_val_batches)
            print("Bicubic Absolute median: ",val_absolute_median_bc/num_val_batches)
            print("Bicubic Median: ", val_median_bc/num_val_batches)
            print("Bicubic NMAD: ", val_nmad_bc/num_val_batches)
            
    def visualize_results(self):

        with open(self.args.test_hr_path,"r") as hr:
            hr_files = hr.readlines()[:10]  ## 10 for dsm files
        
        with open(self.args.test_rgb_hr_path,"r") as rgb_hr:
            rgb_hr_files = rgb_hr.readlines()[:10] ## 10 for dsm files

        for index in range(len(hr_files)):
            
            xoffset = int(hr_files[index].split(",")[1])
            yoffset = int(hr_files[index].split(",")[2])
            
            file_name = hr_files[index].split(",")[0]
            hr = rasterio.open(file_name) 
            
            window = Window(xoffset,yoffset,self.tile_size,self.tile_size)
            temp_profile = hr.profile.copy()  
            transform = hr.window_transform(window)
            temp_profile.update({
                    'height':self.tile_size,
                    'width': self.tile_size,
                    'transform': transform
                    })

            hr_array =hr.read(1,window = window)

            rgb_hr_file_name = rgb_hr_files[index].split(",")[0]
            rgb_hr = rasterio.open(rgb_hr_file_name)
            rgb_hr_window = Window(xoffset,yoffset,self.tile_size,self.tile_size)
            rgb_hr_array = rgb_hr.read(window = rgb_hr_window)
            rgb_profile = rgb_hr.profile.copy()  
            rgb_transform = rgb_hr.window_transform(rgb_hr_window)
            rgb_profile.update({
                    'height':self.tile_size,
                    'width': self.tile_size,
                    'transform': rgb_transform
                    })

            lr_array = cv2.resize(hr_array,(int(self.tile_size*self.scale),int(self.tile_size*self.scale)),cv2.INTER_CUBIC)
            lr_2 = cv2.resize(lr_array,(int(self.tile_size*1/2),int(self.tile_size*1/2)),cv2.INTER_CUBIC)
            if self.args.netG == "pix2pix" or self.args.netG == "pix2pixhd" or self.args.netG == "srvae" or self.args.netG == "srcnn_rgb" or self.args.netG == "enc_srgan" :
                lr_array = cv2.resize(lr_array,(int(self.tile_size),int(self.tile_size)),cv2.INTER_CUBIC)

            mean = np.mean(lr_array)
            normalize = data_normalization.get_transform(mean,self.std)
            lr_array = normalize(lr_array)
            self.lr_array = torch.unsqueeze(lr_array,0)
            hr_array_2 = normalize(hr_array)
            hr_array_2 = torch.unsqueeze(hr_array_2,0)
            
            rgb_img_transform = transforms.Compose([
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])
            rgb_hr_array_norm = rgb_img_transform(torch.FloatTensor(rgb_hr_array))   
            rgb_hr_array_norm = torch.unsqueeze(rgb_hr_array_norm,0)

            if self.args.netG == "esrgan" : #and self.args.use_pretrained:                
                lr_images = self.lr_array.cpu().detach().numpy()
                # hr_images = hr_array.cpu().detach().numpy()
                dummy_RGB_lr = np.ndarray(shape=(lr_images.shape[0],3, lr_images.shape[2], lr_images.shape[3]), dtype= np.float32)
                dummy_RGB_lr[:,0,:,:] = lr_images[:,0,:,:]
                dummy_RGB_lr[:,1,:,:] = lr_images[:,0,:,:]
                dummy_RGB_lr[:,2,:,:] = lr_images[:,0,:,:]
                # dummy_RGB_hr = np.ndarray(shape=(hr_images.shape[0],3, hr_images.shape[2],  hr_images.shape[3]), dtype= np.float32)
                # dummy_RGB_hr[:,0,:,:] = hr_images[:,0,:,:]
                # dummy_RGB_hr[:,1,:,:] = hr_images[:,0,:,:]
                # dummy_RGB_hr[:,2,:,:] = hr_images[:,0,:,:] 

                self.lr_array = torch.from_numpy(dummy_RGB_lr).to(device)
                # hr_array = torch.from_numpy(dummy_RGB_hr).to(device)

            with torch.no_grad():
                
                predicted_hr = self.model(self.lr_array.to(device))
                # esrgan 3 channel
                # predicted_hr = predicted_hr[:,0,:,:]
                # self.lr_array = self.lr_array[:,0,:,:]
                    
            
            predicted_hr = data_normalization.denormalize_numpy(torch.squeeze(predicted_hr,0),mean,self.std)
            # esrgan 3 channel
            #predicted_hr = data_normalization.denormalize_numpy(predicted_hr,mean,self.std)

            if not self.args.netG == "pix2pix" or self.args.netG == "pix2pixhd" or self.args.netG == "srvae" or self.args.netG == "srcnn_rgb" or self.args.netG == "enc_srgan":
                lr_profile = temp_profile.copy()
                t = transform
                transform = Affine(t.a /self.scale, t.b, t.c, t.d, t.e /self.scale, t.f)
                height = self.tile_size * self.scale
                width = self.tile_size * self.scale
                lr_profile.update(transform=transform, driver='GTiff', height=height, width=width, crs=hr.crs)
                lr_out_profile = lr_profile.copy()

            lr_array = data_normalization.denormalize_numpy(torch.squeeze(self.lr_array,0),mean,self.std)

            # esrgan 3 channel
            #lr_array = data_normalization.denormalize_numpy(self.lr_array,mean,self.std)
            

            sr_filename = f"{file_name.split('/')[-1].rsplit('.',1)[0]}_sr.{file_name.rsplit('.',1)[-1]}"
            sr_path = os.path.join(args.output_dir, sr_filename)
            with rasterio.open(sr_path, 'w', **temp_profile) as dst:
                dst.write(predicted_hr)
            
            hr_filename = f"{file_name.split('/')[-1].rsplit('.',1)[0]}_hr.{file_name.rsplit('.',1)[-1]}"
            hr_path = os.path.join(args.output_dir, hr_filename)
            with rasterio.open(hr_path, 'w', **temp_profile) as dst:
                dst.write(np.expand_dims(hr_array,axis=0))

            if self.args.netG =="srcnn_rgb":
                rgb_filename = "rgb_file.tif"
                rgb_path = os.path.join(args.output_dir, rgb_filename)
                with rasterio.open(rgb_path, 'w', **rgb_profile) as dst:
                    dst.write(rgb_hr_array)

            if self.args.netG == "pix2pix" or self.args.netG == "pix2pixhd" or self.args.netG == "srcnn_rgb" or self.args.netG == "enc_srgan":
                bc_filename = f"{file_name.split('/')[-1].rsplit('.',1)[0]}_bc.{file_name.rsplit('.',1)[-1]}"
                input_path = os.path.join(args.output_dir, bc_filename)
                with rasterio.open(input_path, 'w', **temp_profile) as dst:
                    dst.write(lr_array)
            else:
                input_filename = f"{file_name.split('/')[-1].rsplit('.',1)[0]}_lr.{file_name.rsplit('.',1)[-1]}"
                input_path = os.path.join(args.output_dir, input_filename)
                with rasterio.open(input_path, 'w', **lr_out_profile) as dst:
                    dst.write(lr_array)
        
        return sr_path, hr_path, input_path
    

if __name__ =="__main__":

    parser = argparse.ArgumentParser("Testing of models")

    parser.add_argument(
        "--gen_weights",type=str,required=True, help="Assigns the folder for storing checkpoints"
    )
    parser.add_argument(
        "--gen_weights_2",type=str,required=True, help="Assigns the folder for storing checkpoints"
    )
    parser.add_argument(
        "--test_hr_path",type=str, required=True, help="paht to the test high resolution DSMs directory"
    )
    parser.add_argument(
        "--test_rgb_hr_path",type=str, required=True, help="paht to the rgb test high resolution DSMs directory"
    )
    parser.add_argument(
        "--train_rgb_hr_path",type=str, required=True, help="paht to the rgb test high resolution DSMs directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to save the tif files"
    )
    parser.add_argument(
        "--num_samples", type=int, required=False, default=200, help= "num of files to use for test set"
    )
    parser.add_argument(
        "--norm_params", type=str, required=False, default="./train_norm_parameters.txt", help= "normalization parameters"
    )
    parser.add_argument(
        "--netG",type=str,required=True,help="choose the generator of the network"
    )
    parser.add_argument(
        "--train_hr_path",type=str, required=True,help="paht to the train high resolution DSMs directory"
    )
    parser.add_argument(
        "--down_scale",type=int,default=4,required=True,help="diwnsample scale for the dataloader"
    )

    args = parser.parse_args() 

    if not os.path.isdir(args.output_dir):
       os.makedirs(args.output_dir)

    inf = Model_eval(args)
    inf.test()
    sr_path, hr_path,input_path = inf.visualize_results()