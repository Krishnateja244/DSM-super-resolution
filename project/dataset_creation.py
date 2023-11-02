import rasterio
import numpy as np 
import torch
from rasterio.plot import reshape_as_image, reshape_as_raster, show
import rasterio
from rasterio import Affine
from rasterio.windows import Window
import os
import sklearn
import skimage 
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd
import random
import math
import cv2
from interpolation import Interpolation

def rasterio_saver(hr_path,image_size,txt_name,no_images):
    """Function performs the creation of "hr_256_files.txt" for the training and testing dataset files. 
        2000*2000 high resolution elevation tif files are cropped to 256*256 patch

    Args:
        hr_path (path): path to directory containing original hr files
        image_size (int): size of image patch
        txt_name (path): path to store hr_256_files.txt file
        no_images (int): number of images for the dataset
    """
    
    hr_files_list = [file for file in os.listdir(hr_path) if file.split(".")[-1] == "tif"][:no_images]

    with open(txt_name,"w") as txt_file:
        for file in hr_files_list:
            tif_path = os.path.join(hr_path,file)
            with rasterio.open(tif_path) as src:

                xsize, ysize = image_size,image_size

                profile = src.profile
                temp_profile = profile.copy()
                count =0
                for r in range(0,src.width-xsize+1, xsize):
                    for c in range(0,src.height-ysize+1, ysize):
                        xoff,yoff = r,c
  
                        window = Window(xoff, yoff, xsize, ysize)
                
                        transform = src.window_transform(window)

                        temp_profile.update({
                            'height': xsize,
                            'width': ysize,
                            'transform': transform})
                        
                        txt_file.write(f"{tif_path},{str(xoff)},{str(yoff)}\n")
                        # hr_array = src.read(window=window)
                        # file_name = f"{file.rsplit('.',1)[0]}_{count}.{file.rsplit('.',1)[-1]}"
                        
                        # # print(file_name)
                        # save_file = os.path.join(cropped_path,file_name) #file
                        # txt_file.write(f"{save_file},{str(xoff)},{str(yoff)}\n")
                        # out_profiles.append(profile)
                        # with rasterio.open(save_file, 'w', **temp_profile) as dst:
                        #     dst.write(hr_array)
                        count +=1



def downsample_nn(hr_path,lr_path,scale):
    """Function performs saving the nearest neighbour downsampled high resolution files to directory

    Args:
        hr_path (path): path to directory containing original hr files
        lr_path (path): path to store the downsapled hr images
        scale (float): downsapling scale
    """

    hr_files_list = [file for file in os.listdir(hr_path) if file.split(".")[-1] == "tif"]

    for file in hr_files_list:
        tif_path = os.path.join(hr_path,file)
        with rasterio.open(tif_path) as src:
            hr_array = src.read(
                out_shape=(
                    src.count,
                    int(src.height),
                    int(src.width)
                )
            )
            profile = src.profile
        
        t = src.transform
        transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
        height = src.height * scale
        width = src.width * scale
        profile.update(transform=transform, driver='GTiff', height=height, width=width, crs=src.crs)
        out_profile = profile.copy()
        lr_array = reshape_as_image(hr_array)            

        # interp = Interpolation(image=lr_array,num_channels=1, scale=scale)

        resized_image_nn = cv2.resize(lr_array,(int(height),int(width)),cv2.INTER_NEAREST) #interp.nearest_neighbour() 

        image_sampled = np.expand_dims(resized_image_nn,axis=0)

        current_out = os.path.join(lr_path,file)
        with rasterio.open(current_out, 'w', **out_profile) as dst:
            dst.write(image_sampled)


if __name__ == "__main__":

    hr_path = "/home/nall_kr/Documents/sr_dsm/datasets/swiss_dtm/trainset/hr_files/"
    lr_path = "/home/nall_kr/Documents/sr_dsm/datasets/swiss_dtm/trainset/lr_files/"
    hr_cropped_path = "/home/nall_kr/Documents/sr_dsm/datasets/swiss_dtm/trainset/hr_files/"
    lr_cropped_path = "/home/nall_kr/Documents/sr_dsm/datasets/swiss_dtm/trainset/lr_files_cropped/"

    # if not os.path.isdir(hr_cropped_path):
    #     os.makedirs(hr_cropped_path)

    # if not os.path.isdir(lr_cropped_path):
    #     os.makedirs(lr_cropped_path)

    # if not os.path.isdir(lr_path):
    #     os.makedirs(lr_path)

    image_size = 256
    scale = 1/4
    txt_file = "/home/nall_kr/Documents/sr_dsm/datasets/swiss_dtm/trainset/hr_files_256.txt"
    num_images = None

    if not os.path.isdir(hr_cropped_path):
        os.makedirs(hr_cropped_path)
    rasterio_saver(hr_path,image_size,txt_file,hr_cropped_path,num_images)
    # downsample_nn(hr_path,lr_cropped_path,scale=scale)