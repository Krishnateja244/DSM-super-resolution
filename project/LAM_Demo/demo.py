import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt
from PIL import Image

from ModelZoo.utils import load_as_tensor, Tensor2PIL, PIL2Tensor, _add_batch_one
from ModelZoo import get_model, load_model, print_network
from SaliencyModel.utils import vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, grad_norm, prepare_images, make_pil_grid, blend_input
from SaliencyModel.utils import cv2_to_pil, pil_to_cv2, gini
from SaliencyModel.attributes import attr_grad
from SaliencyModel.BackProp import I_gradient, attribution_objective, Path_gradient
from SaliencyModel.BackProp import saliency_map_PG as saliency_map
from SaliencyModel.BackProp import GaussianBlurPath
from SaliencyModel.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
from dataset_loader import CustomDataset, DataLoader
import rasterio
from rasterio import Affine
from rasterio.windows import Window
import data_normalization


model = load_model('enc_srgan@Base')  # You can Change the model name to load different models

window_size = 16  # Define windoes_size of D

    
hr_path = './test_images/swisssurface3d-raster_2018_2696-1263_0.5_2056_5728_hr.tif'### for dsm 
# hr_path = '/home/nall_kr/Documents/sr_dsm/D-SRGAN/LAM_Demo/test_images/swissalti3d_2020_2703-1261_0.5_2056_5728_hr.tif' ## for dtm 
# hr_path = '/home/nall_kr/Documents/sr_dsm/D-SRGAN/LAM_Demo/test_images/rgb_file.tif'

hr_im = rasterio.open(hr_path)
hr_array = hr_im.read(1)

lr_array = cv2.resize(hr_array,(int(256*1/4),int(256*1/4)),cv2.INTER_CUBIC)

std = 10.2
mean = np.mean(lr_array)
data_norm = data_normalization.get_transform(mean,std)
lr_array = data_norm(lr_array)
hr_array= data_norm(hr_array)

print(lr_array.shape)
img_lr, img_hr = prepare_images('./test_images/hr_dsm_single.png',scale=4)  # Change this image name
tensor_lr = PIL2Tensor(img_lr)[:3]  
tensor_hr = PIL2Tensor(img_hr)[:3]



print(tensor_lr.shape)
print(tensor_hr.shape)
cv2_lr = np.moveaxis(tensor_lr.numpy(), 0, 2) ; cv2_hr = np.moveaxis(tensor_hr.numpy(), 0, 2)
print(cv2_lr.shape)
w = 90  # The x coordinate of your select patch, 125 as an example
h = 100  # The y coordinate of your select patch, 160 as an example
         # And check the red box
         # Is your selected patch this one? If not, adjust the `w` and `h`.


draw_img = pil_to_cv2(img_hr)
print(type(draw_img)) 

cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil = cv2_to_pil(draw_img) # we get pil type image
print(type(position_pil))

draw_img_lr = pil_to_cv2(img_lr.resize(img_hr.size))
print(type(draw_img_lr)) 

cv2.rectangle(draw_img_lr, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
position_pil_lr = cv2_to_pil(draw_img_lr) # we get pil type image
print(type(position_pil_lr))

sigma = 1.2 ; fold = 50 ; l = 9 ; alpha = 0.6
attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
tensor_lr = tensor_lr.detach().cpu().numpy()
lr_array = lr_array.detach().cpu().numpy() ##(kt)
interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(lr_array, model, attr_objective, gaus_blur_path_func, cuda=True)
grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
abs_normed_grad_numpy = grad_abs_norm(grad_numpy)

print(type(abs_normed_grad_numpy))
saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy,zoomin=4)
# blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size,refcheck=False)) * alpha)
# blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size,refcheck=False)) * alpha)
blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs)* (1.0 - alpha)+ pil_to_cv2(img_lr.resize(img_hr.size))* alpha)
blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) *(1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size))* alpha)
gini_index = gini(abs_normed_grad_numpy)
diffusion_index = (1 - gini_index) * 100
print(f"The DI of this case is {diffusion_index}")
# result = data_normalization.denormalize_torch(result,mean,std)
pil = make_pil_grid(
    [position_pil,
     position_pil_lr,
     saliency_image_abs,
     blend_abs_and_input,
     blend_kde_and_input])
    #  Tensor2PIL(result,mode="L")])#,

pil.save("enc_srgan.png")