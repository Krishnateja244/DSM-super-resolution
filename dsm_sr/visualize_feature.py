import torch 
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from generator import SrganGenerator
from effnetv2 import effnetv2_s
from srgan_resca import SRCAGAN
from rasterio.windows import Window
import rasterio
import cv2
from data_normalization import denormalize_torch, denormalize_numpy
import data_normalization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model= SrganGenerator(1,128).to(device)
model = effnetv2_s().to(device)

pretrained_weights = "/home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/unet_hr_pretrain/model_best.pt"
model.load_state_dict(torch.load(pretrained_weights)["g_model"])


# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []# get all the model children as list
model_children = list(model.children())#counter to keep count of the conv layers]
print(model_children)
counter = 0 #append all the conv layers and their respective wights to the list

for i in range(len(model_children[0])):
    # print(type(model_children[i]))
    if type(model_children[i]) == torch.nn.modules.container.Sequential:
        for j in range(len(model_children[i])):
            if type(model_children[i][j]) == torch.nn.modules.conv.Conv2d:
                counter+=1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])

    else:
        seq = model_children[i].conv
        if type(seq) == torch.nn.modules.container.Sequential:
            for j in range(len(seq)):
                if type(seq[j]) == torch.nn.modules.conv.Conv2d:
                    counter+=1
                    model_weights.append(seq[j].weight)
                    conv_layers.append(seq[j])

    
print(f"Total convolution layers: {counter}")
# print(model_weights)
# print(conv_layers)

test_hr_path = "../datasets/swiss_dsm/testset/hr_256_files.txt"
with open(test_hr_path,"r") as hr:
    hr_files = [hr.readlines()[10]]
    
    for index in range(0,len(hr_files)):
        print(index)
        xoffset = int(hr_files[index].split(",")[1])
        yoffset = int(hr_files[index].split(",")[2])
        
        # lr = np.squeeze(reshape_as_image(lr_array),axis=2)
        file_name = hr_files[index].split(",")[0]
        hr = rasterio.open(file_name) 
        
        window = Window(xoffset,yoffset,256,256)
        temp_profile = hr.profile.copy()  
        transform = hr.window_transform(window)

        temp_profile.update({
                'height':256,
                'width': 256,
                'transform': transform
                })

        hr_array = hr.read(1, window = window)

        # hr_array =torch.tensor(hr_array).unsqueeze(0)
        # lr_transform= T.Resize(size=(int(self.tile_size*self.scale),int(self.tile_size*self.scale)),interpolation=InterpolationMode.NEAREST)
        # lr_array = lr_transform(hr_array)
        # lr_array = np.squeeze(lr_array.numpy(),axis=0)
        # hr_array = np.squeeze(hr_array,axis=0)
        lr_array = cv2.resize(hr_array,(int(256*1/2),int(256*1/2)),cv2.INTER_CUBIC)

        mean = np.mean(lr_array)
        normalize = data_normalization.get_transform(mean,10.25)
        lr_array = normalize(lr_array)
        lr_array = torch.unsqueeze(lr_array,0)

print(lr_array.shape)
outputs = []
names = []
image = lr_array.to(device)
for layer in conv_layers[:-2]:
    print(layer)
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))

print(len(outputs))#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())

for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split(',')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')