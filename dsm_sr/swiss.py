# Original Code from: https://github.com/prs-eth/graph-super-resolution
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from swiss_utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip
import os
from PIL import Image
from torchvision import transforms
SWISS_BASE_SIZE = (256, 256)
from torch.utils.data import Dataset , DataLoader


class SwissDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            resolution='HR',
            scale=1.0,
            crop_size=(256, 256),
            do_horizontal_flip=False,
            max_rotation_angle: int = 15,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            in_memory=True,
            split='train',
            crop_valid=False,
            crop_deterministic=False,
            scaling=8,
            std = 0
    ):
        self.scale = scale
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.scale_interpolation = scale_interpolation
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling
        data_dir = Path(data_dir)

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        

        if split == 'train':
            train_image_list = str(data_dir / f'train.txt')
            image_dir = str(data_dir / f'train/dsm')
            rgb_image_dir = str(data_dir / f'train/rgb')
            image_paths, rgb_image_paths = self.get_paths(train_image_list, image_dir, rgb_image_dir)
            self.image_paths = image_paths
            self.rgb_image_paths = rgb_image_paths
            # self.image_dir = args.image_dir
            # self.images_name = self.all_images_name[:28000]
            #self.labels_attr = self.all_labels_attr[:28000]

        elif split == 'test':
            val_image_list = str(data_dir / f'test.txt')
            image_val_dir = str(data_dir / f'test/dsm')
            rgb_image_val_dir = str(data_dir / f'test/rgb')
            image_paths, rgb_image_paths = self.get_paths(val_image_list, image_val_dir, rgb_image_val_dir)
            self.image_paths = image_paths
            self.rgb_image_paths = rgb_image_paths
            # self.image_dir = args.image_val_dir
            # self.images_name = self.all_images_name[28000:]
            #self.labels_attr = self.all_labels_attr[28000:]

        else:
            print("Error")

        #mmap_mode = None if in_memory else 'c'

        #self.images = np.load(str(data_dir / f'npy/images_{split}_{resolution}.npy'), mmap_mode)
        #self.depth_maps = np.load(str(data_dir / f'npy/depth_{split}_{resolution}.npy'), mmap_mode)
        #assert len(self.images) == len(self.depth_maps)
        if not split =="train":
          self.global_std = std
        else:
          self.global_std = self.calculate_global_std()
          
        self.H, self.W = int(SWISS_BASE_SIZE[0] * self.scale), int(SWISS_BASE_SIZE[1] * self.scale)

        if self.crop_valid:
            if self.max_rotation_angle > 45:
                raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')

            # make sure that max rotation angle is valid, else decrease
            max_angle = np.floor(min(
                2 * np.arctan
                    ((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
                2 * np.arctan
                    ((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
            ) * (180. / np.pi))

            if self.max_rotation_angle > max_angle:
                print(f'max rotation angle too large for given image size and crop size, decreased to {max_angle}')
                self.max_rotation_angle = max_angle
    def get_paths(self, image_list, image_dir, rgb_image_dir):
        names = open(image_list).readlines()
        filenames = list(map(lambda x: x.strip('\n')+'.tif', names))
        rgb_filenames = list(map(lambda x: x.strip('\n')+'.png', names))
        image_paths = list(map(lambda x: os.path.join(image_dir, x), filenames))
        rgb_image_paths = list(map(lambda x: os.path.join(rgb_image_dir, x), rgb_filenames))
        return image_paths, rgb_image_paths
    
    def calculate_global_std(self):
        std_list = []
        for index in range(len(self.image_paths)):
            img = Image.open(self.image_paths[index])
            img_np = np.array(img)  # Convert image to numpy array
            std = np.std(img_np)  # Calculate standard deviation
            std_list.append(std)

        global_std = np.mean(std_list)  # Calculate global standard deviation
        print(f"STD: {global_std}")
        return global_std
        # Create the denormalization transform


    def __getitem__(self, index):
        depth_map = Image.open(self.image_paths[index])
        depth_map_mean = np.mean(depth_map)

        rgb_image = Image.open(self.rgb_image_paths[index])
        rgb_img = rgb_image.convert('RGB')


        self.depth_map_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=depth_map_mean,
                                                        std=self.global_std)
                                    ])
        

        self.rgb_img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                ])
        
        



        #att = torch.tensor((self.labels_attr[index] + 1) // 2)
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = index // (num_crops_h * num_crops_w)
        else:
            im_index = index

        image = self.rgb_img_transform(rgb_img)
        depth_map = self.depth_map_transform(depth_map)
        
        
        #resize = Resize((self.H, self.W), self.scale_interpolation)
        #image, depth_map = resize(image), resize(depth_map)

        # # apply user transforms
        # if self.image_transform is not None:
        #     image = self.image_transform(image)
        # if self.depth_transform is not None:
        #     depth_map = self.depth_transform(depth_map)

        source = downsample(depth_map.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_lr = (~torch.isnan(source)).float()

        depth_map[depth_map == -9999.] = 0.
        source[source == -9999.] = 0.        

        mask_hr[depth_map == -9999.] = 0.
        mask_lr[source == -9999.] = 0.    

        y_bicubic = torch.from_numpy(
            bicubic_with_mask(source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
        y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))

        #return {'guide': image, 'y': depth_map, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
        #        'y_bicubic': y_bicubic}
        filename="st"
        return y_bicubic,image,depth_map,depth_map_mean,self.global_std,filename

    def __len__(self):
        if self.crop_deterministic:
            return len(self.image_paths) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.image_paths)
        
if __name__ == "__main__":
    data_dir = "../datasets/"
    train_dataset = SwissDataset(data_dir,scale_interpolation=InterpolationMode.BICUBIC, split='train')
      
                
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    std = next(iter(train_loader))[4]
    print(std)
    test_dataset = SwissDataset(data_dir,scale_interpolation=InterpolationMode.BICUBIC, split='test',std =std)
                          
    test_loader = DataLoader(test_dataset,batch_size=4,shuffle=True) 
    test_batch = next(iter(train_loader))
    print(test_batch[0].shape)
    print(test_batch[1].shape)
    print(test_batch[2].shape)
    
    print(test_batch[3])
    print(test_batch[4])
    print(test_batch[4])
