import numpy as np
import torch
from torchvision import transforms


def get_transform(mean, std):
    """Function performs the normalization of data using mean and std

    Args:
        mean (float): mean of dataset
        std (float): standard deviation of dataset

    Returns:
        transform object
    """
    if isinstance(mean, np.ndarray):
        mean = mean.tolist()
    if isinstance(mean, torch.Tensor):
        mean = mean.tolist()
    elif type(mean) is not list:
        mean = [mean]

    if isinstance(std, np.ndarray):
        std = std.tolist()
    if isinstance(std, torch.Tensor):
        std = std.tolist()
    elif type(std) is not list:
        std = [std]

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return data_transform


def denormalize_torch(data, mean, std):
    """Function performs denormalization of data

    Args:
        data (tensor): dataset batch
        mean (float): Mean of dataset
        std (float): standard deviation of the dataset

    Returns:
        denormalized tensor array
    """
    if isinstance(std, torch.Tensor) or isinstance(std, list):
        data_denorm = data.clone()

        for i, (mean_i, std_i) in enumerate(zip(mean.tolist(), std.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :] * std_i) + mean_i
    else:
        data_denorm = (data * std) + mean

    return data_denorm


def denormalize_numpy(data, mean, std):
    """Function performs denormalization of data

    Args:
        data (numpy): dataset batch
        mean (float): Mean of dataset
        std (float): standard deviation of the dataset

    Returns:
        denormalized numpy array
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    if isinstance(std, np.ndarray) or isinstance(std, torch.Tensor) or isinstance(std, list):
        data_denorm = np.zeros_like(data)

        for i, (mean_i, std_i) in enumerate(zip(mean.tolist(), std.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :] * std_i) + mean_i
    else:
        data_denorm = (data * std) + mean

    return data_denorm

def denormalize_numpy_min_max(data,max,min):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    if isinstance(min, np.ndarray) or isinstance(min, torch.Tensor) or isinstance(min, list):
        data_denorm = np.zeros_like(data)

        for i, (max_i, min_i) in enumerate(zip(max.tolist(), min.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :]*(max_i-min_i)) + min_i
    else:
        data_denorm = (data * (max-min)) + min_i
    return data_denorm

def denormalize_torch_min_max(data, max, min):
    if isinstance(min, torch.Tensor) or isinstance(min, list):
        data_denorm = data.clone()

        for i, (max_i, min_i) in enumerate(zip(max.tolist(), min.tolist())):
            data_denorm[i, :, :, :] = (data[i, :, :, :] *(max_i-min_i)) + min_i
    else:       
        data_denorm = (data *(max-min)) + min

    return data_denorm

def normalize_min_max(data,max,min):
    norm_data = (data-min)/(max-min)
    return torch.tensor(norm_data)

def denormalize_numpy_minus(data,max,min):
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()

    if isinstance(min, np.ndarray) or isinstance(min, torch.Tensor) or isinstance(min, list):
        data_denorm = np.zeros_like(data)

        for i, (max_i, min_i) in enumerate(zip(max.tolist(), min.tolist())):
            data_denorm[i, :, :, :] = ((data[i, :, :, :]+1)*(max_i-min_i))/2 + min_i
    else:
        data_denorm = ((data+1)*(max_i-min_i))/2 + min_i
    return data_denorm

def denormalize_torch_minus(data, max, min):
    if isinstance(min, torch.Tensor) or isinstance(min, list):
        data_denorm = data.clone()

        for i, (max_i, min_i) in enumerate(zip(max.tolist(), min.tolist())):
            data_denorm[i, :, :, :] = ((data[i, :, :, :]+1)*(max_i-min_i))/2 + min_i
    else:       
        data_denorm = ((data+1)*(max_i-min_i))/2 + min_i

    return data_denorm

def normalize_minus(data,min,max):
    norm_data = (2*(data-min)/(max-min))-1
    return torch.tensor(norm_data)
