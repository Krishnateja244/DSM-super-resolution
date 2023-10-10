import torch
from ignite.metrics import PSNR,SSIM
from ignite.engine import Engine


def total_variation(image_in):
    tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
    tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
    tv_loss = tv_h + tv_w
    return tv_loss 

def TV_loss(im_batch, weight):
    TV_L = 0.0
    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx,0,:,:])
    TV_L = TV_L/len(im_batch)
    return weight*TV_L


def eval_step(engine, batch): 
    return batch 
def calculate_error(img1,img2):
    default_evaluator = Engine(eval_step)
    # calculating PSNR 
    y1 = img1.to("cuda")
    y2 = img2.to("cuda")
    psnr = PSNR(data_range=1)
    psnr.attach(default_evaluator,'psnr')
    state1 = default_evaluator.run([[y1,y2]])
    # print(f"PSNR :{state1.metrics['psnr']}")

    # # calculating SSIM
    y1,y2 = torch.tensor(y1,dtype=torch.float32),torch.tensor(y2,dtype=torch.float32)
    metric = SSIM(data_range=1)
    metric.attach(default_evaluator, 'ssim')
    state2 = default_evaluator.run([[y1, y2]])


    # print(f"SSIM : {state2.metrics['ssim']}")
    return state1.metrics['psnr'],state2.metrics['ssim']    
