import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from generator import SrganGenerator, Encoder
from discriminator import Discriminator, NLayerDiscriminator, UNetDiscriminatorSN
from torchvision.utils import make_grid
from dataset_loader import CustomDataset, compute_local_dsm_std_per_centered_patch
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import argparse
import data_normalization
from effnetv2 import effnetv2_s
from pix2pix import UnetGenerator
import torchvision.models as models
from torch.optim import lr_scheduler
from focal_frequency_loss import FocalFrequencyLoss as FFL 
from srgan_resca import SRCAGAN
import cv2
from ESRGAN import ESRGAN,RRDBNet
from basicsr.archs.rrdbnet_arch import RRDBNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import segmentation_models_pytorch as smp

class PerceptualLoss(nn.Module):
    def __init__(self):
      super(PerceptualLoss, self).__init__()
      vgg16 = models.vgg16(pretrained=True).to(device)
      # vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      loss_network = nn.Sequential(*list(vgg16.features)[:30]).eval().to(device)
      for param in loss_network.parameters():
          param.requires_grad = False
      self.loss_network = loss_network
      self.l1_loss = nn.L1Loss()
        
    def forward(self, predicted_images, hr_images):
      predicted_images = predicted_images.cpu().detach().numpy()
      hr_images = hr_images.cpu().detach().numpy()
      dummy_RGB_predicted = np.ndarray(shape=(predicted_images.shape[0],3, predicted_images.shape[2], predicted_images.shape[3]), dtype= np.float32)
      dummy_RGB_predicted[:,0,:,:] = predicted_images[:,0,:,:]
      dummy_RGB_predicted[:,1,:,:] = predicted_images[:,0,:,:]
      dummy_RGB_predicted[:,2,:,:] = predicted_images[:,0,:,:]
      dummy_RGB_hr = np.ndarray(shape=(hr_images.shape[0],3, hr_images.shape[2],  hr_images.shape[3]), dtype= np.float32)
      dummy_RGB_hr[:,0,:,:] = hr_images[:,0,:,:]
      dummy_RGB_hr[:,1,:,:] = hr_images[:,0,:,:]
      dummy_RGB_hr[:,2,:,:] = hr_images[:,0,:,:]
      
      perception_loss = self.l1_loss(self.loss_network(torch.tensor(dummy_RGB_predicted).to(device)), self.loss_network(torch.tensor(dummy_RGB_hr).to(device)))
      return perception_loss

class Model_train:

  def __init__(self,args):
    self.args = args

    config = {"learning_rate":self.args.lr_rate,"Architecture":self.args.netG,"Dataset":"Swiss_dataset","Epochs":self.args.epochs, \
              "Batch_size":self.args.batch_size,"lamda":self.args.lamda}

    wandb.init(project="sr-gan",name=self.args.checkpoint_name,config=config)

    self.tv_weight = 0.08 ## 0.0000005
    self.tile_size = 256
    self.num_eval_samples = 200 

    if self.args.netG == "srgan":
      print("using SRGAN model")
      self.generator =  SrganGenerator(1, 128,self.args.down_scale).to(device) 

    elif self.args.netG == "srgan_atten":
      print("using SRGAN RESIDUAL CHANNEL ATTENTION")
      self.generator = SRCAGAN(1,128,10,10,16).to(device) # original 20 atten blocks 

    elif self.args.netG =="esrgan":
      print("using ESRGAN model")
      self.generator =  RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device) #ESRGAN(1,1,scale_factor=self.args.down_scale).to(device) 

    elif self.args.netG == "srgan_colearn":
      print("using co-learning based srgan models")
      self.generator = SrganGenerator(1, 128,self.args.down_scale).to(device)
      self.generator_2x = SrganGenerator(1, 128,self.args.down_scale/2).to(device)

    elif self.args.netG == "netv2":
      print("Using Efficientnetv2 model")
      self.generator =  effnetv2_s().to(device)

    elif self.args.netG == "pix2pix":
      print("Using pix2pix(unet) generator")
      self.generator = UnetGenerator(1,1,8).to(device) #smp.Unet('resnet50', classes=1, in_channels=3).to(device) #
    
    elif self.args.netG == "enc_srgan":
      self.generator = Encoder(1,128).to(device)
    
    if self.args.use_pretrained:
        print(f"using pretrained model for {self.args.netG}")
        # we = torch.load(self.args.use_pretrained)
        # if 'params_ema' in we:
        #         keyname = 'params_ema'
        # else:
        #     keyname = 'params'
        # self.generator.load_state_dict(we[keyname], strict=True)
        self.generator.load_state_dict(torch.load(self.args.use_pretrained)["g_model"])
    else:
      self.init_weights(self.generator,self.args.init_type)
      if self.args.netG == "srgan_colearn":
        print("initalizing G2x network for colearning with %s" % self.args.init_type)
        self.init_weights(self.generator_2x,self.args.init_type)

    self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_rate, betas=(0.5, 0.999))
    self.scheduler_G = self.get_scheduler(self.optim_G)

    if self.args.netG == "srgan_colearn":
      print("intializing the adam optimizer for co-learning")
      self.optim_G_2x = torch.optim.Adam(self.generator_2x.parameters(), lr=self.args.lr_rate, betas=(0.5, 0.999))
      self.scheduler_G_2x = self.get_scheduler(self.optim_G_2x)
      
    # self.ffl = FFL(loss_weight=1.0, alpha=1.0)

    if self.args.gan_mode:
      print("Enabled discriminator training")
      if self.args.netD == "pixel":
        print("using pixel based discriminator")
        self.discriminator = Discriminator(1, 128).to(device)
      elif self.args.netD == "patch":
        print("using patch discriminator")
        self.discriminator = NLayerDiscriminator(1,64).to(device)
      elif self.args.netD == "unet":
        print("using unet disciminator")
        self.discriminator = UNetDiscriminatorSN(3,64).to(device)

      
      print('initialize D network with %s' % self.args.init_type)
      self.init_weights(self.discriminator,self.args.init_type)
      self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_rate, betas=(0.5, 0.999))
      self.scheduler_D = self.get_scheduler(self.optim_D)

      self.gan_loss = nn.BCEWithLogitsLoss()
      

    if self.args.l1_loss or self.args.netG == "srgan_colearn" or self.args.netG == "enc_srgan":
      self.L1_loss = torch.nn.L1Loss() #MSGIL_NORM_Loss(scale=2) #
      print("using L1 loss function")
    else:
      self.l2_loss = torch.nn.MSELoss()
      print("using L2 loss function")
    
    if self.args.label_smooth:
      print("performing label smoothing")

    if self.args.relat_gan:
      print("enabled relativistic calculation")
    
    if self.args.percep_loss:
      print("enabling perceptual loss")
      self.vgg_loss = PerceptualLoss()
    
    self.checkpoints_dir = f"./checkpoints/{self.args.checkpoint_name}"

    if not os.path.isdir(self.checkpoints_dir):
      os.makedirs(self.checkpoints_dir)

  def get_scheduler(self,optimizer):
    epoch_count = 1
    if self.args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - self.args.n_epochs) / float(self.args.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif self.args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.args.lr_decay_iters, gamma=0.1)
    elif self.args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif self.args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', self.args.lr_policy)
    return scheduler

  def update_learning_rate(self,optimizer,scheduler):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

  def get_dataloader(self):
      
      train_dataset = CustomDataset(hr_dir=self.args.train_hr_path, rgb_hr_dir=self.args.train_rgb_hr_path, \
                                  num_samples=self.args.no_train_samples,tile_size=self.tile_size,scale=1/self.args.down_scale,model=self.args.netG,transform=None)
      

      train_loader = DataLoader(train_dataset,batch_size=self.args.batch_size,shuffle=True)

      mean,std = compute_local_dsm_std_per_centered_patch(train_loader)

      print(f"Normalization parameters Mean: {mean}, Std :{std}")
      param_file = open("train_norm_parameters.txt","w")
      param_file.write(f"Mean,{mean}\n")
      param_file.write(f"Std,{std}")

      train_dataset = CustomDataset(hr_dir=self.args.train_hr_path,rgb_hr_dir=self.args.train_rgb_hr_path, \
                                  num_samples=self.args.no_train_samples,tile_size=self.tile_size,scale=1/self.args.down_scale,model=self.args.netG,transform=True,dsm_mean=None,dsm_std=std)

      train_loader = DataLoader(train_dataset,batch_size=self.args.batch_size,shuffle=True)

      test_dataset = CustomDataset(hr_dir=self.args.test_hr_path,rgb_hr_dir=self.args.test_rgb_hr_path, \
                                  num_samples=self.num_eval_samples,tile_size=self.tile_size,scale=1/self.args.down_scale,model=self.args.netG,transform=True,dsm_mean=None,dsm_std=std,test_set=False)

      test_loader = DataLoader(test_dataset,batch_size=self.args.batch_size,shuffle=True) 
       
      print("shape of input: ",next(iter(test_loader))[0].shape)
      print("shape of lr2 : ",next(iter(test_loader))[1].shape)
      print("shape of gt : ",next(iter(test_loader))[2].shape)
      print(("shape of rgb_array : ",next(iter(test_loader))[3].shape))
      return train_loader,test_loader

  def init_weights(self,net, init_type, init_gain=0.02):
    def init_func(m):  # define the initialization function
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
          if init_type == 'normal':
              init.normal_(m.weight.data, 0.0, init_gain)
          elif init_type == 'xavier':
              init.xavier_normal_(m.weight.data, gain=init_gain)
          elif init_type == 'kaiming':
              init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
          elif init_type == 'orthogonal':
              init.orthogonal_(m.weight.data, gain=init_gain)
          else:
              raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
          if hasattr(m, 'bias') and m.bias is not None:
              init.constant_(m.bias.data, 0.0)
      elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
          init.normal_(m.weight.data, 1.0, init_gain)
          init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>

  def total_variation(self,image_in):
        tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
        tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
        tv_loss = tv_h + tv_w
        return tv_loss 

  def TV_loss(self,im_batch, weight):
      TV_L = 0.0
      for tv_idx in range(len(im_batch)):
          TV_L = TV_L + self.total_variation(im_batch[tv_idx,0,:,:])
      TV_L = TV_L/len(im_batch)
      return weight*TV_L

  def train(self,train_loader,epoch):
    
    print(f"Epoch {epoch}: ", end ="")
    
    G_adv_loss = 0
    G_rec_loss = 0
    G_tot_loss = 0
    D_adv_loss = 0
    D_real_loss = 0
    D_fake_loss = 0
    G_percep_loss = 0
    TV_loss = 0
    CL_loss = 0 
    CL_loss_2x = 0
    G_rec_loss_2x = 0 
    G_tot_loss_2x = 0
    
    self.generator.train()

    if self.args.netG == "srgan_colearn":
      self.generator_2x.train()

    if self.args.gan_mode:
      self.discriminator.train()

    for batch, (lr,lr2, hr,rgb_hr,mean,std,file_name) in enumerate(train_loader):

      lr_images = lr.to(device)
      hr_images = hr.to(device)
      lr2 = lr2.to(device)
      rgb_hr = rgb_hr.to(device)

      if self.args.netG == "esrgan" and self.args.use_pretrained:
        lr_images = lr_images.cpu().detach().numpy()
        hr_images = hr_images.cpu().detach().numpy()
        dummy_RGB_lr = np.ndarray(shape=(lr_images.shape[0],3, lr_images.shape[2], lr_images.shape[3]), dtype= np.float32)
        dummy_RGB_lr[:,0,:,:] = lr_images[:,0,:,:]
        dummy_RGB_lr[:,1,:,:] = lr_images[:,0,:,:]
        dummy_RGB_lr[:,2,:,:] = lr_images[:,0,:,:]
        dummy_RGB_hr = np.ndarray(shape=(hr_images.shape[0],3, hr_images.shape[2],  hr_images.shape[3]), dtype= np.float32)
        dummy_RGB_hr[:,0,:,:] = hr_images[:,0,:,:]
        dummy_RGB_hr[:,1,:,:] = hr_images[:,0,:,:]
        dummy_RGB_hr[:,2,:,:] = hr_images[:,0,:,:] 

        lr_images = torch.from_numpy(dummy_RGB_lr).to(device)
        hr_images = torch.from_numpy(dummy_RGB_hr).to(device)

      if self.args.netG == "srgan_colearn":
        predicted_hr_images= self.generator(lr_images)
        predicted_hr_images_2x = self.generator_2x(lr2)

      elif self.args.netG == "enc_srgan":
        # lr_images = hr_images # comment when not pretraining hte model with hr images
        predicted_hr_images = self.generator(lr_images)
        
      else:
        predicted_hr_images = self.generator(lr_images) 

      if self.args.gan_mode:
        ############## Training Discriminator ###############################
        # training discriminator
        for p in self.discriminator.parameters():
          p.requires_grad = True
        self.optim_D.zero_grad()

        adv_hr_real = self.discriminator(hr_images)
        adv_hr_fake = self.discriminator(predicted_hr_images.detach())

        if self.args.relat_gan:
          d_real_loss = self.gan_loss(adv_hr_real-torch.mean(adv_hr_fake), torch.ones_like(adv_hr_real))
          d_fake_loss = self.gan_loss(adv_hr_fake-torch.mean(adv_hr_real), torch.zeros_like(adv_hr_fake))
        else:
          d_real_loss = self.gan_loss(adv_hr_real, torch.ones_like(adv_hr_real))

          d_fake_loss = self.gan_loss(adv_hr_fake, torch.zeros_like(adv_hr_fake))
        
        df_loss = (d_real_loss +d_fake_loss)*0.5
        
        D_adv_loss += df_loss.item()
        D_fake_loss +=d_fake_loss.item()
        D_real_loss +=d_real_loss.item()
        df_loss.backward()
        
        self.optim_D.step()

      ################# Training Generator $$$$$$$$$$$$$$$$$$$$$$$$
      if self.args.gan_mode:
        for p in self.discriminator.parameters():
          p.requires_grad = False
      
      self.optim_G.zero_grad()

      # reconstruction loss
      if self.args.netG == "srgan_colearn":
        self.optim_G_2x.zero_grad()
        gr_loss = self.L1_loss(predicted_hr_images, hr_images)
        gr_loss_2x = self.L1_loss(predicted_hr_images_2x, hr_images)       
        colearn_loss_2x = 0 #self.L1_loss(predicted_hr_images_2x,predicted_hr_images.detach())*self.args.lamda
        colearn_loss = self.L1_loss(predicted_hr_images,predicted_hr_images_2x.detach())*self.args.lamda

      elif self.args.l1_loss:
        gr_loss = self.L1_loss(predicted_hr_images, hr_images)*self.args.lamda # L1 loss
        tv_loss = self.TV_loss(predicted_hr_images,self.tv_weight)

      else:
        gr_loss = self.l2_loss(predicted_hr_images, hr_images)*self.args.lamda # L2 loss

      if self.args.gan_mode:
        predicted_hr_labels = self.discriminator(predicted_hr_images)

        if self.args.relat_gan:
          adv_hr_real = self.discriminator(hr_images.detach())
          gf_loss = (self.gan_loss(predicted_hr_labels-torch.mean(adv_hr_real), torch.ones_like(predicted_hr_labels)) \
            +self.gan_loss(adv_hr_real-torch.mean(predicted_hr_labels),torch.zeros_like(adv_hr_real)))/2
        else:
          gf_loss = self.gan_loss(predicted_hr_labels,torch.ones_like(predicted_hr_labels))

        if self.args.percep_loss:
          per_loss = self.vgg_loss(predicted_hr_images,hr_images)
          g_loss = gf_loss + gr_loss +(per_loss*self.args.lamda)
          G_percep_loss += per_loss.item()

        elif self.args.netG == "srgan_colearn":
          g_loss = gf_loss + gr_loss + colearn_loss
          CL_loss += colearn_loss
          g_loss_2x = gr_loss_2x + colearn_loss_2x
          CL_loss_2x += colearn_loss_2x
          G_rec_loss_2x += gr_loss_2x.item()
          G_tot_loss_2x += g_loss_2x.item() 

        else:
          g_loss = gf_loss + gr_loss

        G_adv_loss += gf_loss.item()
        G_rec_loss += gr_loss.item()
        G_tot_loss += g_loss.item()
        
      else:
        if self.args.percep_loss:
          per_loss = self.vgg_loss(predicted_hr_images,hr_images)
          g_loss = gr_loss + (per_loss*self.args.lamda) + tv_loss
          G_percep_loss += per_loss.item()
          TV_loss += tv_loss.item()
          
        elif self.args.netG == "srgan_colearn":
          g_loss = gr_loss + colearn_loss
          CL_loss += colearn_loss
          g_loss_2x = gr_loss_2x + colearn_loss_2x
          CL_loss_2x += colearn_loss_2x
          G_rec_loss_2x += gr_loss_2x.item()
          G_tot_loss_2x += g_loss_2x.item() 

        else:
          g_loss = gr_loss

        G_rec_loss += gr_loss.item()
        G_tot_loss += g_loss.item()

      g_loss.backward()
      self.optim_G.step()
      if self.args.netG == "srgan_colearn":
        g_loss_2x.backward()
        self.optim_G_2x.step()
      

    grid1 = make_grid(lr)
    grid2 = make_grid(hr)
    grid3 = make_grid(predicted_hr_images)
    grid1 = wandb.Image(grid1, caption="Low Resolution Image")
    grid2 = wandb.Image(grid2, caption="High Resolution Image")
    grid3 = wandb.Image(grid3, caption="Reconstructed High Resolution Image")
    wandb.log({"Train Original LR": grid1})
    wandb.log({"Train Original HR": grid2})
    wandb.log({"Train Reconstruced": grid3})
    return G_adv_loss,D_adv_loss,D_fake_loss,D_real_loss,G_rec_loss,G_tot_loss,G_percep_loss,TV_loss,CL_loss,CL_loss_2x,G_rec_loss_2x,G_tot_loss_2x
      

  def validation(self,val_loader):
    with torch.inference_mode():
        val_mae = 0
        val_rmse = 0 
        
        self.generator.eval()

        for batch_idx, (lr,lr2, hr,rgb_hr,mean,std,file_name) in enumerate(val_loader):

          lr = lr.to(device)
          hr = hr.to(device)
          lr2 = lr2.to(device)
          rgb_hr = rgb_hr.to(device)
          # lr = rgb_hr
          if self.args.netG == "esrgan" and self.args.use_pretrained:
            lr_images = lr.cpu().detach().numpy()
            hr_images = hr.cpu().detach().numpy()
            dummy_RGB_lr = np.ndarray(shape=(lr_images.shape[0],3, lr_images.shape[2], lr_images.shape[3]), dtype= np.float32)
            dummy_RGB_lr[:,0,:,:] = lr_images[:,0,:,:]
            dummy_RGB_lr[:,1,:,:] = lr_images[:,0,:,:]
            dummy_RGB_lr[:,2,:,:] = lr_images[:,0,:,:]
            dummy_RGB_hr = np.ndarray(shape=(hr_images.shape[0],3, hr_images.shape[2],  hr_images.shape[3]), dtype= np.float32)
            dummy_RGB_hr[:,0,:,:] = hr_images[:,0,:,:]
            dummy_RGB_hr[:,1,:,:] = hr_images[:,0,:,:]
            dummy_RGB_hr[:,2,:,:] = hr_images[:,0,:,:] 

            lr = torch.from_numpy(dummy_RGB_lr).to(device)
            hr = torch.from_numpy(dummy_RGB_hr).to(device)

            predicted_hr = self.generator(lr)
            
          # hr_unnorm = data_normalization.denormalize_torch_min_max(hr,mean,std)
          hr_unnorm = data_normalization.denormalize_torch(hr,mean,std)
          
          # predicted_hr_unnorm = data_normalization.denormalize_torch_min_max(predicted_hr,mean,std)
          predicted_hr_unnorm = data_normalization.denormalize_torch(predicted_hr,mean,std)
          # lr = ((lr+1)(mean-std)/2)+std
          rmse_i= 0 
          mae_i= 0 

          for i in range(hr.shape[0]):
              mae = mean_absolute_error(np.squeeze(hr_unnorm[i].cpu().numpy(),axis=0),np.squeeze(predicted_hr_unnorm[i].cpu().numpy(),axis=0))
              rmse = mean_squared_error(np.squeeze(hr_unnorm[i].cpu().numpy(),axis=0),np.squeeze(predicted_hr_unnorm[i].cpu().numpy(),axis=0),squared=False)
              # mae = mean_absolute_error((hr_unnorm[i].cpu().numpy()).flatten(),(predicted_hr_unnorm[i].cpu().numpy()).flatten())
              # rmse = mean_squared_error((hr_unnorm[i].cpu().numpy()).flatten(),(predicted_hr_unnorm[i].cpu().numpy()).flatten(),squared=False)
              mae_i+= mae
              rmse_i +=rmse

          rmse = rmse_i/hr.shape[0]
          mae = mae_i/hr.shape[0]

          val_rmse += rmse
          val_mae += mae

          grid1 = make_grid(lr)
          grid2 = make_grid(hr)
          grid3 = make_grid(predicted_hr)
          grid1 = wandb.Image(grid1, caption="Low Resolution Image")
          grid2 = wandb.Image(grid2, caption="High Resolution Image")
          grid3 = wandb.Image(grid3, caption="Reconstructed High Resolution Image")
          wandb.log({"Original LR": grid1})
          wandb.log({"Original HR": grid2})
          wandb.log({"Reconstruced": grid3})
      
    return val_rmse, val_mae

  def main(self):

    best_mae = math.inf

    train_loader,val_loader = self.get_dataloader()
    
    num_train_batches = float(len(train_loader))
    num_val_batches = float(len(val_loader))

    for epoch in range(self.args.epochs):
      
      G_adv_loss,D_adv_loss,D_fake_loss,D_real_loss,G_rec_loss,G_tot_loss,G_percep_loss,TV_loss,CL_loss,CL_loss_2x,G_rec_loss_2x,G_tot_loss_2x = self.train(train_loader,epoch)

      if self.args.gan_mode:
          wandb.log({"G Adversarial Loss": G_adv_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"D Adversarial Loss": D_adv_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"D fake": D_fake_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"D real": D_real_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"TV Loss":TV_loss/num_train_batches, 'epoch':epoch })
          # self.update_learning_rate(self.optim_D,self.scheduler_D)
          wandb.log({"G percep Loss":G_percep_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"TV Loss":TV_loss/num_train_batches, 'epoch':epoch })
      
      if self.args.netG == "srgan_colearn":
          wandb.log({"4x cl loss": CL_loss/num_train_batches, 'epoch':epoch })
          wandb.log({"2x cl loss": CL_loss_2x/num_train_batches, 'epoch':epoch })
          wandb.log({"2x G Reconstruction Loss": G_rec_loss_2x/num_train_batches, 'epoch':epoch })
          wandb.log({"2x G Loss Total": G_tot_loss_2x/num_train_batches, 'epoch':epoch })
          # self.update_learning_rate(self.optim_G_2x,self.scheduler_G_2x)

          
      wandb.log({"G Reconstruction Loss": G_rec_loss/num_train_batches, 'epoch':epoch })
      wandb.log({"G Loss Total": G_tot_loss/num_train_batches, 'epoch':epoch })

      
      # self.update_learning_rate(self.optim_G,self.scheduler_G)
      
      val_rmse, val_mae = self.validation(val_loader)
      
      val_rmse /= num_val_batches
      val_mae /= num_val_batches

      checkpoint_gen = self.generator.state_dict().copy()
      if epoch % 5 == 0 or val_mae <= best_mae:

        print(f"saved checkpoint at {epoch}: {val_rmse}")
        if val_mae <= best_mae:
          epoch_id = "best"
          best_mae = val_mae
        else:
          epoch_id = epoch

        if self.args.gan_mode:
          checkpoint_disc = self.discriminator.state_dict().copy()
          torch.save({'g_model':checkpoint_gen,
                      'd_model':checkpoint_disc,
                      'epoch':epoch,
                      'g_optim':self.optim_G.state_dict(),
                      'd_optim':self.optim_D.state_dict(),
                      'rmse':val_rmse,
                      'mae':val_mae}, '%s/model_%s.pt' % (self.checkpoints_dir,epoch_id))

        elif self.args.netG == "srgan_colearn":
         checkpoint_gen_2x = self.generator_2x.state_dict().copy()
         torch.save({'g_model':checkpoint_gen,
                     'epoch':epoch,
                     'g_optim':self.optim_G.state_dict(),
                     'g_model_2x' : checkpoint_gen_2x,
                     'g_optim_2x' : self.optim_G_2x.state_dict(),
                     'rmse':val_rmse,
                     'mae':val_mae}, '%s/model_%s.pt' % (self.checkpoints_dir,epoch_id))
        else:
          torch.save({'g_model':checkpoint_gen,
                      'epoch':epoch,
                      'g_optim':self.optim_G.state_dict(),
                      'rmse':val_rmse,
                      'mae':val_mae}, '%s/model_%s.pt' % (self.checkpoints_dir,epoch_id))


        best_mae = val_mae

                      
        best_mae = val_mae

      wandb.log({"RMSE" : val_rmse, 'epoch':epoch })
      wandb.log({"MAE" : val_mae, 'epoch':epoch })
      print(f"RMSE : {val_rmse:.3f} MAE: {val_mae:.3f}")


if __name__=="__main__":

  parser = argparse.ArgumentParser("Training of D-SRGAN model")
  
  parser.add_argument(
    "--checkpoint_name",type=str,required=True,help="Assigns the folder for storing checkpoints"
  )
  parser.add_argument(
    "--netG",type=str,required=True,help="choose the generator of the network"
  )
  parser.add_argument(
    "--netD",type=str,required=True,help="choose the discriminator of the network"
  )
  parser.add_argument(
    "--init_type",type=str,required=False,default='normal',help="choosing the weight intialization"
  )
  parser.add_argument(
    "--epochs",type=int,required=True,help="Number of epochs for training"
  )
  parser.add_argument(
    "--lr_rate",type=float,required=False,default=0.0001,help="Learning rate parametr for trianing"
  )
  parser.add_argument(
    "--l1_loss",action='store_true',help="chossing l1 loss for GAN instead of L2 (default)"
  )
  parser.add_argument(
    "--percep_loss",action='store_true',help="adding percep loss to the gan"
  )
  parser.add_argument(
    "--gan_mode",action='store_true',help="enables the discriminator training"
  )
  parser.add_argument(
    "--label_smooth",action='store_true',help="chossing to perform onesided label smoothing on real predictions"
  )
  parser.add_argument(
    "--train_lr_path",type=str,required=False,default='../datasets/swiss_dsm/trainset/lr_files_crop/', 
    help="path to the train low resolution DSMs directory"
  )
  parser.add_argument(
    "--train_hr_path",type=str, required=True, default='../datasets/swiss_dsm/trainset/hr_files_crop/',
    help="paht to the train high resolution DSMs directory"
  )
  parser.add_argument(
    "--test_lr_path",type=str,required=False,default='../datasets/swiss_dsm/testset/lr_files_crop/',
    help="path to the test low resolution DSMs directory"
  )
  parser.add_argument(
    "--test_hr_path",type=str, required=True, default='../datasets/swiss_dsm/testset/lr_files_crop/',
    help="paht to the test high resolution DSMs directory"
  )
  parser.add_argument(
    "--batch_size",type=int,default=4,required=True,help="batch sixe of the dataloader"
  )
  parser.add_argument(
    "--down_scale",type=int,default=4,required=True,help="diwnsample scale for the dataloader"
  )
  parser.add_argument(
    "--lamda",type=int,default=100,required=True,help="lambda for l1 loss in generator"
  )
  parser.add_argument(
    "--no_train_samples",type=int,default=1000,required=True,help="number of training samples"
  )
  parser.add_argument(
    '--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate'
  )
  parser.add_argument(
    '--n_epochs_decay', type=int, default=50, help='number of epochs to linearly decay learning rate to zero'
  )
  parser.add_argument(
    '--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]'
  )
  parser.add_argument(
    "--use_pretrained",type=str,default=None,help="enablees to use the pretrained weights"
  )
  parser.add_argument(
    "--relat_gan",action='store_true',help="enables the relat discriminator training"
  )
  parser.add_argument(
    "--data_dir",type=str,default=None,help="data directory for srcnn rgb and dsm model"
  )
  parser.add_argument(
        "--test_rgb_hr_path",type=str, required=True, help="paht to the rgb test high resolution DSMs directory"
  )
  parser.add_argument(
        "--train_rgb_hr_path",type=str, required=True, help="paht to the rgb test high resolution DSMs directory"
  )


  args = parser.parse_args()

  trainer = Model_train(args)
  trainer.main()
