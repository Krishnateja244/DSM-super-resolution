conda activate sr_thesis
export CUDA_VISIBLE_DEVICES="3"

# python srgan_training.py --checkpoint_name='srgan_unet_rgb' --epochs=150 --lr_rate 0.00001 --test_hr_path=../datasets/swiss_dsm/testset/hr_256_files.txt \
# --train_hr_path=../datasets/swiss_dsm/trainset/hr_256_files.txt --test_rgb_hr_path=../datasets/swiss_dsm/rgb_testset/hr_256_files.txt --train_rgb_hr_path=../datasets/swiss_dsm/rgb_trainset/hr_256_files.txt \
# --batch_size=4 --lamda=50  --no_train_samples=1000 --netG="srcnn_rgb" --netD='patch' --down_scale 4 --l1_loss --gan_mode
#--use_pretrained /home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/enc_srgan_nobn_newM3_gen/model_best.pt
# --data_dir="../datasets/"
# --use_pretrained /home/nall_kr/Documents/sr_dsm/D-SRGAN/weights/RealESRGAN_x4plus.pth
# #--relat_gan
# # # #   
# # # --percep_loss 

###dtm 

# python srgan_training.py --checkpoint_name='srgan_noact_nobn_pix_spec_dtm' --epochs=150 --lr_rate 0.00001 --train_hr_path=../datasets/swiss_dtm/trainset/hr_256_files.txt \
# --test_rgb_hr_path=../datasets/swiss_dsm/rgb_testset/hr_256_files.txt --train_rgb_hr_path=../datasets/swiss_dsm/rgb_trainset/hr_256_files.txt \
# --test_hr_path=../datasets/swiss_dtm/testset/hr_256_files.txt  --batch_size=4 --lamda=100  --no_train_samples=1000 --netG="srgan" --netD='pixel' \
# --down_scale 4 --l1_loss  --gan_mode 
# # #--relat_gan
# # # --use_pretrained
# # --percep_loss 


python test.py --test_hr_path=../datasets/swiss_dsm/testset/hr_256_files.txt --train_hr_path=../datasets/swiss_dsm/trainset/hr_256_files.txt --test_rgb_hr_path=../datasets/swiss_dsm/rgb_testset/hr_256_files.txt --train_rgb_hr_path=../datasets/swiss_dsm/rgb_trainset/hr_256_files.txt \
--gen_weights=/home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/netv2_noact/model_best.pt \
--gen_weights_2=/home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/srgan_noact_2x_256/model_best.pt \
--output_dir=/home/nall_kr/Documents/sr_dsm/datasets/swiss_dsm/report/netv2_noact/ \
--num_samples=200 --netG="netv2" --down_scale 4 
#--data_dir="../datasets/"
 
##dtm 
# python test.py --test_hr_path=../datasets/swiss_dtm/trainset/hr_256_files.txt --train_hr_path=../datasets/swiss_dtm/trainset/hr_256_files.txt --test_rgb_hr_path=../datasets/swiss_dsm/rgb_testset/hr_256_files.txt \
# --train_rgb_hr_path=../datasets/swiss_dsm/rgb_trainset/hr_256_files.txt \
# --gen_weights=/home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/unet_ganmode/model_best.pt \
# --gen_weights_2=/home/nall_kr/Documents/sr_dsm/D-SRGAN/checkpoints/srgan_noact_2x_256/model_best.pt \
# --output_dir=/home/nall_kr/Documents/sr_dsm/datasets/swiss_dsm/report/unet_ganmode/ \
# --num_samples=200 --netG="pix2pix" --down_scale 4 

#############################################################################################################################################################################
# pix2pix

# python train.py --train_hr_path=../datasets/swiss_dsm/trainset/hr_256_files.txt --test_hr_path=../datasets/swiss_dsm/testset/hr_256_files.txt \
# --no_train_samples=1000 --lambda_L1 100 --use_wandb --no_html --name pix2pix_noact_concat --netG "unet_256" --batch_size 4 --gpu_ids '0' \
# --norm 'batch' 
 

 ### pix2pixhd

# CUDA_LAUNCH_BLOCKING=1 torchrun train.py --train_hr_path=../datasets/swiss_dsm/trainset/hr_256_files.txt --test_hr_path=../datasets/swiss_dsm/testset/hr_256_files.txt \
# --no_instance --no_vgg_loss --no_html --name pix2pixhd_netv2_noact --verbose --norm batch --verbose --netG 'netv2' --gpu_ids '0' #--no_lsgan 

### old 

# python net_v2_train.py --checkpoint_name="net_1000_l2_16" --epochs=100 --lr_rate 0.0001 --train_hr_path=../datasets/swiss_dsm/trainset/hr_files_crop/ --train_lr_path=../datasets/swiss_dsm/trainset/lr_files_16_crop/ \
# --test_hr_path=../datasets/swiss_dsm/testset/hr_files_crop/ --test_lr_path=../datasets/swiss_dsm/testset/lr_files_16_crop/ --batch_size=4 


# python run_train.py --dataset Swiss --data-dir ../datasets/ --scaling 4 --save-dir ./diff_data/ --test_hr_path=../datasets/swiss_dsm/testset/hr_256_files.txt \
# --train_hr_path=../datasets/swiss_dsm/trainset/hr_256_files.txt --test_rgb_hr_path=../datasets/swiss_dsm/rgb_testset/hr_256_files.txt --train_rgb_hr_path=../datasets/swiss_dsm/rgb_trainset/hr_256_files.txt \
# --num-epochs 100 --wandb --wandb-project sr-gan --no_train_samples=1000 --down_scale 4 --batch-size 4


# python run_eval.py --dataset Swiss --data-dir ../datasets/ --checkpoint /home/nall_kr/Documents/sr_dsm/sr-diffusion/diff_data/Swiss/experiment_6_231/best_model.pth \
# --down_scale 4 --batch-size 4 --test_rgb_hr_path ../datasets/swiss_dsm/rgb_testset/hr_256_files.txt \
# --test_hr_path ../datasets/swiss_dsm/testset/hr_256_files.txt --output_dir /home/nall_kr/Documents/sr_dsm/datasets/swiss_dsm/report/diffusion/ 
