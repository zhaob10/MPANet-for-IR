

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from numpy.lib.function_base import append
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
from networks.SIPNet_3 import SIPNet
from dataloaders.data_rgb import get_validation_data_sr2, get_validation_data_sr3, get_validation_data_sr4
import utils
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='RGB sr evaluation on the validation set of RealSR')
parser.add_argument('--input_dir', default='../RealSR_patches/val',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/sr/realsr4/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/sr/model_best_sr4.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
#action='store_true'代表只要该变量有传参，就设置为True
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data_sr4(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)
#创建用于测试的dataloader


model_restoration = SIPNet()    #引入模型

utils.load_checkpoint(model_restoration,args.weights)   #将模型训练好的参数加载到模型中
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()


model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    result = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        # rgb_gt = data_test[0].cuda()
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        noise_image = data_test[1].clone()
        noise_image = noise_image.numpy().squeeze().transpose((1,2,0))
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]

        psnr_before = str(psnr_loss(noise_image, rgb_gt))
        

        rgb_restored = model_restoration(rgb_noisy)
        # rgb_restored = torch.clamp(rgb_restored,0,1)
        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
     
        psnr_after = str(psnr_loss(rgb_restored, rgb_gt))
        tmp = psnr_before + ' ' + filenames[0] + ' ' + psnr_after + '\n'
        result.append(tmp)
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

        # rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        # rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        # rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if True:
            # for batch in range(len(rgb_gt)):
            #     denoised_img = img_as_ubyte(rgb_restored[batch])    #将图像转换为8bit图像
            #     utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)
            utils.save_img(args.result_dir + filenames[0][:-4] + '.png', img_as_ubyte(rgb_restored))
            

psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("PSNR: %.2f, SSIM: %.4f" %(psnr_val_rgb, ssim_val_rgb))    #最后输出一个PSNR
f = open('data_sr4.txt','w')
f.writelines(result)
f.close()

