"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from dataloaders.data_rgb_blur import get_validation_data
from networks.SIPNet_3 import SIPNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='../Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/Deblurring/models/SIPNet/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = SIPNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset, 'test')
test_dataset = get_validation_data(rgb_dir_test, img_options={'patch_size': 512})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir_input  = os.path.join(args.result_dir, dataset, 'input')
result_dir_restore  = os.path.join(args.result_dir, dataset, 'restore')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
utils.mkdir(result_dir_input)
utils.mkdir(result_dir_restore)
utils.mkdir(result_dir_gt)

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    result = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        noise_image = data_test[1].clone()
        noise_image = noise_image.numpy().squeeze().transpose((1,2,0))
        input_    = data_test[1].cuda()
        filenames = data_test[2]

        psnr_before = str(psnr_loss(noise_image, rgb_gt))

        # Padding in case images are not multiples of 8
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            factor = 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        if dataset == 'RealBlur_J' or dataset == 'RealBlur_R':
            restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        psnr_after = str(psnr_loss(restored, rgb_gt))
        tmp = psnr_before + ' ' + filenames[0] + ' ' + psnr_after + '\n'
        result.append(tmp)
        psnr_val_rgb.append(psnr_loss(restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(restored, rgb_gt, multichannel=True))

        utils.save_img((os.path.join(result_dir_input,  filenames[0]+'.png')), img_as_ubyte(noise_image))
        utils.save_img((os.path.join(result_dir_restore,  filenames[0]+'.png')), img_as_ubyte(restored))
        utils.save_img((os.path.join(result_dir_gt,  filenames[0]+'.png')), img_as_ubyte(rgb_gt))


psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("PSNR: %.2f, SSIM: %.4f" %(psnr_val_rgb, ssim_val_rgb))    #最后输出一个PSNR
f = open('data_gopro.txt','w')
f.writelines(result)
f.close()
