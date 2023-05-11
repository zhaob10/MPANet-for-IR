from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../SIDD_Medium_Srgb/Data', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../SIDD_patches/train',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=300, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = multiprocessing.cpu_count()      

noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'groundtruth')   

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))   #删除目录

os.makedirs(noisy_patchDir)
os.makedirs(clean_patchDir)

#get sorted folders
files = natsorted(glob(os.path.join(src, '*', '*.PNG')))    #SIDD数据库中包含320个文件夹，每个文件夹中是一对图像
#这里是先glob找到SIDD数据库中所有.png为结尾的路径，存到列表中，再natsorted自然排序，即按照文件名排序
noisy_files, clean_files = [], []
for file_ in files:
    filename = os.path.split(file_)[-1] #split将路径名和文件名断开，返回的是(目录名，文件名)
    if 'GT' in filename:
        clean_files.append(file_)
    if 'NOISY' in filename:
        noisy_files.append(file_)   #将噪声图像和groundtruth的所有路径分成两类

def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    clean_img = cv2.imread(clean_file)      #读入图像

    H = noisy_img.shape[0]      
    W = noisy_img.shape[1]      #获取图像尺寸，opencv读入的图像维度排列为(h,w,c)
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)   #由于是随机裁剪，因此需要先随机选取图像的左上起始点
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]      #从起始点扩展出256×256的图像块

        cv2.imwrite(os.path.join(noisy_patchDir, '{}_{}.png'.format(i+1,j+1)), noisy_patch)
        cv2.imwrite(os.path.join(clean_patchDir, '{}_{}.png'.format(i+1,j+1)), clean_patch)     
        #将裁剪后的图像块读入图像，图像的命名是有规则的，第一个数代表第几对图像，第二个数代表第几个块，
        #这里与CBD用的数据集不同之处在于，所有噪声图像块放在一起，所有gt图像放在一起

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))
#多线程循环，delayed中放的是要循环的函数，tqdm的作用是美化进度条
