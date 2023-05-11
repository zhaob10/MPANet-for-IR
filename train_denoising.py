import os
from config import Config   #这里存放了一些参数
opt = Config('training.yml')    #实例化这些训练参数

gpus = ','.join([str(i) for i in opt.GPU])  #gpus=0，1，
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #指定按照顺序排列GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpus   #按照0，1排序gpu

import torch
torch.backends.cudnn.benchmark = True       #在模型固定的情况下使用能够加快网络训练速度

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from networks.SIPNet_3 import SIPNet
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler

if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)   #使得随机数的生成变得可预测，同时随机数生成情况会随着seed中的值的变化而变化，如果seed中的值不变，那么随机数一直生成相同的某个数字
    np.random.seed(1234)    
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)    #这几个随机数种子的作用应该是将实验中所有用到随机性的部分的初始状态都控制为一个相同的状态，便于比较实验结果

    start_epoch = 1
    mode = opt.MODEL.MODE   #'Denoising'
    session = opt.MODEL.SESSION     #'MIRNet'

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)  #存放结果的路径
    model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)  #存放断点续训模型的路径

    utils.mkdir(result_dir)     
    utils.mkdir(model_dir)      #根据路径创建文件夹

    train_dir = opt.TRAINING.TRAIN_DIR      #'../SIDD_patches/train'
    val_dir   = opt.TRAINING.VAL_DIR        #'../SIDD_patches/val' 
    save_images = opt.TRAINING.SAVE_IMAGES  #False

    ######### Model ###########
    model_restoration = SIPNet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]      #查询当前电脑中有几个GPU
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")    


    new_lr = opt.OPTIM.LR_INITIAL       #初始学习率，为2e-4

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)

    ######### Scheduler ###########
    warmup=True
    if warmup:
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    #这里是使用了预热学习率的方法，先使用较小的学习率训练3个epoch，而后再用大学习率继续训练，这样可以缓解一开始模型不稳定的问题

    ######### Resume ###########断点续训
    if opt.TRAINING.RESUME:
        path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')    #找到最近一次存储模型的路径
        utils.load_checkpoint(model_restoration,path_chk_rest)      #加载模型
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1     #查询当前训练到哪个epoch
        utils.load_optim(optimizer, path_chk_rest)          #加载优化器

        for i in range(1, start_epoch):
            scheduler.step()    #步进之前的epoch数，之所以这么做是由于没有在存储模型时存scheduler
        new_lr = scheduler.get_lr()[0]      #更新初始学习率，以便跟上之前的训练
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids)>1:
        model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids) #多卡并行

    ######### Loss ###########
    criterion = CharbonnierLoss().cuda()

    ######### DataLoaders ###########
    img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}    #训练集中的patch大小为128

    train_dataset = get_training_data(train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

    val_dataset = get_validation_data(val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    mixup = utils.MixUp_AUG()
    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    eval_now = len(train_loader)//4 - 1 #对于5000张图像的数据集，batch size为4的话，则一共分为1250份，此时len(train_loader)为1250
    print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
            
        for i, data in enumerate(tqdm(train_loader), 0):    
            #enumerate的作用是将可遍历的数据对象变成带索引的序列，i就表示这个索引，从0开始
            #data中存放的是一个元组，存放的内容依次为clean, noisy, clean_filename, noisy_filename
            #做数据集的类返回的元组中维度情况应该是[[c,h,w], [c,h,w], str, str]
            #dataloader后元组中维度变为[[b,c,h,w], [b,c,h,w], [b,str], [b, str]]


            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()

            #mixup数据增强
            if epoch>5:
                target, input_ = mixup.aug(target, input_)
            

            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)    #将恢复结果tensor中每个元素的值控制在0，1之间  
            
            loss = criterion(restored, target)  #这个损失是一个batch内的loss
        
            loss.backward()     #反向传播也是在一个batch内进行的
            optimizer.step()
            epoch_loss +=loss.item()    #这里汇总的是1个epoch内iteration个batch loss的总和

            #### Evaluation ####
            if i%eval_now==0 and i>0:   #i代表当前的iteration
                if save_images:
                    utils.mkdir(result_dir + '%d/%d'%(epoch,i)) #如果存图像的话就创建文件夹
                model_restoration.eval()    #将model切换到测试模式，即不启用Dropout和BN
                with torch.no_grad():       #这里的目的是在执行np_grad包含的语句时网络停止梯度更新，节省显存，便于测试更大的batch
                    psnr_val_rgb = []       #一个列表，将验证集中所有batch内计算出的PSNR汇总
                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[0].cuda()     #b,c,h,w  gt-image
                        input_ = data_val[1].cuda()     #b,c,h,w  noise-image
                        filenames = data_val[2]         #b,str

                        restored = model_restoration(input_)    #待验证的噪声图像送入网络中
                        restored = torch.clamp(restored,0,1) 
                        psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.)) #计算出一个batch内的PSNR

                        if save_images:
                            target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                            input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                            #将gt，noise，以及网络恢复的结果放到cpu上，再从计算图上detach下来，这里detach是返回
                            #一个新的变量，但与原变量不同的是它不具有梯度，最后再转换成数组

                            for batch in range(input_.shape[0]):    #在batch上遍历
                                temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1)
                                utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))
                                #将输入图像，恢复图像，gt图像拼起来，保存到相应的路径中

                    psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)  #计算出一个epoch内的psnr
                    
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i 
                        torch.save({'epoch': epoch, 
                                    'state_dict': model_restoration.state_dict(),
                                    'optimizer' : optimizer.state_dict()
                                    }, os.path.join(model_dir,"model_best.pth"))

                    print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                
                model_restoration.train()

        scheduler.step()
        
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   

        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    #总而言之，训练中是使用总的数据集进行多个epoch的训练，每个epoch上包含若干个iteration，并且iteration
    #的大小取决与batch size，每隔几个iteration进行一次验证，验证集和测试集是完全相同的，在验证集验证时
    #网络的Dropout，BN和梯度传播都停止，验证集送入网络相当于在一个epoch上进行，在一个batch上算出多幅图像
    #PSNR的值再平均，最后再将多个batch上得到的PSNR_batch平均，得到当前iteration上的验证结果
    #一共存了三个模型，一个是PSNR最后的模型，一个是最新的模型，最后是每个epoch上的模型