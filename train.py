from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os.path as osp
from utils import *
import time
import torch.nn.functional as F
import tqdm
import random
import argparse
from dataset_mask_train import Dataset as Dataset_train
from dataset_mask_val import Dataset as Dataset_val
import os
import torch
from network import Res_Deeplab
import torch.nn as nn
import numpy as np



parser = argparse.ArgumentParser()


parser.add_argument('-lr',
                    type=float,
                    help='learning rate',
                    default=0.00025)

parser.add_argument('-prob',
                    type=float,
                    help='dropout rate of history mask',
                    default=0.7)


parser.add_argument('-bs',
                    type=int,
                    help='batchsize',
                    default=4)

parser.add_argument('-bs_val',
                    type=int,
                    help='batchsize for val',
                    default=64)



parser.add_argument('-fold',
                    type=int,
                    help='fold',
                    default=1)



parser.add_argument('-gpu',
                    type=str,
                    help='gpu id to use',
                    default='0,1')



parser.add_argument('-iter_time',
                    type=int,
                    default=5)



options = parser.parse_args()


data_dir = '/your/dataset/dir/VOCdevkit/VOC2012'




#set gpus
gpu_list = [int(x) for x in options.gpu.split(',')]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

torch.backends.cudnn.benchmark = True




IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
num_class = 2
num_epoch = 200
learning_rate = options.lr  # 0.000025#0.00025
input_size = (321, 321)
batch_size = options.bs
weight_decay = 0.0005
momentum = 0.9
power = 0.9

cudnn.enabled = True


# Create network.
model = Res_Deeplab(num_classes=num_class)
#load resnet-50 preatrained parameter
model = load_resnet50_param(model, stop_layer='layer4')
model=nn.DataParallel(model,[0,1])

# disable the  gradients of not optomized layers
turn_off(model)



checkpoint_dir = 'checkpoint/fo=%d/'% options.fold
check_dir(checkpoint_dir)








# loading data

# trainset
dataset = Dataset_train(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                  normalize_std=IMG_STD,prob=options.prob)
trainloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# valset
# this only a quick val dataset where all images are 321*321.
valset = Dataset_val(data_dir=data_dir, fold=options.fold, input_size=input_size, normalize_mean=IMG_MEAN,
                 normalize_std=IMG_STD)
valloader = data.DataLoader(valset, batch_size=options.bs_val, shuffle=False, num_workers=4,
                            drop_last=False)

save_pred_every =len(trainloader)




optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 10 * learning_rate}],
                          lr=learning_rate, momentum=momentum, weight_decay=weight_decay)




loss_list = []#track training loss
iou_list = []#track validaiton iou
highest_iou = 0








model.cuda()
tempory_loss = 0  # accumulated loss
model = model.train()
best_epoch=0
for epoch in range(0,num_epoch):


    begin_time = time.time()
    tqdm_gen = tqdm.tqdm(trainloader)


    for i_iter, batch in enumerate(tqdm_gen):

        query_rgb, query_mask,support_rgb, support_mask,history_mask,sample_class,index= batch

        query_rgb = (query_rgb).cuda(0)
        support_rgb = (support_rgb).cuda(0)
        support_mask = (support_mask).cuda(0)
        query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use
        query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
        history_mask=(history_mask).cuda(0)


        optimizer.zero_grad()

        pred=model(query_rgb, support_rgb, support_mask,history_mask)
        pred_softmax=F.softmax(pred,dim=1).data.cpu()

        #update history mask
        for j in range (support_mask.shape[0]):
            sub_index=index[j]
            dataset.history_mask_list[sub_index]=pred_softmax[j]


        pred = nn.functional.interpolate(pred,size=input_size, mode='bilinear',align_corners=True)#upsample

        loss = loss_calc_v1(pred, query_mask, 0)
        loss.backward()
        optimizer.step()

        tqdm_gen.set_description('e:%d loss = %.4f-:%.4f' % (
        epoch, loss.item(),highest_iou))


        #save training loss
        tempory_loss += loss.item()
        if i_iter % save_pred_every == 0 and i_iter != 0:

            loss_list.append(tempory_loss / save_pred_every)
            plot_loss(checkpoint_dir, loss_list, save_pred_every)
            np.savetxt(os.path.join(checkpoint_dir, 'loss_history.txt'), np.array(loss_list))
            tempory_loss = 0

    # ======================evaluate now==================
    with torch.no_grad():
        print ('----Evaluation----')
        model = model.eval()

        valset.history_mask_list=[None] * 1000
        best_iou = 0
        for eva_iter in range(options.iter_time):
            all_inter, all_union, all_predict = [0] * 5, [0] * 5, [0] * 5
            for i_iter, batch in enumerate(valloader):

                query_rgb, query_mask, support_rgb, support_mask, history_mask, sample_class, index = batch

                query_rgb = (query_rgb).cuda(0)
                support_rgb = (support_rgb).cuda(0)
                support_mask = (support_mask).cuda(0)
                query_mask = (query_mask).cuda(0).long()  # change formation for crossentropy use

                query_mask = query_mask[:, 0, :, :]  # remove the second dim,change formation for crossentropy use
                history_mask = (history_mask).cuda(0)

                pred = model(query_rgb, support_rgb, support_mask,history_mask)
                pred_softmax = F.softmax(pred, dim=1).data.cpu()

                # update history mask
                for j in range(support_mask.shape[0]):
                    sub_index = index[j]
                    valset.history_mask_list[sub_index] = pred_softmax[j]

                    pred = nn.functional.interpolate(pred, size=input_size, mode='bilinear',
                                                     align_corners=True)  #upsample  # upsample

                _, pred_label = torch.max(pred, 1)
                inter_list, union_list, _, num_predict_list = get_iou_v1(query_mask, pred_label)
                for j in range(query_mask.shape[0]):#batch size
                    all_inter[sample_class[j] - (options.fold * 5 + 1)] += inter_list[j]
                    all_union[sample_class[j] - (options.fold * 5 + 1)] += union_list[j]


            IOU = [0] * 5

            for j in range(5):
                IOU[j] = all_inter[j] / all_union[j]

            mean_iou = np.mean(IOU)
            print('IOU:%.4f' % (mean_iou))
            if mean_iou > best_iou:
                best_iou = mean_iou
            else:
                break




        iou_list.append(best_iou)
        plot_iou(checkpoint_dir, iou_list)
        np.savetxt(os.path.join(checkpoint_dir, 'iou_history.txt'), np.array(iou_list))
        if best_iou>highest_iou:
            highest_iou = best_iou
            model = model.eval()
            torch.save(model.cpu().state_dict(), osp.join(checkpoint_dir, 'model', 'best' '.pth'))
            model = model.train()
            best_epoch = epoch
            print('A better model is saved')



        print('IOU for this epoch: %.4f' % (best_iou))


        model = model.train()
        model.cuda()



    epoch_time = time.time() - begin_time
    print('best epoch:%d ,iout:%.4f' % (best_epoch, highest_iou))
    print('This epoch taks:', epoch_time, 'second')
    print('still need hour:%.4f' % ((num_epoch - epoch) * epoch_time / 3600))



