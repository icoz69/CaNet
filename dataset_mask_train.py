

import random
import os
import torchvision
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time



class Dataset(object):


    def __init__(self, data_dir, fold, input_size=[321, 321] , normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1],prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.history_mask_list = [None] * self.__len__()
        self.prob=prob#probability of sampling history masks=0

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []

        fold_list=[0,1,2,3]
        fold_list.remove(fold)
        for fold in fold_list:

            f = open(os.path.join(self.data_dir, 'Binary_map_aug', 'train', 'split%1d_train.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = torchvision.transforms.ToTensor()
        # self.resize = torchvision.transforms.Resize(input_size)
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map_aug', 'train', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img

        # print (self.new_exist_class_list)

        support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        # print (query_name,support_name)

        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()
        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', support_name + '.jpg'))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', 'train', str(sample_class),
                                           support_name + '.png')))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]





        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0#random.random()

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'JPEGImages', query_name + '.jpg'))))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map_aug', 'train', str(sample_class),
                                           query_name + '.png')))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]



        if self.history_mask_list[index] is None:

            history_mask=torch.zeros(2,41,41).fill_(0.0)

        else:
            if random.random()>self.prob:
                history_mask=self.history_mask_list[index]
            else:
                history_mask = torch.zeros(2, 41, 41).fill_(0.0)


        return query_rgb, query_mask, support_rgb, support_mask,history_mask,sample_class,index

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.new_exist_class_list)


