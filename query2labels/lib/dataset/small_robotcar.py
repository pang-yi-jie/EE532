from os import listdir
from os.path import splitext
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from .trans import *

class LIGHT_DIS(Dataset):

    def __init__(self, is_train=True):
        # specify annotation file for dataset
        self.is_train = is_train
        self.load_name_file()
        if is_train:
            self.base_dir = '/home2/wangyj/query2labels/EE5346_2023_project/'
            self.trans=Compose([RandomAffineTransform(True),\
                            gammaconstrast(True),\
                            Normalize(),\
                            ToTensor()
                            ])
        else:
            self.base_dir = '/home2/wangyj/query2labels/EE5346_2023_project/'
            self.trans=Compose([RandomAffineTransform(False),\
                           #gammaconstrast(False),\
                            Normalize(),\
                            ToTensor()
                            ])
        
    def __len__(self):
        return len(self.dic_list)
    

    def load_name_file(self):
        images_label_dir = Path('/home2/wangyj/EE5346_2023_project/val.txt')
        f=open(images_label_dir, encoding='gbk')
        self.dic_list0=[]
        self.dic_list=[]
        for line in f:
            my_str=line.strip()
            file_image0=my_str.split(',')[0]
            file_image1=my_str.split(',')[1]
            label=my_str.split(',')[2]
            file_image0=file_image0.split(' ')[-1]
            file_image1=file_image1.split(' ')[-1]
            label=int(label.split(' ')[-1])
            self.dic_list0.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})

        index=np.load('/home2/wangyj/query2labels/lib/dataset/index.npy').astype(np.uint16)
        if self.is_train:
            for i in range(1600):
                self.dic_list.append(self.dic_list0[index[i]])
            print("load ",len(self.dic_list)," val dataset")
        else:
            for i in range(400):
                self.dic_list.append(self.dic_list0[index[i+1599]])
            print("load ",len(self.dic_list)," val dataset")
    def load_data(self,image_pair):   
        image_path0=self.base_dir+image_pair['image0_file']
        image_path1=self.base_dir+image_pair['image1_file']
        assert os.path.exists(image_path0), image_path0+' not find'
        assert os.path.exists(image_path1), image_path1+' not find'
        image0=cv2.imread(str(image_path0), cv2.IMREAD_GRAYSCALE)
        image1=cv2.imread(str(image_path1), cv2.IMREAD_GRAYSCALE)
        return image0,image1

    def __getitem__(self, idx):
        img_pair = self.dic_list[idx]
    
        image0,image1=self.load_data(img_pair)
        image0=self.trans(image0)
        image1=self.trans(image1)
        label=torch.Tensor([img_pair['label']])

        meta={'image0':image0,'image1':image1,"label":label}
        return meta



