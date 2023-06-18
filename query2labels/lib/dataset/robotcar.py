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
            self.base_dir = '/home2/wangyj/EE5346_2023_project/'
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
        if self.is_train:

            images_label_dir = Path('/home2/wangyj/EE5346_2023_project/sun2night_label_20m_50_5000.npy')
            file=np.load(images_label_dir)
            images_label_dir = Path('/home2/wangyj/EE5346_2023_project/sun2autumn_label_20m_50_5000.npy')
            file1=np.load(images_label_dir)
            images_label_dir = Path('/home2/wangyj/EE5346_2023_project/night2autumn_label_20m_50_5000.npy')
            file2=np.load(images_label_dir)
            self.dic_list=[]
            images_label_dir = Path('/home2/wangyj/EE5346_2023_project/negtive_label_15000.npy')
            negtive=np.load(images_label_dir)[:20000]
            for i in range(file.shape[0]):
                file_image0='Suncloud_val/stereo/centre/'+str(int(file[i,0]))+'.jpg'
                file_image1='Night_val/stereo/centre/'+str(int(file[i,1]))+'.jpg'
                label=int(file[i,2])
                self.dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
            for i in range(file1.shape[0]):
                file_image0='Suncloud_val/stereo/centre/'+str(int(file1[i,0]))+'.jpg'
                file_image1='Autumn_val/stereo/centre/'+str(int(file1[i,1]))+'.jpg'
                label=int(file1[i,2])
                self.dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
            for i in range(file2.shape[0]):
                file_image0='Night_val/stereo/centre/'+str(int(file2[i,0]))+'.jpg'
                file_image1='Autumn_val/stereo/centre/'+str(int(file2[i,1]))+'.jpg'
                label=int(file2[i,2])
                self.dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
            for i in range(negtive.shape[0]):
                file_image0=negtive[i,0][1:]
                file_image1=negtive[i,1][1:]
                label=int(negtive[i,2])
                self.dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
            print("load ",len(self.dic_list)," val dataset")


        else:
            images_label_dir = Path('/home2/wangyj/EE5346_2023_project/val.txt')
            f=open(images_label_dir, encoding='gbk')
            self.dic_list=[]
            for line in f:
                my_str=line.strip()
                file_image0=my_str.split(',')[0]
                file_image1=my_str.split(',')[1]
                label=my_str.split(',')[2]
                file_image0=file_image0.split(' ')[-1]
                file_image1=file_image1.split(' ')[-1]
                label=int(label.split(' ')[-1])
                self.dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
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
        label=torch.Tensor([img_pair['label']])
        rand=np.random.randn(1)

        image0=self.trans(image0)
        image1=self.trans(image1)

        meta={'image0':image0,'image1':image1,"label":label}
        return meta
