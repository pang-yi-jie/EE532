from os import listdir
from os.path import splitext
import os
from pathlib import Path
import numpy as np
import random
import cv2


images_dir = Path('/home2/wangyj/EE5346_2023_project/val.txt')
f=open(images_dir, encoding='gbk')
txt=[]
dic_list=[]

for line in f:
    my_str=line.strip()
    file_image0=my_str.split(',')[0]
    file_image1=my_str.split(',')[1]
    label=my_str.split(',')[2]
    file_image0=file_image0.split(' ')[-1]
    file_image1=file_image1.split(' ')[-1]
    label=int(label.split(' ')[-1])
    dic_list.append({'image0_file':file_image0,'image1_file':file_image1,'label':label})
    print(file_image0)
    print(file_image1)
    print(label)
print(len(dic_list))
index=random.sample(range(2000), 2000)
print()
np.save('index.npy',index)
