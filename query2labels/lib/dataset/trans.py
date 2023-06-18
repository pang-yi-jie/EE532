import matplotlib.pyplot as plt  
from torchvision.transforms import functional as F
import imgaug as ia
import numpy as np
import imgaug.augmenters as iaa
import cv2
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)

class Normalize(object):
    def __init__(self):
        self.mean = 0
        self.std = 0

    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - image.mean())
        image/= image.std()
        return image
    
class gammaconstrast(object):
    def __init__(self, train):
        self.train=train
        if self.train:
            self.aug = iaa.Sequential(iaa.SomeOf((2, 5),
                               [
                                  # 每个像素随机加减-10到10之间的数
                                   iaa.Add((-10, 10), per_channel=0.5),

                                   # 像素乘上0.5或者1.5之间的数字.
                                   iaa.Multiply((0.8, 1.2), per_channel=0.5),

                                   # 将整个图像的对比度变为原来的一半或者二倍
                                   iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
                               ],

                               random_order=True  # 随机的顺序把这些操作用在图像上
                               )
                )
        else:
            self.aug=iaa.Sequential(
            [iaa.Affine(
                    scale={"x": (1, 1), "y": (1, 1)},  # 图像缩放为80%到120%之间
  
                )])
            
    def __call__(self, image):
        image=self.aug(image=image) 
        return image
    
class RandomAffineTransform(object):
    def __init__(self,train):
        self.train = train
        self.OUTPUT_SIZE=(320,320)
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        if self.train:
            self.aug = iaa.Sequential(
            [iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},  # 图像缩放为90%到110%之间
                    translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},  # 平移±10%之间
                    rotate=(-10,10),  # 旋转±10度之间
                    order=[1],  # 使用最邻近差值或者双线性差值
                    cval=(0),  # 全白全黑填充
                )])
        else:
            self.aug = iaa.Sequential(
            [iaa.Affine(
                    scale={"x": (1, 1), "y": (1, 1)}, #不处理
  
                )])

    def __call__(self, image):
        image_aug = self.aug(image=image)
        image_smaller = ia.imresize_single_image(image_aug,self.OUTPUT_SIZE)
        return image_smaller
