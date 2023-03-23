import torchvision.transforms as t
from torch.utils.data import Dataset
import torch
from constants import MEAN, STD
import PIL
import numpy as np
from utils import OwnRandAug
from torchvision.transforms.functional import InterpolationMode

class CustomDataset(Dataset):
    def __init__(self, 
                 data, 
                 norm_mean_std = 'own', 
                 simple_transformation=False, 
                 own_transformation=False, 
                 label_smooth = 0.0,
                 image_load_mode = 'shrink',
                 img_size=224,
                 bilinear = False
                 ):
        '''
        Args:
        data (numpy array): Image paths and labels in a single numpy array
        norm_mean_std (str): What normalization means and stds to use. Default 'own'. Possible: 'ImageNet', 'own'
        simple_transformation (bool): If True does not use image augmentation. Default False
        label_smooth (float): Amount of label smoothing. Default 0.0. Possible range [0,1]. NOT USED ANYMORE!
        image_load_mode (str): What loading mode to use. Default 'shrink'. Possible: 'shrink', 'PIL'
        
        Outputs:
        image (torch.Tensor): Image as torch Tensor. Shape (1,3,224,224)
        label (torch.Tensor): Label indicating whether image is considered empty. 0=empty, 1=ok 
        '''
        self.data = data        
        self.simple_transformation = simple_transformation
        self.own_transformation = own_transformation
        self.label_smooth = label_smooth
        self.mode = image_load_mode
        self.img_size = img_size
        #Amount of training images in each class. Used for class balanced loss
        self.num_class_one = np.sum(self.data[:,1].astype('float16'))
        self.num_class_zero = len(self.data) - self.num_class_one

        # CHOOSE WHICH MEAN args.mean_std
        means = MEAN[norm_mean_std]
        stds = STD[norm_mean_std]
        
        if bilinear:
            inter = InterpolationMode.BILINEAR
        else:
            inter = InterpolationMode.BICUBIC
        
        if (self.simple_transformation):
            self.transformation = t.Compose([
                            t.Resize((img_size,img_size), interpolation=inter),
                            t.ToTensor(),
                            t.Normalize(means, stds)
                        ])
        elif (self.own_transformation):
            self.transformation = OwnRandAug(means, stds, img_size, inter=inter)
        else:
            self.transformation = t.Compose([t.Resize((img_size,img_size), interpolation=inter),
                                             t.TrivialAugmentWide(),
                                             t.ToTensor(),
                                             t.Normalize(means, stds)])
            

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        #load image With PIL
        path = self.data[idx,0]
        if self.mode == 'PIL':
            img = PIL.Image.open(path)
            img = self.transformation(img)
        elif self.mode == 'shrink':
            img = PIL.Image.open(path)
            img.draft('RGB',(self.img_size,self.img_size))
            img = self.transformation(img.convert("RGB"))
        elif self.mode == 'oldshrink': #might crash! for some reason doestn change image into RGB
            img = PIL.Image.open(path)
            img.draft('RGB',(self.img_size,self.img_size))
            img = self.transformation(img)
            
        #load label
        label = float(self.data[idx,1])
        #if self.label_smooth > 0.0:
        #    if label == 1:
        #        label = label - 0.5 * self.label_smooth
        #    else:
        #        label = label + 0.5 * self.label_smooth
        
        label = torch.tensor(label)
        return img, label.long()
    
    def get_num_images(self):
        '''
        Returns a list separating number of benign and cancer images for calculating class balanced loss
        '''
        return [self.num_class_zero, self.num_class_one]