import os
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import glob
from .utils.pallete import *
import re
import numpy as np

import torch
import random

def encode_ab_ind(data_ab, opt):
    # Source: https://github.com/richzhang/colorization-pytorch/blob/master/util/util.py#L279
    # Encode ab value into an index
    # INPUTS
    #   data_ab   Nx2xHxW \in [-1,1]
    # OUTPUTS
    #   data_q    Nx1xHxW \in [0,Q)
    A = 2 * opt[1] / opt[2] + 1
    B = A
    data_ab_rs = np.round((data_ab*opt[0] + opt[1])/opt[2]) # normalized bin number
    data_q = data_ab_rs[..., 0]*A + data_ab_rs[..., 1]
    max_channel_value = round(255 * opt[0] + opt[1]) / opt[2]
    max_channel_value = max_channel_value * A + max_channel_value
    return data_q, max_channel_value

class NYUDepth(Dataset):
    def __init__(self, path_img, path_target, transforms):
        self.path_img = path_img
        with open(path_target, 'rb') as f:
            self.targets = pickle.load(f)
        self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        target = self.targets[index]

        
        img = Image.open(os.path.join(self.path_img, target['name']))
        anno = scipy.io.loadmat(os.path.join(self.path_img, target['name']).replace('/train','/train_anno_mat').replace('.png','.mat'))['anno'].astype('uint8')
        anno = Image.fromarray(anno)
        if self.transforms:
            num_targets = len(target['x_A'])
            landmarks = np.zeros((num_targets*2,2))

            landmarks[:num_targets,0] = target['x_A']
            landmarks[:num_targets,1] = target['y_A']
            landmarks[num_targets:,0] = target['x_B']
            landmarks[num_targets:,1] = target['y_B']
            
            img,anno,landmarks = self.transforms(img,anno,landmarks)
            
            lm = np.hstack([landmarks[:num_targets,:],landmarks[num_targets:,:],target['ordinal_relation'][:,np.newaxis]])

            ind1 = np.where(lm[:,:3]<0)[0]
            ind2 = np.where(lm[:,[0,2]]>=img.shape[1])[0]
            ind3 = np.where(lm[:,[1,3]]>=img.shape[2])[0]
            #ind = np.concatenate([ind1,ind2,ind3])
            #ind = np.unique(ind)
            #print(ind1.shape,ind2.shape,ind3.shape,ind.shape)
            #np.logical_or(np.logical_or(lm[:,0]<0,lm[:,1]<0),np.logical_or(np.logical_or(lm[:,2]<0,lm[:,3]<0),
            #np.logical_or(np.logical_or(lm[:,0]>=img.shape[0],lm[:,2]>=img.shape[0]),np.logical_or(lm[:,1]>=img.shape[1],lm[:,3]>=img.shape[1])))))
            #landmarks = np.delete(lm,ind,0)
            landmarks = lm
            landmarks[ind1,4] = 2
            landmarks[ind2,4] = 2
            landmarks[ind3,4] = 2

            target = {}
            target['x_A'] = landmarks[:,0]
            target['y_A'] = landmarks[:,1]
            target['x_B'] = landmarks[:,2]
            target['y_B'] = landmarks[:,3]
            target['ordinal_relation'] = landmarks[:,4]
        
            
        return img, anno, target

class NYUSeg(Dataset):
    def __init__(self, path_img, path_targets, transforms, opt, params):
        self.path_img = path_img
        self.tasks = params['tasks']
        self.imgs = sorted(glob.glob(path_img+'/*.jpg'))
        self.lbls = {}
        for t in params['tasks']:
            self.lbls[t] = sorted(glob.glob(path_targets[t]+'/*.png'))
            if t == "random_seg":
                random.shuffle(self.lbls[t])
        self.sp = path_targets
        #with open(path_target, 'rb') as f:
        #    self.targets = pickle.load(f)
        #self.imgs = [target['name'] for target in self.targets]
        self.transforms = transforms
        self.opt = opt

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        anno = {}

        max_val = -np.inf
        for i, t in enumerate(self.tasks):
            anno[t] = cv2.imread(self.lbls[t][index], cv2.IMREAD_UNCHANGED)[..., 1:]
            if t == "color":
                max_channel_value = np.max(anno[t])
                anno[t] = ((anno[t] / max_channel_value) * 15) # Normalize pixel values to the new range
            anno[t] = Image.fromarray(anno[t])

        if self.transforms:
            
            img,anno,landmarks = self.transforms(img,anno,landmarks = None)

            target = {}
            target['x_A'] = []
            target['y_A'] = []
            target['x_B'] = []
            target['y_B'] = []
            target['ordinal_relation'] = []

        # if max_val > 15:
        #     print(max_val)
            # exit(0)
        # return img, anno['seg'], anno['color'], anno['adaptive_threshold']
        # return img, anno['seg'], anno['adaptive_threshold']
        return img, anno['seg'], anno['random_seg']



     
