from typing import Any
import torch
import models.dpt.transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms as Tr
import numpy as np
from PIL import Image

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2 * base_size)

        transforms = []
        transforms.append(T.RandomResize(min_size, max_size))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        transforms.extend(
            [
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = T.Compose(transforms)

    def __call__(self, image, anno, landmarks=None):
        return self.transforms(image, anno, landmarks)
    

class SegmentationPresetVal:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.base_size = base_size
        self.mean = mean
        self.std = std

        transforms = []
        transforms.extend(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.transforms = T.Compose(transforms)

    def __call__(self, image, annos, landmarks=None):
        # Calculate the scale factor
        scale = min(self.base_size / float(min(image.height, image.width)), self.base_size)
        
        # Calculate the target height and width
        target_height, target_width = int(image.height * scale), int(image.width * scale)

        # Resize the image and annotation
        image = F.resize(image, (target_height, target_width))
        for key in annos:
            annos[key] = F.resize(annos[key], (target_height, target_width), interpolation=Tr.InterpolationMode.NEAREST)

        # Padding to the base size
        padding_height = self.base_size - target_height
        padding_width = self.base_size - target_width
        padding = (0, 0, padding_width, padding_height)  # Left, top, right, bottom padding


        image, annos, landmarks = self.transforms(image, annos, landmarks)
        image = F.pad(image, padding)
        for key in annos:
            annos[key] = F.pad(annos[key], padding, fill=255) # Change in transforms also (fill in crop) # , padding_mode="constant"

        #return self.transforms(image, anno, landmarks)
        return image, annos, landmarks
