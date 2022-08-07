import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class RealWorldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir)) # without sorting the images and masks do not correspond
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(img_path).convert('RGB')) # .transpose(2,0,1) ##################
        #print(image.shape)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        # 0.0, 255.0
        
        mask[mask==255.0] = 1.0 # because sigmoid will be used on last app; ##### maybe this might have caused the predictions to be black? when uncommented it turns white
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        
        return image, mask

