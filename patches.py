''' pip install patchify '''

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import os

def patches():
    #large_image_path = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/train/images/'
    #large_mask_path = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/train/gt'
    
    large_image_path = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/images/'
    large_mask_path = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/masks'
    
    large_images = os.listdir(large_image_path)
    img_name = []
    
    for i in large_images:
        im_name, _ = os.path.splitext(i)
        img_name.append(im_name)
    img_name = sorted(img_name) # use this to name the files
    #print(img_name)
        
    #large_images = sorted(large_images)[0:20]
    #large_images = sorted(large_images)[20:40]
    #large_images = sorted(large_images)[40::]
    
    large_images = sorted(large_images)[0:20]
    #large_images = sorted(large_images)[20:40]
    #large_images = sorted(large_images)[40::]
    #print(large_images)
    
    l_image = []
    l_mask = []
    for i in large_images:
        if os.path.isdir(i):
            continue
            
        image = tiff.imread(os.path.join(large_image_path, i))
        #print(image.shape) # (10000, 10000, 3)
        mask = tiff.imread(os.path.join(large_mask_path, i))
        #print(mask)
        l_image.append(image)
        l_mask.append(mask)
    #print(type(l_mask))
    #print(len(l_image)) # 15
    #print(l_image[0].shape) # (10000, 10000, 3)
    
    for img in range(len(l_image)):
        #print(img)
        l_img = l_image[img]
    #print(l_img.shape) # (10000, 10000, 3)
        #print(l_msk)
        #print(type(l_msk)) # <class 'numpy.ndarray'>
        patches_img = patchify(l_img, (512,512,3), step = 256) # step 256 means 50% overlap
    #print(patches_img.shape) # (38,38,1,512,512,3) WTF?

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                
                # Perform one line at a time
                #tiff.imwrite('patches/images_train/' + 'image_' + str(img) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                #tiff.imwrite('patches/images_train/' + 'image_' + str(img+19) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                #tiff.imwrite('patches/images_train/' + 'image_' + str(img+19+6) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                
                # Perform one line at a time
                #tiff.imwrite('data/images_test/' + 'image_' + str(img) + '_' + str(i)+str(j)+ '.tif', single_patch_img) # automate folder creation
                #tiff.imwrite('data/images_test/' + 'image_' + str(img+19) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                tiff.imwrite('data/images_test/' + 'image_' + str(img+19+6) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
        #print('i: ', i)
        #print('j: ', j)

    for img in range(len(l_mask)):
        l_msk = l_mask[img]
    #print(l_img.shape) # (10000, 10000, 3)
        #print(l_msk)
        #print(type(l_msk)) # <class 'numpy.ndarray'>
        patches_img = patchify(l_msk, (512,512), step = 256) # step 256 means 50% overlap
    #print(patches_img.shape) # (38,38,1,512,512,3) WTF?

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i,j,:,:]
                # Perform one line at a time
                #tiff.imwrite('patches/masks_train/' + 'mask_' + str(img) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                #tiff.imwrite('patches/masks_train/' + 'mask_' + str(img+19) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                #tiff.imwrite('patches/masks_train/' + 'mask_' + str(img+19+6) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                
                # Perform one line at a time
                #tiff.imwrite('data/masks_test/' + 'mask_' + str(img) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                #tiff.imwrite('data/masks_test/' + 'mask_' + str(img+19) + '_' + str(i)+str(j)+ '.tif', single_patch_img)
                tiff.imwrite('data/masks_test/' + 'mask_' + str(img+19+6) + '_' + str(i)+str(j)+ '.tif', single_patch_img)

    
def main():
    patches()

if __name__ == '__main__':
    main()