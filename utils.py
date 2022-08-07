import os
import torch
import torchvision
from my_dataset import RealWorldDataset
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('my_runs/unetresnet_200') # Tensorboard setup: Makes a folder 'my_runs/my_poly'
import torchvision
import numpy as np
import pandas as pd
#from torchmetrics import JaccardIndex
#from my_private_dataset import PrivateDataset
import torch.nn as nn
#import data_transforms

def save_checkpoint(state, filename='my_checkpoint_unetresnet_200.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = RealWorldDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    
    val_ds = RealWorldDataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    #print('train_dataset', train_ds)
    return train_loader, val_loader

def get_test_loaders(test_dir,
                    test_maskdir,
                    batch_size,
                    num_workers=4,
                    pin_memory=True,):
    
    test_ds = RealWorldDataset(
        image_dir = test_dir,
        mask_dir = test_maskdir,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    
    return test_loader


def mean_std(loader):
    mean = 0
    std = 0
    n_samples = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples
    mean /= n_samples
    std /= n_samples
    
    return mean, std
        
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory )    

        
def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    #for epoch in range(3):
    #Test the model # Cuda out of memory!
    with torch.no_grad():
        #print('Loader type :',type(loader))
        for x, y in loader:
            x = x.to(device) #.permute((0,3,1,2)).float() ## permute to change [B,H,W,C] to [B,C,H,W] during test, float to cast the input so that input type and weight type are same
            #print(x.size()) # [8,512,3,512]; without permute ([8, 3, 512, 512])
            y = y.to(device).unsqueeze(1) # because label doesnot have channel
            #print(type(y))
            preds= torch.sigmoid(model(x)) 
            preds = (preds>0.5).float()
            #print('pred shape',preds.shape)
            #print('target shape',y.shape)

            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(preds, y)
            
            for i in range(3):
                writer.add_scalar('Validation Loss', loss.item(), i)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            acc = num_correct/num_pixels*100

            # DICE score: better metrics than accuracy!
            dice_score += (2 * (preds * y).sum()) / (
                (preds+y).sum() + 1e-8
            )       

            # IoU:           
            iou_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
            for iou_threshold in iou_thresholds:
                IoU = iou(preds.reshape(preds.shape[0], -1), y.reshape(y.shape[0], -1), iou_threshold)
                mean_iou = torch.mean(IoU)

            # if idx == 0:
            #     writer.add_scalar('Accuracy_in_utils', acc, global_step = idx) # did not work
    print(
        f'Got {num_correct}/{num_pixels} with acc {acc:.2f}'
    )
    print(f'Dice score: {dice_score/len(loader)}')
    print(f'IoU: {mean_iou}')
    
    
    model.train()
    
    #return num_correct/num_pixels*100:.2f
    
def save_predictions_as_imgs(
    loader, model, folder = 'saved_images/unetresnet_200', device='cuda'
):
    model.eval()
    #for epoch in range(config['num_epochs']):
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float() # 0.5
            #print('Predicion from utils Line 199 : ',preds) # all zeros
            
            # save predictions as csv: Not working: Maybe try later
            # pred_np = preds.cpu().numpy()
            # df = pd.DataFrame(pred_np) # ValueError: Must pass 2-d input. shape=(4, 1, 512, 512)
            # df.to_csv('predictions', index = False)

            # if idx == 0:
            #     writer.add_image('Predictions ', preds[0], global_step = idx) # adds steps in the tensorboard images
            #     writer.add_image('Groundtruths ', y.unsqueeze(1)[0], global_step = idx) # no output

            img_grid_1 = torchvision.utils.make_grid(preds) # check here by multiplying with 255 to see if the predictions show
            img_grid_2 = torchvision.utils.make_grid(y.unsqueeze(1))
            img_grid_3 = torchvision.utils.make_grid(x)

            #show images
            matplotlib_imshow(img_grid_1, one_channel = True)
            matplotlib_imshow(img_grid_2, one_channel = True)
            matplotlib_imshow(img_grid_3, one_channel = True)

            #write to tensorboard
            writer.add_image('Predictions', img_grid_1) 
            writer.add_image('Groundtruths', img_grid_2)
            writer.add_image('Images', img_grid_3)
            
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )

        torchvision.utils.save_image(y.unsqueeze(1), f'{folder}/gt_{idx}.png')
        torchvision.utils.save_image(x, f'{folder}/images_{idx}.png')
        writer.close()

    model.train()            
     
# For tensorboard 

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        

def iou(y_pred, y_true, threshold):
    assert len(y_pred.shape) == len(y_true.shape) == 2, "Input tensor shapes should be (N, .)"
    mask_pred = threshold < y_pred
    mask_true = threshold < y_true
    intersection = torch.sum(mask_pred * mask_true, dim=-1)
    union = torch.sum(mask_pred + mask_true, dim=-1)
    r = intersection.float() / union.float()
    r[union == 0] = 1
    return r


    
# def mean_std(loader, dataset):
#     psum    = torch.tensor([0.0, 0.0, 0.0])
#     psum_sq = torch.tensor([0.0, 0.0, 0.0])

#     for i in loader:
#         psum += i.sum(axis=[0,2,3])
#         psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])

#     count = len(dataset) * 512 * 512

#     total_mean = psum / count
#     total_var  = (psum_sq / count) - (total_mean ** 2)
#     total_std  = torch.sqrt(total_var)

#     return total_mean/255, total_std/255

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc