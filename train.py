import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNet
from unetresnet50 import UNetWithResnet50Encoder

import argparse
import json


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('my_runs/unetresnet_200') # Tensorboard setup: Makes a folder 'my_runs/my_poly'
import torchvision

# For modifications

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    matplotlib_imshow,
    mean_std,
    createFolder,
    binary_acc,
)

# Hyperparameters
image_height = 512 # 1280 original, cannot use this big resolution directly
image_width = 512 # 1918 original
train_img_dir = '/home/jovyan/Segmentation/data/train_images'
train_mask_dir = '/home/jovyan/Segmentation/data/train_masks'
val_img_dir = '/home/jovyan/Segmentation/data/val_images'
val_mask_dir = '/home/jovyan/Segmentation/data/val_masks'
#test_img_dir = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/images'
#test_mask_dir = '/home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/masks'


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train(config, loader, model, optimizer, loss_fn, scaler):
#def train(loader, model, optimizer, loss_fn, scaler):
    
    #running_loss = 0
    for epoch in range(config['num_epochs']):
        
        total_loss = 0
        total_correct = 0
        total_acc = 0
        
        loop = tqdm(loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=config['device'])
            targets = targets.float().unsqueeze(1).to(device=config['device'])
            
            if batch_idx == 0:
                writer.add_image('Images ', data[0], global_step = epoch) # adds steps in the tensorboard images
                writer.add_image('Masks ', targets[0], global_step = epoch) # adds steps in the tensorboard images

            #forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                #print('target size',predictions.size()) # torch.Size([1, 2, 512, 512])
                #predictions = predictions[:,None,:,:] doesnot work like this torch.Size([1, 1, 2, 512, 512])
                #print('target size',predictions.size())
                #print('input size', targets.size()) # torch.Size([1, 1, 512, 512]) These two cause size mismatch while using unetresnet50
                loss = loss_fn(predictions, targets)
                #acc = binary_acc(predictions, targets)
                      
                total_correct += get_num_correct(predictions, targets)
                #total_acc += acc.item()
                
                writer.add_scalar('Training Loss', loss.item(), epoch) # the range looks fine


                #running_loss += loss.item()
                # if batch_idx % 100 == 0:
                #     writer.add_scalar('training loss',
                #                 running_loss / 100,
                #                 epoch * len(loader) + batch_idx)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #update tqdm loop
            loop.set_postfix(loss=loss.item())
                
            
            #print(f'Epoch {epoch+0:03}: | Loss: {total_loss/len(loader):.5f} | Acc: {total_correct/len(loader):.3f}')

    
def main():   
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width = image_width),
            A.Rotate(limit=35, p = 1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    
    val_transform = A.Compose(
         [
            A.Resize(height=image_height, width = image_width),
             A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value = 255.0,
            ),
             ToTensorV2(),
        ],
    )
    
#     mean_train = [0.5301, 0.5316, 0.5136]
#     std_train = [0.1551, 0.1404, 0.1308]
#     mean_val = [0.5306, 0.5186, 0.5002]
#     std_val = [0.1640, 0.1539, 0.1465]
    
        
    # Simple Model
    #model = UNet(in_channels=3, out_channels=1).to(config['device'])
    model = UNetWithResnet50Encoder().to(config['device'])
    
    # Simple Loss function
    loss_fn = nn.BCEWithLogitsLoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr = config['learning_rate'])
    
    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        config['batch_size'],
        train_transform,
        val_transform,
        config['num_workers'],
        config['pin_memory']
    )
     
    #print('train_dataloader : ', train_loader)
    
    # Writing to Tensorboard
    
    #get some random training images
    dataiter = iter(train_loader)
    images, masks = dataiter.next()
    
    # Inspect model using Tensorboard
    writer.add_graph(model, images.to(device=config['device'])) # floatTensor cudaFloatTensor issue: solved by adding to device, now works
    # # create grid of images
    # img_grid = torchvision.utils.make_grid(images)
    # # show images
    # matplotlib_imshow(img_grid, one_channel = True)
    # # write to tensorboard
    # writer.add_image('images', img_grid) # this works: but the images look weird
      
    writer.close()
    
    if config['load_model'] == True:
        load_checkpoint(torch.load('my_checkpoint_unetresnet_200.pth.tar'), model)
    
    # check accuracy
    check_accuracy(val_loader, model, device=config['device'])
        
    scaler = torch.cuda.amp.GradScaler()
    
    #for epoch in range(num_epochs):
    #args = get_args()
    #in args.in_filepath:
    train( config, train_loader, model, optimizer, loss_fn, scaler)
    createFolder('./saved_images/unetresnet_200') # Create folder 

    # save model
    checkpoint = {
        'state_dict':model.state_dict(),
        'optimizer':optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy for validation
    check_accuracy(val_loader, model, device=config['device'])

    # print some examples to a folder
    save_predictions_as_imgs(
        val_loader, model, folder='saved_images/unetresnet_200', device=config['device'] 
    )

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('-c', '--config', dest='config', help='Absolute path to configuration file.')
    args = parser.parse_args()
    
    if not args.config:
        print('No configuration file provided')
        exit()
    else:
        with open(args.config, 'r') as inp:
            config=json.load(inp)
            #config = load_config(config, filepath_key="defaults_filepath")
        
    main()

