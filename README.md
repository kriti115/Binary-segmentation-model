# Binary-segmentation-model
A repository which performs binary segmentation on large scale real world dataset consisting of 10,000 x 10,000 size images of cities from Lower Saxony, Germany. It segments the buildings from the images using pre-trained UNetResNet architecture.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATASET
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The data for training can be downloaded from the link provided below:

You can use your own data by running patches.py where the large images will be cropped into patches which can be saved into the data folder under the subfolders as shown above.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
TRAINING
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Run the following in the terminal to train the network:

-- python train.py --config config.json

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
