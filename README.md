# ResincNet
#intro : this is our final project for deep learning course. 
In this project we implemented  a new archticture called ResincNet base on SincNet && ResNet architectures that try to give a new soultion to the speaker verfication problem. 
we appends 1D multi-scale SincNet to 2D ResNet as the first convolutional layer in an attempt to learn 1D filtters and 2D fillters during the training stage,we demonstrate that our proposed archticture preform better than  the baseline SincNet and many well-known hand-crafted features on the Timit dataset.

## How to run :

in order to run the project sevreal steps have to be done : 
1. clone/download this project from this repo.
2. donwload and extract th TIMIT data-set: 
  2.1 download TIMIT  data-set from here https://deepai.org/dataset/timit
  2.2 from the downloaded zip go to data folder and extract the TRAIN and TEST directories into the datasets folder thar placed in     src/datasets.
3. run the project. 

## chose configuration and change paramaters and model 
in order to control the configuration there is yaml conficuriation file under the configs file in src/configs/cfg.yml.
your'e  welcome to try the configurtion of the model as you wish and exmine the result e.g: change the lr , num ephocs ,batch size etc. 

in order to run another model than the ResincNet and change the architecture of the model to test the  other models that we compore to change the value of type (which under the model label ) in the cfg.yml file to one of the follwing
"mfcc,cnn,sinc,resincNet"
mfcc=>mfcc with cnn 
sinc=>sincNet
cnn=>regular cnn
resincNet =>our architcture.


 
  
