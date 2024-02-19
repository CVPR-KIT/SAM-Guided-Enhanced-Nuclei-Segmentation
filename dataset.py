import cv2
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import sys
import matplotlib.pyplot as plt
import logging


class nucleiDataset(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            print("Config not found")
        else:
            self.config = config

        self.debug = self.config["debug"]
        self.debugDilution = self.config["trainingDilution"]
        self.project = self.config["wandbProjectName"]
    
        return 
    
    def __len__(self):
        if self.debug:
            return self.debugDilution
        else:
            factor = 3
            if self.project == "monunet-segmenation":
                factor = 4
            return len(os.listdir(self.img_dir))//factor -1

    def __getitem__(self, index):
        try:
            image = cv2.imread(os.path.join(self.img_dir,str(index)+'.png')) / 255
        except TypeError:
            print(os.path.join(self.img_dir,str(index)+'.png'))     

        try:
            label = cv2.imread(os.path.join(self.img_dir,str(index+1)+'_label'+'.png'),cv2.IMREAD_GRAYSCALE)
        except:
            print(os.path.join(self.img_dir,str(index+1)+'_label'+'.png'))  

        try:
            samencoding = torch.load(os.path.join(self.img_dir,str(index)+'_en.pt'), map_location=torch.device('cpu'))   
        except:
            print(os.path.join(self.img_dir,str(index)+'_en.pt'))
        
        #print(index)
        #print(samencoding.shape)
        
        if self.config["input_img_type"] == "rgb":
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,image.shape[0],image.shape[1]))

        
        label[label==255] = 1
        label[label==0] = 0

        class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
        class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

        label = np.stack([class_0, class_1], axis=0)
        #print(samencoding.shape)
        #print(index)


        return torch.Tensor(image),torch.LongTensor(label), samencoding.squeeze(0)

class nucleiValDataset(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            print("Config not found")
        else:
            self.config = config
        self.debug = self.config["debug"]
        self.debugDilution = self.config["validationDilution"]
    
        return 
    
    def __len__(self):
        if self.debug:
            return self.debugDilution
        else:
            factor = 3
            if self.project == "monunet-segmenation":
                factor = 4
            return len(os.listdir(self.img_dir))//factor -1


    def __getitem__(self, index):
        try:
            image = cv2.imread(os.path.join(self.img_dir,str(index)+'.png')) / 255
        except TypeError:
            print(os.path.join(self.img_dir,str(index)+'.png'))        
        
        if self.config["input_img_type"] == "rgb":
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image,(1,image.shape[0],image.shape[1]))
        
        try:
            samencoding = torch.load(os.path.join(self.img_dir,str(index)+'_en.pt'), map_location=torch.device('cpu'))   
        except:
            print(os.path.join(self.img_dir,str(index)+'_en.pt'))

        
        label = cv2.imread(os.path.join(self.img_dir,str(index+1)+'_label'+'.png'),cv2.IMREAD_GRAYSCALE)
        label[label==255] = 1
        label[label==0] = 0

        class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
        class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

        label = np.stack([class_0, class_1], axis=0)


        return torch.Tensor(image),torch.LongTensor(label), samencoding.squeeze(0)


class nucleiTestDataset(Dataset):
    def __init__(self, img_dir, config = None):
        self.img_dir = img_dir

        if config is None:
            print("Config not found")
        else:
            self.config = config
            
        self.wid = 512 # default value and is replaced later 
        self.hit = 512 # default value and is replaced later 
        logging.basicConfig(filename=self.config["log"] + "dataloader.log", filemode='w', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        self.len = 256
        return 
    
    def __len__(self):
        return len(os.listdir(self.img_dir))//3

    def __getitem__(self, index):

        if self.config["input_img_type"] == "rgb":
            image = cv2.imread(os.path.join(self.img_dir,str(index)+".png"))/255
            
            # Normalize image
            #image = normalize_image(image)
        else:
            image = cv2.imread(os.path.join(self.img_dir,str(index)+".png"),cv2.IMREAD_GRAYSCALE)/255
            image = np.reshape(image,(1,image.shape[0],image.shape[1]))
        

        label = cv2.imread(os.path.join(self.img_dir,str(index)+"_label.png"),cv2.IMREAD_GRAYSCALE)

        try:
            samencoding = torch.load(os.path.join(self.img_dir,str(index)+'_en.pt'), map_location=torch.device('cpu'))   
        except:
            print(os.path.join(self.img_dir,str(index)+'_en.pt'))

        self.wid = image.shape[0]
        self.hit = image.shape[1]

        # reshape image
        image = cv2.resize(image, (self.len, self.len), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (self.len, self.len), interpolation=cv2.INTER_CUBIC)
        
        
        label[label==255] = 1
        label[label==0] = 0

        class_0 = np.where(label == 0, 1, 0)  # Channel for class 0
        class_1 = np.where(label == 1, 1, 0)  # Channel for class 1

        label = np.stack([class_0, class_1], axis=0)
            
        
        image = np.transpose(image, (2, 0, 1))

        return torch.Tensor(image),torch.LongTensor(label), samencoding.squeeze(0)
    