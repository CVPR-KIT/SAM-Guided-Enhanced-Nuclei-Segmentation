import torch
from tqdm import tqdm
import cv2
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from segment_anything import SamPredictor, sam_model_registry

#from auxiliary.utils import createDir



if __name__ == '__main__':

    # Get the path to the data directory
    #trainDir = '/home/bishal/TexVizSeg/data/trainNormal/'
    #valDir = '/home/bishal/TexVizSeg/data/valNormal/'
    testDir = '/mnt/BishalFiles/Datasets/MoNuSeg/wEncodings/testNormal/'


    path = '/mnt/BishalFiles/Datasets/MoNuSeg/wEncodings/testNormal/'
    #createDir([path, path+'train/', path+'val/', path+'test/'])

    # Load SAM
    samWeight = "segment-anything/SAMWeight/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=samWeight)
    predictor = SamPredictor(sam)

    #dirs = [trainDir, valDir]
    dirs = [testDir]
    for dir_ in dirs: 
        print(f"Generating {dir_} encodings")
        #mode = dir_.split('/')[-2].split('Normal')[0] + '/'
       #print(len(os.listdir(dir_))//3)
        for i in tqdm(range(len(os.listdir(dir_))//2)):
            img = cv2.imread(dir_ + str(i)+'.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            predictor.set_image(img)
            patch_embeddings = predictor.features
            torch.save(patch_embeddings, path + str(i)+'_en.pt')
