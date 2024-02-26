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
    '''trainDir = '/mnt/Datasets/NuInsSeg/final/train/'
    valDir = '/mnt/Datasets/NuInsSeg/final/val/'
    testDir = '/mnt/Datasets/NuInsSeg/final/test/'''

    trainDir = '/mnt/Datasets/CryoNuSeg/final/train/'
    valDir = '/mnt/Datasets/CryoNuSeg/final/val/'
    testDir = '/mnt/Datasets/CryoNuSeg/final/test/'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #path = '/mnt/BishalFiles/Datasets/MoNuSeg/wEncodings/testNormal/'
    #createDir([path, path+'train/', path+'val/', path+'test/'])

    # Load SAM
    samWeight = "segment-anything/SAMWeights/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=samWeight).to(device)
    predictor = SamPredictor(sam)

    dirs = [trainDir, valDir, testDir]
    #dirs = [testDir]
    for dir_ in dirs: 
        print(f"Generating {dir_} encodings")
        #mode = dir_.split('/')[-2].split('Normal')[0] + '/'
        #print(len(os.listdir(dir_))//2)

        for i in tqdm(range(len(os.listdir(dir_))//2)):
            img = cv2.imread(dir_ + str(i)+'.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            predictor.set_image(img)
            patch_embeddings = predictor.features
            torch.save(patch_embeddings, dir_ + str(i)+'_en.pt')
