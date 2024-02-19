from dataset import nucleiTestDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import cv2
from auxilary.utils import *
from networkModules.modelUnet3p import UNet_3Plus
from datetime import datetime
from sklearn.metrics import average_precision_score, jaccard_score
import json
import matplotlib.pyplot as plt
from auxilary.lossFunctions import weightedDiceLoss

import logging

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_dir', type=str, default='none', help='Path to the experiment directory.')
    return parser.parse_args()

def make_preRunNecessities(expt_dir):
    # Read config.json file
    print("PreRun: Reading config file")
    
    # read json file
    config = None
    with open(expt_dir + "config.json") as f:
        config = json.load(f)   
        #print(config)
    
    # Create the required directories
    print("PreRun: Creating required directories")
    createDir([config['log'], config['expt_dir']+'inference/', config['expt_dir']+'inference/testData/'])

    return config if config is not None else print("PreRun: Error reading config file")

def calculate_class_weights(targets, num_classes):
        # Calculate class weights based on target labels
        class_counts = torch.bincount(targets.flatten(), minlength=num_classes)
        total_samples = targets.numel()
        class_weights = total_samples / (num_classes * class_counts.float())
        '''print("class weights:", class_weights)
        print("class counts:", class_counts)
        print("total samples:", total_samples)'''
        return class_weights


    
def runInference(data, model, device, config, img_type):
    accList = []
    count= 0
    mAPs = []
    dices = []
    mious = []
    aji = []
    losses = []
    pqs = []

    criterion = weightedDiceLoss()

    for i,(images,y, enc) in enumerate(tqdm(data)):
        input = (images.to(device), enc.to(device))
        pred = model(input)
        gt = y.to(device)

        #print(pred.shape)

        if not count:
            #torch.onnx.export(model, images.to(device), 'SS_MODEL.onnx', input_names=["Input Image"], output_names=["Predected Labels"])
            count+=1
        #print(int(pred.shape[2]))
        (wid, hit) = (int(pred.shape[2]), int(pred.shape[3]))
            
        #y = y.reshape((1,wid,hit))
        
        class_weights = calculate_class_weights(y, 2)
        criterion = weightedDiceLoss()
        criterion.setWeights(class_weights.to(device))
        

        _, rslt = torch.max(pred,1)
        _, gt = torch.max(gt,1)

        rslt = rslt.squeeze().type(torch.uint8)

        #loss = 1- criterion(pred, y)
        #loss = 0
        #losses.append(loss.item())

        #print(f"rslt: {rslt.shape}")
        #print(f"y: {y.shape}")
        y = y.type(torch.uint8).cpu()

        test_acc = torch.sum(rslt.cpu() == y)


        if config["input_img_type"] == "rgb":
            #print(f"RGB Image : {images.shape}")
            #images = torch.reshape(images,(wid,hit, 3))
            images = images.squeeze(0)
            images = images.permute(1, 2, 0)
            #print(f"RGB Image reshaped : {images.shape}")
        else:
            images = torch.reshape(images,(wid,hit,1))
        

        
        iou = sk.metrics.jaccard_score(gt.flatten().cpu(), rslt.flatten().cpu(), average='weighted')
        accuracy = sk.metrics.accuracy_score(gt.flatten().cpu(), rslt.flatten().cpu())
        dice = sk.metrics.f1_score(gt.flatten().cpu(), rslt.flatten().cpu(), average='weighted')

        accList.append(accuracy)
        mAPs.append(average_precision_score(gt.flatten().cpu(), rslt.flatten().cpu()))
        dices.append(dice)
        mious.append(iou)
        pq_score = calculate_pq(gt, rslt.cpu())
        pqs.append(pq_score)

        

        ji = jaccard_score(y.cpu().detach().numpy().reshape(-1), rslt.cpu().detach().numpy().reshape(-1))
        aji.append(ji)


        images = images.cpu().detach().numpy()
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+'img.png',images*255)

            
        y = y.squeeze()
        label_color = result_recolor(y)
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+'label.png',label_color)

        rslt = rslt.squeeze()
        rslt_color = result_recolor(rslt.cpu().detach().numpy())
        cv2.imwrite(config['expt_dir']+'inference/'+img_type+'/'+str(i)+'_'+str(test_acc.item()/(wid*hit))[:5]+'_'+'predict.png',rslt_color)
    return np.average(accList), np.average(mAPs), np.average(dices), np.average(mious), np.average(aji), np.average(losses), np.average(pqs)


'''
    1. load model
    2. load dataset
    3. inference
    4. save result
'''

def main():
    # Load Config
    args = arg_init()

    if args.expt_dir == 'none':
        print("Please specify experiment directory")
        sys.exit(1)

    
    # run preRun
    config = make_preRunNecessities(args.expt_dir)

    # set logging
    logging.basicConfig(filename=config["log"] + "Test.log", filemode='a', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Testing Initiated")
    logging.info("PreRun: Creating required directories")


    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Using {device} device")

    # set weight path
    weight_path = args.expt_dir + "model/best_model.pth"
    # check if weight path exists
    if not os.path.exists(weight_path):
        print("Please specify valid model type")
        sys.exit(1)

    # log weight path
    logging.info("Weight Path: " + weight_path)
    
    # set model
    model = UNet_3Plus(config)

    # Start inference
    logging.info("Starting Inference")

    logging.info(f"Loading Model at {weight_path}")
    checkpoint = torch.load(weight_path)

    logging.info("Loading checkpoints")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    logging.info("Starting evaluations")
    model.eval()

    # Write to File for future reference
    f = open(config["log"] + "inferences.csv", "a")


    paths = [(config["testDataset"], 'testData')]
    for path, img_type in paths:
        # Load Dataset
        logging.info("Loading dataset")
        dataset = nucleiTestDataset(path, config)
        data = DataLoader(dataset,batch_size=1)
        acc, mAP, mdice, miou, aji, meanloss, mpq = runInference(data, model, device, config, img_type)
        f.write(f"{args.expt_dir},{img_type},{np.average(acc)} \n")
        print(f"Testing Accuracy -{args.expt_dir}-{img_type}- {acc} \n")
        print(f"Testing mAP -{args.expt_dir}-{img_type}- {mAP} \n")
        print(f"Testing Dice -{args.expt_dir}-{img_type}- {mdice} \n")
        print(f"Testing mIoU -{args.expt_dir}-{img_type}- {miou} \n")
        print(f"Testing mean Loss -{args.expt_dir}-{img_type}- {meanloss} \n")
        print(f"Testing PQ -{args.expt_dir}-{img_type}- {mpq} \n")
           # print(f"Testing AJI -{args.expt_dir}-{img_type}- {aji} \n")


    
    f.close()

    
if __name__ == '__main__':
    '''
    run command: python train_test.py --expt_dir <path to experiment directory> --img_dir all
    '''
    main()
