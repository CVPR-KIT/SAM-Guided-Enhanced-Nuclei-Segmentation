
from tqdm import tqdm
import numpy as np
from inference import main as inference

    
if __name__ == '__main__':
    '''
    for CryoNuSeg Dataset
    '''
    cv1Path = "Outputs/Experiments/experiment_02-22_08.25.13/"
    cv2Path = "Outputs/Experiments/experiment_02-22_12.50.53/"
    cv3Path = "Outputs/Experiments/experiment_02-22_15.12.02/"
    cv4Path = "Outputs/Experiments/experiment_02-22_18.36.51/"
    cv5Path = "Outputs/Experiments/experiment_02-23_00.22.41/"
    cv6Path = "Outputs/Experiments/experiment_02-23_07.17.34/"
    cv7Path = "Outputs/Experiments/experiment_02-23_10.03.57/"
    cv8Path = "Outputs/Experiments/experiment_02-23_14.55.23/"
    cv9Path = "Outputs/Experiments/experiment_02-23_21.28.54/"
    cv10Path = "Outputs/Experiments/experiment_02-24_00.49.09/"
   

    cvPaths = [cv1Path, cv2Path, cv3Path, cv4Path, cv5Path, cv6Path, cv7Path, cv8Path, cv9Path, cv10Path]
    accList = []
    mAPList = []
    mdiceList = []
    miouList = []
    ajiList = []
    meanlossList = []
    mpqList = []

    for expt_dir in tqdm(cvPaths):
        results = inference(expt_dir, saveImages=False)
        # unzip (acc, mAP, mdice, miou, aji, meanloss, mpq)
        acc, mAP, mdice, miou, aji, meanloss, mpq = results
        accList.append(acc)
        mAPList.append(mAP)
        mdiceList.append(mdice)
        miouList.append(miou)
        ajiList.append(aji)
        meanlossList.append(meanloss)
        mpqList.append(mpq)

    # print average values
    print(f"Average Accuracy - {np.average(accList)}")
    print(f"Average mAP - {np.average(mAPList)}")
    print(f"Average Dice - {np.average(mdiceList)}")
    print(f"Average mIoU - {np.average(miouList)}")
    print(f"Average AJI - {np.average(ajiList)}")
    print(f"Average Mean Loss - {np.average(meanlossList)}")
    print(f"Average PQ - {np.average(mpqList)}")


