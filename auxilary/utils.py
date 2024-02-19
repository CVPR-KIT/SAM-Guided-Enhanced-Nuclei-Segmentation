import os
import sys
import json
import numpy as np
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import sklearn as sk

# Create directory
def createDir(dirs):
    '''
    Create a directory if it does not exist
    dirs: a list of directories to create
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Directory %s already exists' %dir)

# Check if the path exists
def checkPath(path):
    '''
    Check if the path exists
    path: path to check
    '''
    if not os.path.exists(path):
        return False
    else:
        return True
    
# Generate Configuration File
def makeJson(dict_, path = None):
    '''
    Generate a json file from the dictionary
    config: configuration dictionary
    path: path to save the json file
    '''
    if path == None:
        print("No path specified for  json file")
        return
    # convert dict config to json
    f_ = open(path, "w")
    f_.write(json.dumps(dict_, indent=4))
    f_.close()
    #print("Json file saved at: ", path)

# Read json file
def readJson(path = None):
    '''
    Read the json file
    path: path to the json file
    '''
    if path == None:
        print("No path specified for json file")
        return None
    f = open(path, "r")
    jsonFile = json.load(f)
    f.close()
    return jsonFile

# Read Configuration File
def readConfig(configPath = None):
    '''
    Read the configuration file
    configPath: path to the configuration file
    '''
    if configPath == None:
        print("Failed to read config file.")
        return None

    f = open(configPath, "r")
    exp = ["[", "#"]
    fileLines = f.readlines()
    config = {}
    for line in fileLines:
        if line[0] in exp:
            continue
        line = line.split("#")[0]
        #print(line)
        if len(line) < 2:
            continue
        # for string inputs
        if "\"" in line.split("=")[1]:
            config[line.split("=")[0].strip()] = line.split("=")[1].strip()[1:-1]
            if config[line.split("=")[0].strip()] == "True":
                config[line.split("=")[0].strip()] = True
            elif config[line.split("=")[0].strip()] == "False":
                config[line.split("=")[0].strip()] = False
            if config[line.split("=")[0].strip()] == "None":
                config[line.split("=")[0].strip()] = None
            if config[line.split("=")[0].strip()] == "none":
                config[line.split("=")[0].strip()] = None
            if line.split("=")[0].strip() in ["class1", "class2", "class3", "class4"]:
                config[line.split("=")[0].strip()] = getPixel(config[line.split("=")[0].strip()])
        # int or float inputs       
        else:
            if "." in line.split("=")[1]:
                config[line.split("=")[0].strip()] = float(line.split("=")[1].strip())
            else:
                config[line.split("=")[0].strip()] = int(line.split("=")[1].strip())
    f.close()
    # print config json for sanity check
    #makeConfigJson(config)
    return config

# get an array of pixel values from string
def getPixel(pixelStr):
    pixelStr = pixelStr.split(",")
    pixel = []
    for i in pixelStr:
        pixel.append(int(i))
    return pixel

# To read Organ Info from the json file
def getOrganInfo(path):
    '''
    Read the organ information from the json file
    path: path to the json file
    '''
    # if path doesnt exist, show error
    if not os.path.exists(path):
        print('Path to organ json file does not exist. Specified file path: ', path)
        return
    
    # load the json file
    data = readJson(path)
    # Mapping image IDs to their organ information
    organ_info_map = {item["ID"]: item["Organ"] for item in data}
    return organ_info_map


def saveTorchSummary(model, input_size, path="modelSummary.txt"):
    sys.stdout = open(path, "w")
    #f.write(summary(model, input_size))
    summary(model, input_size)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    #f.close()


def result_recolor(gray_img, config):
    img = np.zeros((gray_img.shape[0], gray_img.shape[1], 3), np.uint8)

    colormap = [config["class1"],config["class2"]] # black, white
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            #print(gray_img[i,j])
            img[i,j,:] = colormap[gray_img[i,j]]
    return img

# Calculate confusion matrix
def calc_confusion_matrix(label, pred, num_classes):
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    # Reshape label and pred tensors to match
    #print shapes of label and pred
    #print(label.shape)
    #print(pred.shape)
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    #print(label.shape)
    #print(pred.shape)

    # Calculate the confusion matrix
    cm = confusion_matrix(label, pred, labels=range(num_classes))

    return cm

# Calculate mIoU
def calc_mIoU(confusion_matrix):
    mIoU = 0
    for row in range(len(confusion_matrix)):
        intersect = confusion_matrix[row,row]
        union = np.sum(confusion_matrix[row,:]) + np.sum(confusion_matrix[:,row]) - intersect
        if union == 0:
            continue
        mIoU += intersect / union
    return mIoU / len(confusion_matrix)

# Calculate accuracy
def calc_accuracy(confusion_matrix):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    return accuracy

#calculate Dice Score
def calc_dice_score(confusion_matrix):
    dice_score = 0
    for row in range(len(confusion_matrix)):
        intersect = confusion_matrix[row,row]
        union = np.sum(confusion_matrix[row,:]) + np.sum(confusion_matrix[:,row])
        if union == 0:
            continue
        dice_score += 2*intersect / union
    return dice_score / len(confusion_matrix)

def classify_segments(preds, targets, iou_threshold=0.5):
    tp, fp, fn = 0, 0, len(targets)
    for pred in preds:
        matched = False
        for target in targets:
            iou = sk.metrics.jaccard_score(pred, target, average='weighted')
            if iou > iou_threshold:
                tp += 1
                matched = True
                break
        if matched:
            fn -= 1
        else:
            fp += 1
    return tp, fp, fn

def calculate_pq(preds, targets, iou_threshold=0.5):
    tp, fp, fn = classify_segments(preds, targets, iou_threshold)
    total_iou = sum(sk.metrics.jaccard_score(pred, target, average='weighted') for pred, target in zip(preds, targets) if sk.metrics.jaccard_score(pred, target, average='weighted') > iou_threshold)

    sq = total_iou / tp if tp > 0 else 0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    pq = sq * dq

    return pq