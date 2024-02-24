from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import sklearn as sk

import random
from dataset import nucleiDataset, nucleiValDataset, nucleiTestDataset
import argparse
import os
import time
from datetime import datetime
import logging
import wandb

from networkModules.modelUnet3p import UNet_3Plus

from auxilary.utils import *
from auxilary.lossFunctions import *


def wandb_init(config):
    # Initialize wandb
    wandb.init(project=config["wandbProjectName"], config=config)


def make_preRunNecessities(config):
    # Create the log directory
    logging.info("PreRun: Creating required directories")
    print("PreRun: Creating required directories")
    currTime = datetime.now().strftime("%m-%d_%H.%M.%S")
    config["expt_dir"] = f"Outputs/Experiments/experiment_{currTime}/"
    createDir(['Outputs/', 'Outputs/Experiments/',config["log"], config["expt_dir"], config["expt_dir"] + "model/", config["expt_dir"] + "step/"])

    # Create the config file
    logging.info("PreRun: Generating experiment config file")
    print("PreRun: generating experiment config file")
    makeJson(config, config["expt_dir"] + "config.json")
    
def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the config file.')
    return parser.parse_args()

def calculate_class_weights(targets, num_classes):
        # Calculate class weights based on target labels
        class_counts = torch.bincount(targets.flatten(), minlength=num_classes)
        total_samples = targets.numel()
        class_weights = total_samples / (num_classes * class_counts.float())
        '''print("class weights:", class_weights)
        print("class counts:", class_counts)
        print("total samples:", total_samples)'''
        return class_weights


def run_epoch_pipeline(model, data_loader, criterion, optimizer, epoch, device, mode, config, runOnce = False):


    if mode == 'train':
        model.train()
    else:
        model.eval()

    weightable_losses = ['weighteddice', 'modJaccard', 'jaccard', 'pwcel', 'improvedLoss', 'ClassRatioLossPlus', 'focalDiceLoss', 'focalDiceHDLoss']

    total_loss = 0
    confusion_matrix = np.zeros((config["num_classes"], config["num_classes"]))
    ious = []
    accs = []

    for idx, (image, label, enc) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}/{config['epochs']}")):
        image, label, enc = image.to(device), label.to(device), enc.to(device)
        

        gt = label.squeeze() if mode == 'train' else torch.reshape(label, (1, config["num_classes"], image.shape[2], image.shape[3]))


        if config["loss"] in weightable_losses:
            class_weights = calculate_class_weights(gt, config["num_classes"]).to(device)
            criterion.setWeights(class_weights)

        gt = gt.float()

        
        outputs = model((image, enc))

        loss = criterion(outputs, gt)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _, gt = torch.max(gt, 1)

        confusion_matrix += calc_confusion_matrix(gt, predicted, config["num_classes"])
        # IoU from sklearn
        IoU = sk.metrics.jaccard_score(gt.flatten().cpu(), predicted.flatten().cpu(), average='weighted')
        ious.append(IoU)

        acc = sk.metrics.accuracy_score(gt.flatten().cpu(), predicted.flatten().cpu())
        accs.append(acc)

        if mode == 'val' and idx == 0:
            # Save validation images, labels, and predictions
            save_validation_images(epoch, gt, predicted, config)

 
    mIoU = np.mean(ious)
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(accs)

    return avg_loss, confusion_matrix, mIoU, accuracy

def save_validation_images(epoch, gt, predicted, config):
    label = gt.reshape((1, gt.shape[1], gt.shape[2])).permute(1, 2, 0).cpu().numpy() * (255 / config["num_classes"])
    cv2.imwrite(f"{config['expt_dir']}step/{epoch}_label.png", label)

    predicted = predicted.permute(1, 2, 0).cpu().numpy() * (255 / config["num_classes"])
    cv2.imwrite(f"{config['expt_dir']}step/{epoch}_pred.png", predicted)

def main():
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.set_grad_enabled(True)
    start_time = time.time()

    sys.stdout = sys.__stdout__

    # Set cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"



    # Read configFile
    userConfig = arg_init().config
    if userConfig == 'none':
        print('please input config file')
        exit()
    config = readConfig(userConfig)

    createDir([config["log"]])
    logging.basicConfig(filename=config["log"] + "System.log", filemode='a', 
                        level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("\n\n#################### New Run ####################")
    #logging.basicConfig(level=logging.CRITICAL)


    if config["logWandb"]:
        wandb_init(config)



    learning_rate = config["learning_rate"]
    num_epochs = config["epochs"]
    
    # Make necessary directories
    make_preRunNecessities(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    logging.info(f"Using {device} device")

    
    # Configuring pipelines and model
    print('Setting pipilines')
    logging.info('Setting pipilines')

    print(f"loading encoding model")
    logging.info('loading encoding model')

    #Load config for encoding model
    if config["BaseModel"] == "UNet3Plus":
        print("loading UNet3Plus")
        logging.info("loading UNet3Plus")
        model = UNet_3Plus(config)
    else:
        print("encoding model not found")

    model.to(device)
    


    # Configuring DataLoaders
    print('configuring data loaders')
    logging.info('configuring data loaders')

    trainPaths = config["trainDataset"]
    valPaths = config["valDataset"]
    train_dataset = nucleiDataset(trainPaths, config)
    val_dataset = nucleiValDataset(valPaths, config)
    train_data = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_data = DataLoader(val_dataset,batch_size=1,num_workers=4)

    

        
    if config["loss"] =="weighteddice":
        criterion = weightedDiceLoss()
    elif config["loss"] =="focaldice":
        criterion = focalDiceLoss()
    else:
        print("loss not found. Using default loss: weightedDiceLoss()")
        criterion = weightedDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=1e-5)




    # Logging Parameters
    print('logging parameters')
    logging.info('logging parameters')
    logParam = {}
    
    logParam["debug"] = config["debug"]
    logParam["epochs"] = num_epochs
    logParam["learning_rate"] = learning_rate
    logParam["batch_size"] = config["batch_size"]
    logParam["optimizer"] = "Adam"
    logParam["loss"] = config["loss"]
    logParam["activation"] = config["activation"]

    logging.info(json.dumps(logParam, indent=4))

    
    train_losses = []
    train_accuracies = []
    val_losses = []
    best_val_accuracy = 0
    best_val_cm = None
    best_val_mIoU = None
    val_accuracies = []
    train_mIoUs = []
    val_mIoUs = []

    print('starting training')
    logging.info('starting training')

    
    
    for epoch in range(num_epochs):


        train_loss ,train_confusion_matrix, train_mIoU,train_accuracy =  run_epoch_pipeline(model, train_data, criterion, optimizer, epoch, device, 'train',config)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_mIoUs.append(train_mIoU)


        print('train_loss:',train_loss)
        #logging.info('train_loss:',train_loss)
        print('train_mIoU:',train_mIoU)
        #logging.info('train_mIoU:',train_mIoU)
        print('train_accuracy:',train_accuracy)
        #logging.info('train_accuracy:',train_accuracy)
        
        val_loss, val_confusion_matrix,val_mIoU,val_accuracy = run_epoch_pipeline(model, val_data, criterion, optimizer, epoch, device, 'val',config)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_mIoUs.append(val_mIoU)

        print('val_loss:',val_loss)
        #logging.info('val_loss:',val_loss)
        print('val_mIoU:',val_mIoU)
        #logging.info('val_mIoU:',val_mIoU)
        print('val_accuracy:',val_accuracy)
        #logging.info('val_accuracy:',val_accuracy)

        if config["logWandb"]:
            wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "train_mIoU": train_mIoU, "val_loss": val_loss, "val_accuracy": val_accuracy, "val_mIoU": val_mIoU})
        

        save_interval = 5  # Save every 5 epochs
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_losses[-1]
            best_val_cm = val_confusion_matrix
            best_val_mIoU = val_mIoU
            if epoch % save_interval == 0:
                print("saving model")
                
                logging.info(f"saving model at {config['expt_dir']+'model/best_model_'+str(epoch)+'.pth'}")
                #torch.save({'model_state_dict' : decodingModel.state_dict(),'optimizer_state_dict':optimizer.state_dict()},config["expt_dir"]+'model/decoding_best_model_'+str(epoch)+'.pth')
                torch.save({'model_state_dict' : model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},config["expt_dir"]+'model/best_model.pth')
            best_val_cm = val_confusion_matrix
            best_val_mIoU = val_mIoU


    print("loading model for testing")
    logging.info(f"loading model for testing from {config['expt_dir']+'model/best_model.pth'}")
    
    best_model = UNet_3Plus(config)
    checkpoint = torch.load(config["expt_dir"]+'model/best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model.to(device)

    print("Testing...")
    logging.info("Testing...")
    test_dataset = nucleiTestDataset(config["testDataset"], config)
    test_data = DataLoader(test_dataset,batch_size=1,num_workers=1)
    # make testing directory
    createDir([config["expt_dir"]+"testResults/"])
    pgbar = enumerate(tqdm(test_data))
    #val_confusion_matrix = np.zeros((config.num_classes, config.num_classes))
    ious = []
    accs = []
    dices = []
    for batch_idx, (images,label, enc) in pgbar:       
            pred = model((images.to(device), enc.to(device)))
            

            gt = label.to(device)

            
            _, rslt = torch.max(pred,1)
            _, gt = torch.max(gt,1)

            rslt = rslt.squeeze().type(torch.uint8)
            #cm = calc_confusion_matrix(y, rslt, config)
            #val_confusion_matrix += cm
            iou = sk.metrics.jaccard_score(gt.flatten().cpu(), rslt.flatten().cpu(), average='weighted')
            accuracy = sk.metrics.accuracy_score(gt.flatten().cpu(), rslt.flatten().cpu())
            dice = sk.metrics.f1_score(gt.flatten().cpu(), rslt.flatten().cpu(), average='weighted')

            ious.append(iou)
            accs.append(accuracy)
            dices.append(dice)


            # saving images
            logging.info("saving images")
            if config["input_img_type"] == "rgb":
                images = torch.reshape(images,(images.shape[2],images.shape[3],3))
            else:
                images = torch.reshape(images,(images.shape[2],images.shape[3],1))
            images = images.cpu().detach().numpy()
            cv2.imwrite(config["expt_dir"]+"testResults/"+str(batch_idx)+'_img'+'.png',images*255)
                           
            rslt_color = result_recolor(rslt.cpu().detach().numpy(), config)
            cv2.imwrite(config["expt_dir"]+"testResults/"+str(batch_idx)+'_pred_color'+'.png',rslt_color)
    
    test_mIoU = np.mean(ious)
    test_accuracy = np.mean(accs)
    test_dice_score = np.mean(dices)

    # Saving Training Stats
    print('Saving Training Stats')
    logging.info(f'Saving Training Stats at {config["expt_dir"]+"loss_log.txt"}')
    loss_log_file = open(config["expt_dir"]+'metrics.txt','w')
    loss_log_file.write("===============Training Losses ===========\n")
    for loss in train_losses:
        loss_log_file.write(str(loss)+', ')
    loss_log_file.write("\n\n===============Training Accuracies ===========\n")
    for acc in train_accuracies:
        loss_log_file.write(str(acc)+', ')
    loss_log_file.write('\n\n===============Training mIoUs ===========\n')
    for mIoU in train_mIoUs:
        loss_log_file.write(str(mIoU)+', ')
    loss_log_file.write('\n\n===============Validation Losses ===========\n')
    for loss in val_losses:
        loss_log_file.write(str(loss)+', ')
    loss_log_file.write('\n\n===============Validation Accuracies ===========\n')
    for acc in val_accuracies:
        loss_log_file.write(str(acc)+', ')
    loss_log_file.write('\n')
    loss_log_file.write('\n\n===============Validation mIoUs ===========\n')
    for mIoU in val_mIoUs:
        loss_log_file.write(str(mIoU)+', ')

    
    
    
    print('\n\n','================     val mIoUs    ================','\n\n')
    loss_log_file.write('\n\n================     val mIoUs    ================\n\n')
    print(test_mIoU)
    loss_log_file.write(str(test_mIoU))

    print('\n\n','================    val accuracy   ================','\n\n')
    loss_log_file.write('\n\n================    val accuracy   ================\n\n')
    print(test_accuracy)
    loss_log_file.write(str(test_accuracy))

    print('\n\n','================    Dice Score   ================','\n\n')
    loss_log_file.write('\n\n================    Dice Score   ================\n\n')
    print(test_dice_score)
    loss_log_file.write(str(test_dice_score))
    
    print('\n\n','================       end        ================','\n\n')
    loss_log_file.write('\n\n================       end        ================\n\n')

    end_time = time.time()
    elapsed_time = round((end_time - start_time ) / 3600, 2)
    print('elapsed time : ',elapsed_time,' hours')
    loss_log_file.write('\n\n================       Time Taken        ================\n\n')
    loss_log_file.write(str(elapsed_time)+' hours')
    loss_log_file.close()

    # Logging run time
    print('Logging run time')
    logging.info(f'Experiment took {elapsed_time} hours')

    #generting loss and accuracy plot
    print('generting loss and accuracy plot')
    logging.info(f'generting loss and accuracy plot at {config["expt_dir"]+"loss_plot.png"}')
    N = config["epochs"]
    plt.style.use("default")
    plt.figure()
    plt.plot(np.arange(0, N), train_losses, label="train_loss")
    plt.plot(np.arange(0, N), val_losses, label="val_loss")
    plt.plot(np.arange(0, N), train_accuracies, label="train_acc")
    plt.plot(np.arange(0, N), val_accuracies, label="val_acc")
    title = "Training Loss and Accuracy plot"
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config['expt_dir']+"plot.png")



    
    # Experiment End
    logging.info('Experiment End')
    
if __name__ == '__main__':
    main()