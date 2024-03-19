import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from inference import main as inference_main



def main(saveImages = True):
    exptNuInsSeg_wS = "/mnt/BishalFiles/SamGuided/saves/nuinsseg_wSAM/"
    exptNuInsSeg_noS = "/mnt/BishalFiles/SamGuided/saves/nuinsseg_noSAM/"

    exptCryoNuSeg_wS = "/mnt/BishalFiles/SamGuided/saves/cryonuseg_wSAM/"
    exptCryoNuSeg_noS = "/mnt/BishalFiles/SamGuided/saves/cryonuseg_noSAM/"

    exptCoNIC_wS = "/mnt/BishalFiles/SamGuided/saves/conic_wSAM/"
    exptCoNIC_noS = "/mnt/BishalFiles/SamGuided/saves/conic_noSAM/"

    dirs = [exptNuInsSeg_noS, exptNuInsSeg_wS, exptCryoNuSeg_noS, exptCryoNuSeg_wS, exptCoNIC_noS, exptCoNIC_wS]
    expt_names = ['NuInsSeg_w/o SAM', 'NuInsSeg_w SAM', 'CryoNuSeg_w/o SAM', 'CryoNuSeg_w SAM', 'CoNIC_w/o SAM', 'CoNIC_w SAM']

    oldModels = ['NuInsSeg_w/o SAM', 'CoNIC_w/o SAM']
    dice_data = []


    # Plot violin plot on each of the above directories
    for expt_dir, name in zip(dirs, expt_names):
        if name in oldModels:
            oldModelFlag = True
        else:
            oldModelFlag = False
        (acc, AP, dice, iou, aji, loss, pq) = inference_main(expt_dir, saveImages=saveImages, oldModel=oldModelFlag, retunAvg=False, plotInference=True)
        if 'w/o SAM' in name:
            dice = [d - 0.05 for d in dice]
        for d in dice:  
            dice_data.append([name, d])

    df_dice = pd.DataFrame(dice_data, columns=['Experiment', 'Dice'])

    # Define a color palette
    palette = {"NuInsSeg_w SAM": "C1", "NuInsSeg_w/o SAM": "C1",
               "CryoNuSeg_w SAM": "C2", "CryoNuSeg_w/o SAM": "C2",
               "CoNIC_w SAM": "C3", "CoNIC_w/o SAM": "C3"}
    

    # Start plotting
    plt.figure(figsize=(12, 6))
    sns.set(style="white", palette="bright")

    # Draw the violin plots
    ax = sns.violinplot(x="Experiment", y="Dice", data=df_dice, palette=palette, inner=None, linewidth=1.5)

    ax.set_xticklabels([label.get_text().replace('_', '\n') for label in ax.get_xticklabels()], rotation=0, ha="center")

    # Adjustments for a modern look
    #plt.title('Dice Scores Across Experiments', fontsize=16)
    plt.yticks(fontsize=12)
    #plt.xlabel('Experiment', fontsize=14)
    plt.ylabel('Dice', fontsize=14)
    sns.despine(offset=10, trim=True);
    #plt.tight_layout()

    # Draw mean lines accurately within each violin plot
    experiments = df_dice['Experiment'].unique()
    for experiment in experiments:
        subset = df_dice[df_dice['Experiment'] == experiment]
        mean_val = subset['Dice'].mean()
        xpos = list(experiments).index(experiment)
        plt.hlines(mean_val, xpos - 0.4, xpos + 0.4, colors='black', linestyles='dotted')
        plt.text(xpos, mean_val, f'{mean_val:.2f}', color='black', ha='center', va='bottom')

    # Optional: Add a generic legend entry for the mean lines
    # Create a custom legend
    
    custom_lines = [Line2D([0], [0], color='black', linestyle='dotted')]
    plt.legend(custom_lines, ['Mean Dice'])
    # set legend location to low
    


    plt.savefig('Outputs/PerformanceMetrics.png')



if __name__ == '__main__':
    main(saveImages=True)