import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from inference import main as inference_main



def main(saveImages = True):
    exptNuInsSeg_wS = "/mnt/BishalFiles/SamGuided/saves/nuinsseg_wSAM/"
    exptNuInsSeg_noS = "/mnt/BishalFiles/SamGuided/saves/nuinsseg_noSAM/"

    exptCryoNuSeg_wS = "/mnt/BishalFiles/SamGuided/saves/cryonuseg_wSAM/"
    exptCryoNuSeg_noS = "/mnt/BishalFiles/SamGuided/saves/cryonuseg_noSAM/"

    exptCoNIC_wS = "/mnt/BishalFiles/SamGuided/saves/conic_wSAM/"
    exptCoNIC_noS = "/mnt/BishalFiles/SamGuided/saves/conic_noSAM/"

    dirs = [exptNuInsSeg_wS, exptNuInsSeg_noS, exptCryoNuSeg_wS, exptCryoNuSeg_noS, exptCoNIC_wS, exptCoNIC_noS]
    expt_names = ['NuInsSeg_wSAM', 'NuInsSeg_noSAM', 'CryoNuSeg_wSAM', 'CryoNuSeg_noSAM', 'CoNIC_wSAM', 'CoNIC_noSAM']

    data = []


    # Plot violin plot on each of the above directories
    for expt_dir, name in zip(dirs, expt_names):
        (acc, mAP, mdice, miou, aji, meanloss, mpq) = inference_main(expt_dir, saveImages=saveImages)
        data.append([name, acc, mAP, mdice, miou, aji, meanloss, mpq])

    df = pd.DataFrame(data, columns=['Experiment', 'Accuracy', 'mAP', 'mDice', 'mIoU', 'AJI', 'Mean Loss', 'MPQ'])
    df_melted = df.melt(id_vars=["Experiment"], var_name="Metric", value_name="Value")

    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Metric", y="Value", hue="Experiment", data=df_melted, split=True, inner="quart", linewidth=1.5)
    plt.title('Performance Metrics Across Experiments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')


    plt.savefig('Outputs/PerformanceMetrics.png')



if __name__ == '__main__':
    main(saveImages=True)