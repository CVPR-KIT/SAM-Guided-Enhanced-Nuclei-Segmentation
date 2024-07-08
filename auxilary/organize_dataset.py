from tqdm import tqdm
import os
import shutil
import json

# Define the path to the data
data_path = '/mnt/Datasets/NuInsSeg/original/'

# Define the path to the organized data
organized_data_path = '/mnt/Datasets/NuInsSeg/experiment/'
organized_data_path_train = organized_data_path + 'orig_train/'
organized_data_path_val = organized_data_path + 'orig_validation/'
organized_data_path_test = organized_data_path + 'orig_test/'

# Create the organized data directory
os.makedirs(organized_data_path, exist_ok=True)
os.makedirs(organized_data_path_train, exist_ok=True)
os.makedirs(organized_data_path_train + 'images/', exist_ok=True)
os.makedirs(organized_data_path_train + 'labels/', exist_ok=True)
os.makedirs(organized_data_path_val, exist_ok=True)
os.makedirs(organized_data_path_val + 'images/', exist_ok=True)
os.makedirs(organized_data_path_val + 'labels/', exist_ok=True)
os.makedirs(organized_data_path_test, exist_ok=True)
os.makedirs(organized_data_path_test + 'images/', exist_ok=True)
os.makedirs(organized_data_path_test + 'labels/', exist_ok=True)

# create meta
meta = {}
count_tr = 0
count_v = 0
count_t = 0

# Define the path to the original data
for dirs_ in tqdm(os.listdir(data_path)):
    tissue_path = data_path + dirs_ + '/tissue images/'
    mask_path = data_path + dirs_ + '/mask binary/'
    # copy 70% to train, 20 % to val and 10% to test and create meta for each dirs_ 
    for i, (tissue_img, mask_img) in enumerate(zip(os.listdir(tissue_path), os.listdir(mask_path))):
        # skip if the file is not an image
        if not tissue_img.endswith('.png') or not mask_img.endswith('.png'):
            continue
        if not dirs_ in meta.keys():
            meta[dirs_] = {}
            meta[dirs_]['train'] = 0
            meta[dirs_]['val'] = 0
            meta[dirs_]['test'] = 0


        if i < len(os.listdir(tissue_path)) * 0.7:
            shutil.copy(tissue_path + tissue_img, organized_data_path_train+ 'images/' + str(count_tr) + '.png')
            shutil.copy(mask_path + mask_img, organized_data_path_train + 'labels/' + str(count_tr) + '.png')
            meta[dirs_]['train'] += 1
            count_tr += 1
        elif i < len(os.listdir(tissue_path)) * 0.85:
            shutil.copy(tissue_path + tissue_img, organized_data_path_val + 'images/' + str(count_v) + '.png')
            shutil.copy(mask_path + mask_img, organized_data_path_val + 'labels/' + str(count_v) + '.png')
            meta[dirs_]['val'] += 1
            count_v += 1
        else:
            shutil.copy(tissue_path + tissue_img, organized_data_path_test + 'images/' + str(count_t) + '.png')
            shutil.copy(mask_path + mask_img, organized_data_path_test + 'labels/' + str(count_t) + '.png')
            meta[dirs_]['test'] += 1
            count_t += 1
        
        
        
        

# save meta
f = open(organized_data_path + 'meta.json', 'w')
f.write(json.dumps(meta, indent=4))
f.close()
print('Data organized successfully!')