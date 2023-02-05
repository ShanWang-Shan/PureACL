# gen train, val, text split files

import robotcar_gps_coord_func as gps_func
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv

root = "/data/dataset/robotcar"
save_dir = "/data/dataset/robotcar/split"

train_dataset = ['2015-06-26-08-09-43', '2015-08-28-09-50-22', '2015-03-03-11-31-36', '2014-12-12-10-45-15', '2015-10-30-13-52-14',
        '2015-04-24-08-15-07', '2015-11-12-13-27-51', '2015-08-17-13-30-19', '2015-03-17-11-08-44', '2015-11-10-11-55-47',
        '2014-11-28-12-07-13']
val_dataset = ['2015-08-17-10-42-18']
test_dataset = ['2015-08-14-14-54-57', '2015-08-12-15-04-18', '2015-02-10-11-58-05']

# for file_name in ('lookup_training.txt', 'lookup_validation.txt','lookup_test.txt'):
#     content = np.loadtxt(os.path.join(root, file_name), dtype="str")
#     np.savetxt(os.path.join(root, file_name[7:]), content[:,0], fmt="%s")

for name, dataset in zip(('train.txt', 'val.txt', 'test.txt'),(train_dataset, val_dataset, test_dataset )):
    dataset_filelist = []
    for folder in dataset:
        file_list = os.listdir(os.path.join(root,folder,'stereo/centre'))
        file_list = [os.path.join(folder,'stereo/centre', file) for file in file_list]
        if len(dataset_filelist) == 0:
            # remove first 200 frames
            dataset_filelist.extend(file_list[200:])
        else:
            dataset_filelist.extend(file_list)
    dataset_filelist = np.array(dataset_filelist)
    np.savetxt(os.path.join(save_dir, name), dataset_filelist, fmt="%s")




