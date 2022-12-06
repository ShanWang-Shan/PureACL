import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors
import os.path
import glob
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import transformations
import yaml
import math

root_folder = "/data/dataset/Ford_AV"
log_id = "2017-08-04-V2-Log3" #"2017-10-26-V2-Log6" #

log_folder = os.path.join(root_folder, log_id)

for dir in os.listdir(log_folder):
    # log-FL/RR~
    subdir = os.path.join(log_folder, dir)
    if not os.path.isdir(subdir):
        continue

    if ('-FL' in dir) or ('-RR' in dir) or ('-SL' in dir) or ('-SR' in dir):
        print('process '+dir)
    else:
        continue

    file_list = os.listdir(subdir)
    file_list.sort()

    # # ignore reconstruction images
    # if '2017-10-26-V2-Log1' in dir:
    #     file_list = file_list[2900:4900]+file_list[5200:8300]
    # if '2017-08-04-V2-Log1' in dir:
    #     file_list = file_list[2000:3601] + file_list[4500:7900]
    # if '2017-10-26-V2-Log5' in dir:
    #     file_list = file_list[200:2300]
    # if '2017-08-04-V2-Log5' in dir:
    #     file_list = file_list[:3500] + file_list[7000:]

    txt_file_name = os.path.join(log_folder, 'info_files', dir + '-names.txt')
    with open(txt_file_name, 'w') as f:
        f.write(str(dir) + '\n')
        for name in file_list:
            f.write(str(name)+'\n')