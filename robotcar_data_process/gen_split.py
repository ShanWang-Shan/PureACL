# gen train, val, text split files

import robotcar_gps_coord_func as gps_func
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv

root = "split"

# satellite images gps
for file_name in ('lookup_training.txt', 'lookup_validation.txt','lookup_test.txt'):
    content = np.loadtxt(os.path.join(root, file_name), dtype="str")
    np.savetxt(os.path.join(root, file_name[7:]), content[:,0], fmt="%s")






