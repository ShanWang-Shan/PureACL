# get the overall gps coverage
import os
import numpy as np
import robotcar_gps_coord_func as gps_func
import math
import csv

root_folder = "/data/robotcar"
log_id = "2015-06-26-08-09-43"

def read_csv(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), newline='') as f:
        reader = csv.reader(f)
        cur_file = list(reader)
    return cur_file

def write_numpy(root_folder, file_name, current_file):
    with open(os.path.join(root_folder, file_name), 'wb') as f:
        np.save(f, current_file)

log_folder = os.path.join(root_folder, log_id)

rtk_data = read_csv(log_folder, "rtk.csv")
# remove the headlines
rtk_data.pop(0)
# save timestamp -- >> gps, angle
rtk_times = np.zeros((len(rtk_data), 1))
rtk_gps = np.zeros((len(rtk_data), 3))
sat_gsd = np.zeros((len(rtk_data)))
for i, line in zip(range(len(rtk_data)), rtk_data):
    rtk_times[i] = float(line[0])
    rtk_gps[i,0] = float(line[1])
    rtk_gps[i,1] = float(line[2])
    rtk_gps[i,2] = float(line[3])
    sat_gsd[i] = 156543.03392 * np.cos(rtk_gps[i,0]*np.pi/180.0) / np.power(2, 20) / 2.0 # a scale at 2 when downloading the dataset

# I use a consevative way
sat_gsd_min = np.amin(sat_gsd)

# convert all gps coordinates to NED coordinates
# use the first gps as reference
ENU_coords = np.zeros((len(rtk_data), 3))
for i in range(len(rtk_data)):
    x,y,z = gps_func.GeodeticToEcef(rtk_gps[i,0]*np.pi/180.0, rtk_gps[i,1]*np.pi/180.0, rtk_gps[i,2])
    xEast,yNorth, zUp = gps_func.EcefToEnu( x,  y,  z, rtk_gps[0,0]*np.pi/180.0, rtk_gps[0,1]*np.pi/180.0, rtk_gps[0,2])
    ENU_coords[i,0] = xEast
    ENU_coords[i, 1] = yNorth
    ENU_coords[i, 2] = zUp

ENU_coords_max = np.amax(ENU_coords, axis=0) + 1.0
ENU_coords_min = np.amin(ENU_coords, axis=0) - 1.0

ENU_coords_length = ENU_coords_max - ENU_coords_min

# the coverage of each satellite image
# I use a scale factor for ensurance
scale_factor = 0.25
imageSize = 1200.0
cell_length = sat_gsd_min * imageSize * scale_factor


cell_length_sqare = cell_length * cell_length
delta = cell_length + 1.0

numRows = math.ceil((ENU_coords_max[0] - ENU_coords_min[0]) / delta)
numCols = math.ceil((ENU_coords_max[1] - ENU_coords_min[1]) / delta)


hashTab = []

for i in range(numRows):
    hashTab.append([])
    for j in range(numCols):
        hashTab[i].append([])

inds= np.ceil((ENU_coords[:,:2].T -  ENU_coords_min[:2,None]) / delta  )

for i in range(len(rtk_data)):
    rowIndx = int(inds[0,i]) - 1
    colIndx = int(inds[1,i]) - 1
    hashTab[rowIndx][colIndx].append(i)

# check non-empty cells

numNonEmptyCells = 0

cellCenters = []

for i in range(numRows):
    for j in range(numCols):
        if len(hashTab[i][j]) > 0:
            # this is a valid cell
            center_x = i * delta  + 0.5 * delta
            center_y = j * delta + 0.5 * delta
            center_z = np.sum(ENU_coords[hashTab[i][j],2]) / len(hashTab[i][j])

            avg_x = np.sum(ENU_coords[hashTab[i][j],0]) / len(hashTab[i][j]) - ENU_coords_min[0]
            avg_y = np.sum(ENU_coords[hashTab[i][j], 1]) / len(hashTab[i][j]) - ENU_coords_min[1]

            numNonEmptyCells += 1
            cellCenters.append([center_x, center_y, center_z])

cellCenters = np.asarray(cellCenters)

cellCenters[:,0] += ENU_coords_min[0]
cellCenters[:,1] += ENU_coords_min[1]

# after collect cellCenters, transform back to Geodetic coordinates system

Geodetic = np.zeros((cellCenters.shape[0], 3))

for i in range(cellCenters.shape[0]):
    x_ecef, y_ecef, z_ecef = gps_func.EnuToEcef( cellCenters[i,0],  cellCenters[i,1], cellCenters[i,2], rtk_gps[0,0]*np.pi/180.0, rtk_gps[0,1]*np.pi/180.0, rtk_gps[0,2])
    lat, lon, h = gps_func.EcefToGeodetic( x_ecef, y_ecef, z_ecef)
    Geodetic[i, 0] = lat
    Geodetic[i, 1] = lon
    Geodetic[i, 2] = h

# save the gps coordinates of satellite images
write_numpy(log_folder, 'satellite_gps_center.npy', Geodetic)





