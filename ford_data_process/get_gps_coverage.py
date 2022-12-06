# get the overall gps coverage
from input_libs import *
from angle_func import convert_body_yaw_to_360
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy, read_csv, write_numpy
import gps_coord_func as gps_func

root_folder = "/data/dataset/Ford_AV"
log_id = "2017-10-26-V2-Log6"#"2017-08-04-V2-Log6" #
save_dir = 'info_files'

log_folder = os.path.join(root_folder, log_id, save_dir)


gps_data = read_csv(log_folder , "gps.csv")
# remove the headlines
gps_data.pop(0)
# save timestamp -- >> gps
gps_dict = {}
gps_times = np.zeros((len(gps_data), 1))
gps_lat = np.zeros((len(gps_data)))
gps_long = np.zeros((len(gps_data)))
gps_height = np.zeros((len(gps_data)))
sat_gsd = np.zeros((len(gps_data)))
for i, line in zip(range(len(gps_data)), gps_data):
    gps_timeStamp = float(line[0])
    gps_latLongAtt = "%s_%s_%s" % (line[10], line[11], line[12])
    gps_dict[gps_timeStamp] = gps_latLongAtt
    gps_times[i] = gps_timeStamp
    gps_lat[i] = float(line[10])
    gps_long[i] = float(line[11])
    gps_height[i] = float(line[12])
    sat_gsd[i] = 156543.03392 * np.cos(gps_lat[i]*np.pi/180.0) / np.power(2, 20) / 2.0 # a scale at 2 when downloading the dataset

# I use a consevative way
sat_gsd_min = np.amin(sat_gsd)

# convert all gps coordinates to NED coordinates
# use the first gps as reference
NED_coords = np.zeros((len(gps_data), 3))
for i in range(len(gps_data)):
    x,y,z = gps_func.GeodeticToEcef(gps_lat[i]*np.pi/180.0, gps_long[i]*np.pi/180.0, gps_height[i])
    xEast,yNorth,zUp = gps_func.EcefToEnu( x,  y,  z, gps_lat[0]*np.pi/180.0, gps_long[0]*np.pi/180.0, gps_height[0])
    NED_coords[i,0] = xEast
    NED_coords[i, 1] = yNorth
    NED_coords[i, 2] = zUp

NED_coords_max = np.amax(NED_coords, axis=0) + 1.0
NED_coords_min = np.amin(NED_coords, axis=0) - 1.0

NED_coords_length = NED_coords_max - NED_coords_min

# the coverage of each satellite image
# I use a scale factor for ensurance
scale_factor = 0.25
imageSize = 1200.0
cell_length = sat_gsd_min * imageSize * scale_factor


cell_length_sqare = cell_length * cell_length
delta = cell_length + 1.0

numRows = math.ceil((NED_coords_max[0] - NED_coords_min[0]) / delta)
numCols = math.ceil((NED_coords_max[1] - NED_coords_min[1]) / delta)


hashTab = []

for i in range(numRows):
    hashTab.append([])
    for j in range(numCols):
        hashTab[i].append([])

inds= np.ceil((NED_coords[:,:2].T -  NED_coords_min[:2,None]) / delta  )

for i in range(len(gps_data)):
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
            # all_z = NED_coords[hashTab[i][j],2]
            center_z = np.sum(NED_coords[hashTab[i][j],2]) / len(hashTab[i][j])

            avg_x = np.sum(NED_coords[hashTab[i][j],0]) / len(hashTab[i][j]) - NED_coords_min[0]
            avg_y = np.sum(NED_coords[hashTab[i][j], 1]) / len(hashTab[i][j]) - NED_coords_min[1]

            numNonEmptyCells += 1
            cellCenters.append([center_x, center_y, center_z])

cellCenters = np.asarray(cellCenters)

cellCenters[:,0] += NED_coords_min[0]
cellCenters[:,1] += NED_coords_min[1]

# after collect cellCenters, transform back to Geodetic coordinates system

Geodetic = np.zeros((cellCenters.shape[0], 3))

for i in range(cellCenters.shape[0]):
    x_ecef, y_ecef, z_ecef = gps_func.EnuToEcef( cellCenters[i,0],  cellCenters[i,1], cellCenters[i,2], gps_lat[0]*np.pi/180.0, gps_long[0]*np.pi/180.0, gps_height[0])
    lat, lon, h = gps_func.EcefToGeodetic( x_ecef, y_ecef, z_ecef)
    Geodetic[i, 0] = lat
    Geodetic[i, 1] = lon
    Geodetic[i, 2] = h

# save the gps coordinates of satellite images
write_numpy(log_folder, 'satellite_gps_center.npy', Geodetic)





