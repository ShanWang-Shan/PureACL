# for each query image, find the nearest satellite image, and calculate their distance

from input_libs import *
from angle_func import convert_body_yaw_to_360
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy, read_csv, write_numpy
import gps_coord_func as gps_func

root_folder = "/data/dataset/Ford_AV"

log_id = "2017-08-04-V2-Log6" #"2017-10-26-V2-Log6"#
info_dir = 'info_files'

log_folder = os.path.join(root_folder, log_id, info_dir)


Geodetic = read_numpy(log_folder, 'satellite_gps_center.npy')

# first, get the query image location
# -----------------------------------------------

# 1. get the image names
imageNames = read_txt(log_folder, log_id + '-FL-names.txt')
imageNames.pop(0)

image_times = np.zeros((len(imageNames), 1))
for i in range(len(imageNames)):
    image_times[i] = float(imageNames[i][:-5])


# 2. read the gps times and data
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
    gps_times[i] = gps_timeStamp / 1000.0
    gps_lat[i] = float(line[10])
    gps_long[i] = float(line[11])
    gps_height[i] = float(line[12])
    sat_gsd[i] = 156543.03392 * np.cos(gps_lat[i]*np.pi/180.0) / np.power(2, 20) / 2.0 # a scale at 2 when downloading the dataset

# 3. for each query image time, find the nearest gps tag

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(gps_times)
# KNN search given the image utms
distances, indices = neigh.kneighbors(image_times, return_distance=True)
distances = distances.ravel()
indices = indices.ravel()


NED_coords_query = np.zeros((len(imageNames), 3))

gps_query = np.zeros((len(imageNames), 3))

for i in range(len(imageNames)):
    x,y,z = gps_func.GeodeticToEcef(gps_lat[indices[i]]*np.pi/180.0, gps_long[indices[i]]*np.pi/180.0, gps_height[indices[i]])
    xEast,yNorth,zUp = gps_func.EcefToEnu( x,  y,  z, gps_lat[0]*np.pi/180.0, gps_long[0]*np.pi/180.0, gps_height[0])
    NED_coords_query[i, 0] = xEast
    NED_coords_query[i, 1] = yNorth
    NED_coords_query[i, 2] = zUp

    gps_query[i, 0] = gps_lat[indices[i]]
    gps_query[i, 1] = gps_long[indices[i]]
    gps_query[i, 2] = gps_height[indices[i]]

NED_coords_satellite = np.zeros((Geodetic.shape[0], 3))
for i in range(Geodetic.shape[0]):
    x,y,z = gps_func.GeodeticToEcef(Geodetic[i,0]*np.pi/180.0, Geodetic[i,1]*np.pi/180.0, Geodetic[i,2])
    xEast,yNorth,zUp = gps_func.EcefToEnu( x,  y,  z, gps_lat[0]*np.pi/180.0, gps_long[0]*np.pi/180.0, gps_height[0])
    NED_coords_satellite[i, 0] = xEast
    NED_coords_satellite[i, 1] = yNorth
    NED_coords_satellite[i, 2] = zUp


# for each query, find the nearest satellite

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(NED_coords_satellite)
# KNN search given the image utms
distances, indices = neigh.kneighbors(NED_coords_query, return_distance=True)
distances = distances.ravel()
indices = indices.ravel()

print("max distance: {};    min distance: {} ".format(np.amax(distances), np.amin(distances)))

# save the ground-view query to satellite matching pair
# save the gps coordinates of query images
write_numpy(log_folder , 'groundview_satellite_pair.npy', indices)
write_numpy(log_folder , 'groundview_gps.npy', gps_query)






