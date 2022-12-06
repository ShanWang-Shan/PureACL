# for each query image, find the nearest satellite image, and calculate their distance

from input_libs import *
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy, read_csv, write_numpy

root_folder = "../"

log_id = "2017-08-04-V2-Log5" #"2017-10-26-V2-Log5"
info_folder = 'info_files'

log_folder = os.path.join(root_folder, log_id, info_folder)

# first, get the query image location
# -----------------------------------------------

# 1. get the image names
imageNames = read_txt(log_folder, log_id + '-FL-names.txt')
imageNames.pop(0)

image_times = np.zeros((len(imageNames), 1))
for i in range(len(imageNames)):
    image_times[i] = float(imageNames[i][:-5])


# # 2. read the vel times and data
# vel_data = read_csv(log_folder , "velocity_raw.csv")
# # remove the headlines
# vel_data.pop(0)
# # save timestamp -- >> vel
# vel_dict = {}
# vel_times = np.zeros((len(vel_data), 1))
# vel_x = np.zeros((len(vel_data)))
# vel_y = np.zeros((len(vel_data)))
# vel_z = np.zeros((len(vel_data)))
# for i, line in zip(range(len(vel_data)), vel_data):
#     vel_timeStamp = float(line[0])
#     vel_times[i] = vel_timeStamp / 1000.0
#     vel_x[i] = float(line[8]) # !!!!
#     vel_y[i] = float(line[9])
#     vel_z[i] = float(line[10])
# # 3. for each query image time, find the nearest vel tag
# neigh = NearestNeighbors(n_neighbors=1)#20000) # not including every 3D points
# neigh.fit(vel_times)

lidar_folder = os.path.join(root_folder, log_id, 'lidar_blue_pointcloud')
pcd_names = glob.glob(lidar_folder + '/*.pcd')
pcd_time_stamps = np.zeros((len(pcd_names),1))
for i, pcd_name in zip(range(len(pcd_names)), pcd_names):
    pcd_time_stamp = float(os.path.split(pcd_name)[-1][:-4])
    pcd_time_stamps[i] = pcd_time_stamp
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(pcd_time_stamps)

# KNN search given the image utms
distances, indices = neigh.kneighbors(image_times, return_distance=True)
distances = distances.ravel()
indices = indices.ravel()

# NED_coords_query = np.zeros((len(imageNames), 3))
#
# vel_query = np.zeros((len(imageNames), 3))
#
# for i in range(len(imageNames)):
#     x = vel_x[indices[i]]
#     y = vel_y[indices[i]]
#     z = vel_z[indices[i]]
#
#     vel_query[i, 0] = x
#     vel_query[i, 1] = y
#     vel_query[i, 2] = z

# save the ground-view query to satellite matching pair
# save the gps coordinates of query images
# write_numpy(root_folder , 'vel_per_images.npy', vel_query)
write_numpy(log_folder , 'vel_time_per_images.npy', pcd_time_stamps[indices])






