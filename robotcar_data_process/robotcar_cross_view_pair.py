# for each query image, find the nearest satellite image, and calculate their distance

#from input_libs import *
import robotcar_gps_coord_func as gps_func
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import csv

root = "/data/dataset/robotcar/"
zoom = 18
Camera_MAX_GAP = 20000 # 20ms
RTK_MAX_GAP = 10000 # 10ms

def read_numpy(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), 'rb') as f:
        cur_file = np.load(f)
    return cur_file

def write_numpy(root_folder, file_name, current_file):
    with open(os.path.join(root_folder, file_name), 'wb') as f:
        np.save(f, current_file)

def read_timestamp(folder, file_name):
    timestamp = []
    with open(os.path.join(folder, file_name)) as ff:
        for line in ff:  # To read a file line by line
            p = line.split()  # To split the elements being read in above line
            if len(p[0]) >= 16:
                timestamp.append(int(p[0]))
    return np.array(timestamp)

def get_timestamp(folder):
    list = os.listdir(folder)
    timestamp = []
    for item in list:
        ts = item.split('.')
        if len(ts[0]) >= 16:
            timestamp.append(int(ts[0]))
    return np.array(timestamp)


def read_csv(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), newline='') as f:
        reader = csv.reader(f)
        cur_file = list(reader)
    return cur_file

def interpolate_poses(rtk_timestamp, rtk_info, front_ts, idx):
    if front_ts == rtk_timestamp[idx]:
        return rtk_info[idx]
    elif front_ts < rtk_timestamp[idx]:
        upper_idx = idx
        lower_idx = idx-1
    else:
        upper_idx = idx+1
        lower_idx = idx
    fractions = (front_ts - rtk_timestamp[lower_idx]) / (rtk_timestamp[upper_idx] - rtk_timestamp[lower_idx])
    positions_interp = (1 - fractions) * np.array(rtk_info[lower_idx][:3]) + fractions * np.array(rtk_info[upper_idx][:3])
    return np.hstack((positions_interp,rtk_info[idx][3:]))


# satellite images gps
gps_center_file = os.path.join(root, 'satellite_gps_center.npy')
Geodetic = np.load(gps_center_file)
ENU_coords_satellite = np.zeros((Geodetic.shape[0], 3))
for i in range(Geodetic.shape[0]):
    x, y, z = gps_func.GeodeticToEcef(Geodetic[i, 0] * np.pi / 180.0, Geodetic[i, 1] * np.pi / 180.0,
                                      Geodetic[i, 2])
    xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, Geodetic[0, 0] * np.pi / 180.0,
                                            Geodetic[0, 1] * np.pi / 180.0, Geodetic[0, 2])
    ENU_coords_satellite[i, 0] = xEast
    ENU_coords_satellite[i, 1] = yNorth
    ENU_coords_satellite[i, 2] = zUp
neigh = NearestNeighbors(n_neighbors=1)    
neigh.fit(ENU_coords_satellite)

# query image gps
# read form txt files
old_log = None
for split in ('train', 'val','test'):
    lines = []
    front_files = np.loadtxt(os.path.join(root, "split", split + '.txt'), dtype="str")
    for front_dir in front_files:
        day_ts = front_dir.split("/")

        log_dir = os.path.join(root, day_ts[0])
        if log_dir != old_log:
            # mono image timestamp
            #left_timestamp = read_timestamp(log_dir, 'mono_left.timestamps')
            # from image list instead of timestamps, not all timestamps has png
            left_timestamp = get_timestamp(os.path.join(log_dir, 'mono_left'))
            neigh_left = NearestNeighbors(n_neighbors=1)
            neigh_left.fit(left_timestamp.reshape(-1,1))
            #right_timestamp = read_timestamp(log_dir, 'mono_right.timestamps')
            right_timestamp = get_timestamp(os.path.join(log_dir, 'mono_right'))
            neigh_right = NearestNeighbors(n_neighbors=1)
            neigh_right.fit(right_timestamp.reshape(-1,1))
            #rear_timestamp = read_timestamp(log_dir, 'mono_rear.timestamps')
            rear_timestamp = get_timestamp(os.path.join(log_dir, 'mono_rear'))
            neigh_rear = NearestNeighbors(n_neighbors=1)
            neigh_rear.fit(rear_timestamp.reshape(-1,1))

            # rtk gt
            rtk_data = read_csv(log_dir, 'rtk.csv')
            # remove the headlines
            rtk_data.pop(0)
            rtk_timestamp = []
            rtk_info = [] #(lat,long,alt,roll,pitch,yaw)
            for line in rtk_data:
                rtk_timestamp.append(int(line[0]))
                rtk_info.append([float(line[1]), float(line[2]), float(line[3]), float(line[11]), float(line[12]), float(line[13])])
            neigh_rtk = NearestNeighbors(n_neighbors=1)
            neigh_rtk.fit(np.array(rtk_timestamp).reshape(-1,1))

            old_log = log_dir

        # from stereo center timestamp to find nearest mono and gt
        front_ts = int(day_ts[-1][:-4])
        # find nearest mono
        ts_key = np.array([[front_ts]])
        _, indices = neigh_left.kneighbors(ts_key, return_distance=True)
        indices = indices.ravel()[0]
        #print(abs(front_ts - left_timestamp[indices]))
        if abs(front_ts - left_timestamp[indices]) > Camera_MAX_GAP:
            continue
        left_ts = left_timestamp[indices]
        _, indices = neigh_right.kneighbors(ts_key, return_distance=True)
        indices = indices.ravel()[0]
        #print(abs(front_ts - right_timestamp[indices]))
        if abs(front_ts - right_timestamp[indices]) > Camera_MAX_GAP:
            continue
        right_ts = right_timestamp[indices]
        _, indices = neigh_rear.kneighbors(ts_key, return_distance=True)
        indices = indices.ravel()[0]
        #print(abs(front_ts - rear_timestamp[indices]))
        if abs(front_ts - rear_timestamp[indices]) > Camera_MAX_GAP:
            continue
        rear_ts = rear_timestamp[indices]

        # find nearest gt
        _, indices = neigh_rtk.kneighbors(ts_key, return_distance=True)
        indices = indices.ravel()[0]
        # near by
        if abs(front_ts - rtk_timestamp[indices]) > RTK_MAX_GAP:
            continue
        # interpolate_pose
        if indices == 0 or indices == len(rtk_timestamp)-1:
            continue
        pose_gt = interpolate_poses(rtk_timestamp, rtk_info, front_ts, indices)
        # pose_gt = rtk_info[indices]

        # get ENU of query
        x, y, z = gps_func.GeodeticToEcef(pose_gt[0] * np.pi / 180.0, pose_gt[1] * np.pi / 180.0, pose_gt[2])
        xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, Geodetic[0, 0] * np.pi / 180.0,
                                                Geodetic[0, 1] * np.pi / 180.0, Geodetic[0, 2])
        ENU_coords_query = np.array([[xEast, yNorth, zUp]])
        _, indices = neigh.kneighbors(ENU_coords_query, return_distance=True)
        indices = indices.ravel()[0]

        # find sat image
        sat_gps = Geodetic[indices]
        SatMap_name = "satellite_" + str(indices) + "_lat_" + str(
            sat_gps[0]) + "_long_" + str(
            sat_gps[1]) + "_zoom_" + str(
            zoom) + "_size_" + str(640) + "x" + str(640) + "_scale_" + str(2) + ".png"
        lines.append([day_ts[0], front_ts, left_ts, right_ts, rear_ts, SatMap_name, pose_gt[0], pose_gt[1], pose_gt[2], pose_gt[3], pose_gt[4], pose_gt[5]])
    # save
    with open(os.path.join(root,"split", split + ".csv"), 'w') as f:
    # create the csv writer
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)






