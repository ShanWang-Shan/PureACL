# this script shows the trajectories
import numpy as np
import os
from kitti_data_process.Kitti_gps_coord_func import angular_distance_to_xy_distance_v2
from pixloc.pixlib.datasets.transformations import euler_matrix
from matplotlib import pyplot as plt

root_dir = '/data/dataset/Kitti'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'
vel_dir = 'velodyne_points/data'
left_color_camera_dir = 'image_02/data'

# ori gps
ori = [49.011094765927, 8.4230185929692, 113.01627349854]

# read the shift pose
def read_numpy(root_folder, file_name):
    with open(os.path.join(root_folder, file_name), 'rb') as f:
        cur_file = np.load(f)
    return cur_file

shift_dir = '/home/users/u7094434/projects/SIDFM/pixloc/kitti_shift/'
shift_R = read_numpy(shift_dir, "pred_R.np")  # pre->gt
shift_T = read_numpy(shift_dir, "pred_T.np")  # pre->gt

# read form txt files
shift_enu = []
gt_enu = []
body2imu = np.array([[1., 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # no shift, x->x, y->-y, z->-z
txt_file_name = os.path.join(root_dir, grdimage_dir, 'kitti_split',  'test_files.txt')
with open(txt_file_name, "r") as txt_f:
    lines = txt_f.readlines()
    i = 0
    for line in lines:
        line = line.strip()
        # check grb file exist
        grb_file_name = os.path.join(root_dir, grdimage_dir, line[:38], left_color_camera_dir,
                                     line[38:].lower())
        if not os.path.exists(grb_file_name):
            print(grb_file_name + ' do not exist!!!')
            continue

        day_dir = line[:10]
        drive_dir = line[:38]
        image_no = line[38:]
        # get location & rotation
        oxts_file_name = os.path.join(root_dir, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
        location = [float(content[0]), float(content[1]), float(content[2])]
        roll, pitch, heading = float(content[3]), float(content[4]), float(content[5])
        imu2ENU = euler_matrix(roll, pitch, heading)
        body2ENU = imu2ENU@body2imu

        east, north = angular_distance_to_xy_distance_v2(ori[0], ori[1], location[0], location[1])
        up = location[2] - ori[2]
        cur_enu = np.array([east,north,up])

        gt_enu.append(cur_enu)

        shift_enu.append(cur_enu + body2ENU[:3,:3]@shift_T[i])
        # if i == 5775 or i == 5780:
        #     print(grb_file_name)
        i += 1
gt_enu = np.array(gt_enu)
shift_enu = np.array(shift_enu)


# plot the trajectories
Min_Pose = 3200 #5775#5700 #3200 #550
Max_Pose = 4953#-1 #5780#5950 #1300
plt.plot(gt_enu[Min_Pose:Max_Pose, 0], gt_enu[Min_Pose:Max_Pose, 1], alpha=0.5, color='r', marker='o', markersize=1) #linewidth=1) #
plt.plot(shift_enu[Min_Pose:Max_Pose, 0], shift_enu[Min_Pose:Max_Pose, 1], alpha=0.5, color='b', marker='o', markersize=1 )# linewidth=1)#

plt.xlabel("East(m)")
plt.ylabel("North(m)")
plt.legend()
plt.show()

temp = 0


