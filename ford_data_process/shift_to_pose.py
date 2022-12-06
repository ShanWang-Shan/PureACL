# check the matching relationship between cross-view images

from input_libs import *
from angle_func import convert_body_yaw_to_360
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy
import gps_coord_func as gps_func

root_folder = "/data/dataset/Ford_AV"
log_id = "2017-10-26-V2-Log4"

log_folder = os.path.join(root_folder, log_id)
info_folder = os.path.join(log_folder,'info_files')
FL_image_names = read_txt(info_folder, log_id + '-FL-names.txt')
FL_image_names.pop(0)
nb_query_images = len(FL_image_names)
groundview_gps = read_numpy(info_folder, 'groundview_gps.npy') # 'groundview_gps_2.npy'
groundview_yaws = read_numpy(info_folder, 'groundview_yaws_pose_gt.npy') # 'groundview_yaws_pose.npy'
shift_pose = read_numpy(log_folder, 'shift_pose.npy') # yaw in pi, north, est

pre_pose = []
for i in range(nb_query_images):
    shift_pose = read_numpy(log_folder, 'shift_pose.npy')  # yaw in pi, north, est
    query_gps = groundview_gps[i]
    # print(query_gps)
    # print(groundview_yaws[i])
    # using the original gps reference and calculate the offset of the ground-view query
    east, north = gps_func.angular_distance_to_xy_distance(query_gps[0], query_gps[1])
    east = east + shift_pose[i][2]
    north = north + shift_pose[i][1]
    yaw = groundview_yaws[i]+ shift_pose[i][0]*180./np.pi
    # turn to +- pi
    if abs(yaw) > 180.:
        yaw = yaw - np.sign(yaw)*180
    pre_pose.append([yaw,north,east])

with open(os.path.join(log_folder, 'pred_pose.npy'), 'wb') as f:
    np.save(f, pre_pose)
