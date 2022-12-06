from input_libs import *
from pose_func import read_csv, read_txt, write_numpy
root_folder = "/data/dataset/Ford_AV/"

log_id = "2017-10-26-V2-Log6" #"2017-08-04-V2-Log6" #
info_folder = 'info_files'
log_folder = os.path.join(root_folder , log_id, info_folder)


# read the ground-truth yaw angle from the file pose_ground_truth
pose_data = read_csv(log_folder , "pose_ground_truth.csv")
# remove the headlines
pose_data.pop(0)

pose_times = np.zeros((len(pose_data), 1))
pose_quat_x = np.zeros((len(pose_data)))
pose_quat_y = np.zeros((len(pose_data)))
pose_quat_z = np.zeros((len(pose_data)))
pose_quat_w = np.zeros((len(pose_data)))

pose_roll = np.zeros((len(pose_data)))
pose_pitch = np.zeros((len(pose_data)))
pose_yaw = np.zeros((len(pose_data)))

pose_rotation = np.zeros((4,4,len(pose_data)))
pose_NED = np.zeros((len(pose_data),3))

for i, line in zip(range(len(pose_data)), pose_data):
    pose_timeStamp = float(line[0]) / 1000.0
    pose_times[i] = pose_timeStamp
    pose_quat_x[i] = float(line[13])
    pose_quat_y[i] = float(line[14])
    pose_quat_z[i] = float(line[15])
    pose_quat_w[i] = float(line[16])
    pose_rotation[:,:,i] = transformations.quaternion_matrix([pose_quat_w[i], pose_quat_x[i], pose_quat_y[i], pose_quat_z[i]])

    euler_angles = transformations.euler_from_matrix(pose_rotation[:,:,i])

    pose_roll[i] = euler_angles[0]*180.0/np.pi
    pose_pitch[i] = euler_angles[1]*180.0/np.pi
    pose_yaw[i] = euler_angles[2] * 180.0 / np.pi
    # print(pose_rotation[:,:,i])

    # NED pose
    pose_NED[i,0] = float(line[9])
    pose_NED[i,1] = float(line[10])
    pose_NED[i,2] = float(line[11])

# read the time of each query image, and fetch the yaw angle of each query image

# 1. get the image names
FL_image_names = read_txt(log_folder, log_id + '-FL-names.txt')
FL_image_names.pop(0)

nb_query_images = len(FL_image_names)

image_times = np.zeros((len(FL_image_names), 1))
for i in range(len(FL_image_names)):
    image_times[i] = float(FL_image_names[i][:-5])

# 3. for each query image time, find the nearest gps tag

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(pose_times)
# KNN search given the image utms
distances, indices = neigh.kneighbors(image_times, return_distance=True)
distances = distances.ravel()
indices = indices.ravel()

query_image_yaws = pose_yaw[indices]
query_image_rolls = pose_roll[indices]
query_image_pitchs = pose_pitch[indices]
query_image_NED = pose_NED[indices]

# save the yaw angles of qeury images
write_numpy(log_folder,  'groundview_yaws_pose_gt.npy', query_image_yaws)
write_numpy(log_folder,  'groundview_rolls_pose_gt.npy', query_image_rolls)
write_numpy(log_folder,  'groundview_pitchs_pose_gt.npy', query_image_pitchs)
write_numpy(log_folder,  'groundview_NED_pose_gt.npy', query_image_NED)


x = np.linspace(0, query_image_yaws.shape[0]-1, query_image_yaws.shape[0])
plt.plot(x, query_image_yaws)
plt.show()

print("finish")

