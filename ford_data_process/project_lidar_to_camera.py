import open3d as o3d
import glob
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import yaml
import transformations
import cv2

# based on https://github.com/Ford/AVData/issues/26
#          https://github.com/Ford/AVData/issues/28


def quat_from_pose(trans):

    w = trans['transform']['rotation']['w']
    x = trans['transform']['rotation']['x']
    y = trans['transform']['rotation']['y']
    z = trans['transform']['rotation']['z']

    return [w,x,y,z]

def trans_from_pose(trans):

    x = trans['transform']['translation']['x']
    y = trans['transform']['translation']['y']
    z = trans['transform']['translation']['z']

    return [x,y,z]


def inverse_pose(pose):

    pose_inv = np.identity(4)
    pose_inv[:3,:3] = np.transpose(pose[:3,:3])
    pose_inv[:3, 3] = - pose_inv[:3,:3] @ pose[:3,3]

    return pose_inv


# warp lidar to image, project

lidar_folder = "/media/yanhao/8tb/00_data_ford/2017-08-04/V2/Log1/2017_08_04_V2_Log1_lidar_blue_pointcloud/"

image_floder = "/media/yanhao/8tb/00_data_ford/2017-08-04/V2/Log1/2017-08-04-V2-Log1-FL/"

calib_folder = "/media/yanhao/8tb/00_data_ford/Calibration-V2/"

# get the nb of scans

pcd_names = glob.glob(lidar_folder + '*.pcd')
image_names = glob.glob(image_floder + '*.png')

pcd_time_stamps = np.zeros((len(pcd_names),1))

for i, pcd_name in zip(range(len(pcd_names)), pcd_names):
    pcd_time_stamp = float(os.path.split(pcd_name)[-1][:-4])
    pcd_time_stamps[i] = pcd_time_stamp

image_time_stamps = np.zeros((len(image_names),1))

for i, image_name in zip(range(len(image_names)), image_names):
    image_time_stamp = float(os.path.split(image_name)[-1][:-4])
    image_time_stamps[i] = image_time_stamp

neigh = NearestNeighbors(n_neighbors=1)

neigh.fit(image_time_stamps)

# KNN search given the image utms
# find the nearest lidar scan
distances, indices = neigh.kneighbors(pcd_time_stamps, return_distance=True)
distances = distances.ravel()
indices = indices.ravel()

lidar_matched_images = [image_names[indices[i]] for i in range(indices.shape[0])]

# read the lidar point clouds and project to the corresponding image

with open(calib_folder + "cameraFrontLeft_body.yaml", 'r') as stream:
    try:
        FL_cameraFrontLeft_body = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(calib_folder + "cameraFrontLeftIntrinsics.yaml", 'r') as stream:
    try:
        FL_cameraFrontLeftIntrinsics = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


with open(calib_folder + "lidarBlue_body.yaml", 'r') as stream:
    try:
        lidarBlue_body = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# transform 3D points to camera coordinate system
FL_relPose_body = transformations.quaternion_matrix(quat_from_pose(FL_cameraFrontLeft_body))
FL_relTrans_body = trans_from_pose(FL_cameraFrontLeft_body)
FL_relPose_body[0,3] = FL_relTrans_body[0]
FL_relPose_body[1,3] = FL_relTrans_body[1]
FL_relPose_body[2,3] = FL_relTrans_body[2]

LidarBlue_relPose_body = transformations.quaternion_matrix(quat_from_pose(lidarBlue_body))
LidarBlue_relTrans_body = trans_from_pose(lidarBlue_body)
LidarBlue_relPose_body[0,3] = LidarBlue_relTrans_body[0]
LidarBlue_relPose_body[1,3] = LidarBlue_relTrans_body[1]
LidarBlue_relPose_body[2,3] = LidarBlue_relTrans_body[2]

LidarBlue_relPose_FL = inverse_pose(FL_relPose_body) @ LidarBlue_relPose_body

# camera projection matrix
FL_proj_mat = np.asarray(FL_cameraFrontLeftIntrinsics['P']).reshape(3,4)

# image size
image_width = 1656
image_height = 860
for i in range(indices.shape[0]):

    # read image
    img_0 = cv2.imread(lidar_matched_images[i], cv2.IMREAD_GRAYSCALE)
    # read point clouds
    pcd1 = o3d.io.read_point_cloud(pcd_names[i])
    # get the transformed 3D points
    pcd1.transform(LidarBlue_relPose_FL)
    pcd_cam = np.transpose(np.asarray(pcd1.points))
    # non0-visible mask
    non_visible_mask = pcd_cam[2,:] < 0.0
    # project the 3D points to image
    pcd_img = FL_proj_mat[:3,:3] @ pcd_cam + FL_proj_mat[:3,3][:,None]
    pcd_img = pcd_img / pcd_img[2,:]

    # update non0-visible mask
    non_visible_mask = non_visible_mask | (pcd_img[0,:] > image_width) | (pcd_img[0,:] < 0.0)
    non_visible_mask = non_visible_mask | (pcd_img[1, :] > image_height) | (pcd_img[1, :] < 0.0)

    # visible mask
    visible_mask = np.invert(non_visible_mask)

    # get visible point clouds and their projections
    pcd_visible = pcd_cam[:,visible_mask]
    pcd_proj_visible = pcd_img[:, visible_mask]


    # ---------------------------
    # show the pcds
    pcd_visible_o3d = o3d.geometry.PointCloud()
    pcd_visible_o3d.points = o3d.utility.Vector3dVector(pcd_visible.T)
    o3d.io.write_point_cloud("temp.ply", pcd_visible_o3d)

    # ---------------------------
    # show the visible pcds on image
    color_image = cv2.cvtColor(img_0, cv2.COLOR_GRAY2RGB)
    for j in range(pcd_proj_visible.shape[1]):
        cv2.circle(color_image, (np.int32(pcd_proj_visible[0][j]), np.int32(pcd_proj_visible[1][j])), 2, (255, 0, 0), -1)
    plt.imshow(color_image)
    plt.title(os.path.split(lidar_matched_images[i])[-1])
    plt.show()

