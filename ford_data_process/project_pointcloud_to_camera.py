import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import yaml
import transformations
from pose_func import read_calib_yaml, read_txt, read_numpy
import random

# based on https://github.com/Ford/AVData/issues/26
#          https://github.com/Ford/AVData/issues/28

root_folder = "/data/dataset/Ford_AV"
log_id = "2017-08-04-V2-Log5" #"2017-10-26-V2-Log5"
query_size = [1656, 860]
map_ori_gps = [42.294319, -83.223275]

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

def convert_body_yaw_to_360(yaw_body):
    yaw_360 = 0
    # if (yaw_body >= 0.0) and (yaw_body <=90.0):
    #     yaw_360 = 90.0 - yaw_body

    if (yaw_body >90.0) and (yaw_body <=180.0):
        yaw_360 = 360.0 - yaw_body + 90.0
    else:
        yaw_360 = 90.0 - yaw_body

    # if (yaw_body >= -90) and (yaw_body <0.0):
    #     yaw_360 = 90.0 - yaw_body
    #
    # if (yaw_body >= -180) and (yaw_body < -90):
    #     yaw_360 = 90.0 - yaw_body
    return yaw_360


# warp lidar to image, project

calib_folder = os.path.join(root_folder, "V2")
log_folder = os.path.join(root_folder , log_id, 'info_files')

# read the lidar point clouds and project to the corresponding image
with open(os.path.join(calib_folder,"cameraFrontLeft_body.yaml"), 'r') as stream:
    try:
        FL_cameraFrontLeft_body = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(os.path.join(calib_folder,"cameraFrontLeftIntrinsics.yaml"), 'r') as stream:
    try:
        FL_cameraFrontLeftIntrinsics = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# read all query images
FL_image_names = read_txt(log_folder, log_id + '-FL-names.txt')
FL_image_names.pop(0)
nb_query_images = len(FL_image_names)

query_ned = read_numpy(log_folder , "groundview_NED_pose_gt.npy")
query_gps = read_numpy(log_folder , 'groundview_gps.npy') # 'groundview_gps_2.npy'
query_yaws = read_numpy(log_folder , 'groundview_yaws_pose_gt.npy') # 'groundview_yaws_pose.npy'
query_rolls = read_numpy(log_folder, 'groundview_rolls_pose_gt.npy')
query_pitchs = read_numpy(log_folder, 'groundview_pitchs_pose_gt.npy')

# get extrinsics and intrinsics
FL_body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
FL_Intrinsics = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
RR_body = read_calib_yaml(calib_folder, "cameraRearRight_body.yaml")
RR_Intrinsics = read_calib_yaml(calib_folder, "cameraRearRightIntrinsics.yaml")
SL_body = read_calib_yaml(calib_folder, "cameraSideLeft_body.yaml")
SL_Intrinsics = read_calib_yaml(calib_folder, "cameraSideLeftIntrinsics.yaml")
SR_body = read_calib_yaml(calib_folder, "cameraSideRight_body.yaml")
SR_Intrinsics = read_calib_yaml(calib_folder, "cameraSideRightIntrinsics.yaml")

# camera->body
FL_relPose_body = transformations.quaternion_matrix(quat_from_pose(FL_body))
FL_relTrans_body = trans_from_pose(FL_body)
FL_relPose_body[0, 3] = FL_relTrans_body[0]
FL_relPose_body[1, 3] = FL_relTrans_body[1]
FL_relPose_body[2, 3] = FL_relTrans_body[2]

RR_relPose_body = transformations.quaternion_matrix(quat_from_pose(RR_body))
RR_relTrans_body = trans_from_pose(RR_body)
RR_relPose_body[0, 3] = FL_relTrans_body[0]
RR_relPose_body[1, 3] = FL_relTrans_body[1]
RR_relPose_body[2, 3] = FL_relTrans_body[2]

SL_relPose_body = transformations.quaternion_matrix(quat_from_pose(SL_body))
SL_relTrans_body = trans_from_pose(SL_body)
SL_relPose_body[0, 3] = SL_relTrans_body[0]
SL_relPose_body[1, 3] = SL_relTrans_body[1]
SL_relPose_body[2, 3] = SL_relTrans_body[2]

SR_relPose_body = transformations.quaternion_matrix(quat_from_pose(SR_body))
SR_relTrans_body = trans_from_pose(SR_body)
SR_relPose_body[0, 3] = SR_relTrans_body[0]
SR_relPose_body[1, 3] = SR_relTrans_body[1]
SR_relPose_body[2, 3] = SR_relTrans_body[2]

for i in range(nb_query_images):
    # check exist, continue if already exist
    save_folder = os.path.join(root_folder, log_id, 'pcd')
    if os.path.exists(save_folder):
        if os.path.exists(os.path.join(save_folder,FL_image_names[i][:-5]+".pcd")):
            continue

    yaw = query_yaws[i] * np.pi / 180.0
    roll = query_rolls[i] * np.pi / 180.0
    pitch = query_pitchs[i] * np.pi / 180.0
    
    # translate body(X:heading,Y:right,Z:Down) -> point cloud ori(X:North,Y:East,Z:Down)
    # body_relTrans_pc = np.zeros(3)
    # body_relTrans_pc[1], body_relTrans_pc[0] = gps_func.angular_distance_to_xy_distance_v2(map_ori_gps[0],
    #                                                                                        map_ori_gps[1], query_gps[i][0],
    #                                                                                        query_gps[i][1]) # return long(E), lat(N)
    # x, y, z = gps_func.GeodeticToEcef(query_gps[i][0] * np.pi / 180.0, query_gps[i][1] * np.pi / 180.0,
    #                                   query_gps[i][2])
    # yEast, xNorth, zUp = gps_func.EcefToEnu(x, y, z, gps_func.gps_ref_lat, gps_func.gps_ref_long,
    #                                         gps_func.gps_ref_height)
    # body_relTrans_pc[0] = xNorth
    # body_relTrans_pc[1] = yEast
    # body_relTrans_pc[2] = -zUp # z not right
    body_relTrans_pc = query_ned[i]
    
    # get point clouds name base on body pose, get near by 4 pcd
    cur_pcd = (body_relTrans_pc // 64)*64
    other_pcd = cur_pcd + np.sign((body_relTrans_pc%64)/64 - 0.5)*64

    # read point clouds, x:North,y:East
    pcd = []
    for x in (cur_pcd[0], other_pcd[0]):
        for y in (cur_pcd[1], other_pcd[1]):
            #point cloud
            pcd_name = str(int(x))+'_'+str(int(y))+'.pcd'
            pcd_dir = os.path.join(root_folder , log_id, log_id[:10]+'-Map'+log_id[-1], '3d_point_cloud',pcd_name)
            pcd.append(o3d.io.read_point_cloud(pcd_dir))
            # reflectivity
            pcd_dir = os.path.join(root_folder, log_id, log_id[:10]+'-Map'+log_id[-1], 'ground_reflectivity',
                                    pcd_name)  #
            pcd.append(o3d.io.read_point_cloud(pcd_dir))
    total_visible = None

    body_relPose_pc = transformations.compose_matrix(
        angles=np.array([roll,pitch,yaw]), translate=body_relTrans_pc) #[-roll, -pitch, -yaw]

    for grd_folder in ('-RR', '-SL', '-SR', '-FL'): # same point cloud in last camera coordinate
        if grd_folder == '-FL':
            camera_relPose_body = FL_relPose_body
            proj_mat = np.asarray(FL_Intrinsics['P']).reshape(3, 4) # check P
        elif grd_folder == '-RR':
            camera_relPose_body = RR_relPose_body
            proj_mat = np.asarray(RR_Intrinsics['P']).reshape(3, 4)
        elif grd_folder == '-SL':
            camera_relPose_body = SL_relPose_body
            proj_mat = np.asarray(SL_Intrinsics['P']).reshape(3, 4)
        else:
            camera_relPose_body = SR_relPose_body
            proj_mat = np.asarray(SR_Intrinsics['P']).reshape(3, 4)

        # point cloud -> body -> camera
        pc_relPose_camera = inverse_pose(camera_relPose_body) @ inverse_pose(body_relPose_pc)

        # get the transformed 3D points
        # print('ori points:',np.asarray(pcd[0].points)[0])
        pcd_cam = np.transpose(np.asarray(pcd[0].points))
        for pcd_i in pcd[1:]:
            #pcd_i.transform(pc_relPose_camera)
            pcd_cam = np.concatenate((pcd_cam,np.transpose(np.asarray(pcd_i.points))),axis=1)
        pcd_cam = np.concatenate((pcd_cam,np.ones_like(pcd_cam[:1])),axis=0)#[3,n]->[4,n]
        pcd_cam = pc_relPose_camera[:3]@pcd_cam
        # print('trans points:', pcd_cam[:,0])
            
        # non0-visible mask
        non_visible_mask = pcd_cam[2,:] < 0.0
        # print('left points front:', np.sum(~non_visible_mask), '/', pcd_cam.shape[1])
        # project the 3D points to image
        pcd_img = proj_mat[:3,:3] @ pcd_cam + proj_mat[:3,3][:,None]
        pcd_img = pcd_img / pcd_img[2,:]
        # print('project points:', pcd_img[:, 0])

        # update non0-visible mask
        non_visible_mask = non_visible_mask | (pcd_img[0,:] > query_size[0]) | (pcd_img[0,:] < 0.0)
        non_visible_mask = non_visible_mask | (pcd_img[1, :] > query_size[1]) | (pcd_img[1, :] < 0.0)

        # visible mask
        visible_mask = np.invert(non_visible_mask)
        # print('left in image points:', np.sum(visible_mask), '/', pcd_cam.shape[1])

        # get visible point clouds and their projections
        pcd_visible = pcd_cam[:,visible_mask]
        pcd_proj_visible = pcd_img[:, visible_mask]

        # ---------------------------
        if 0: # debug
            # show the visible pcds on image
            image_name = os.path.join(root_folder, log_id, log_id+grd_folder, FL_image_names[i][:-1])
            color_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
            for j in range(pcd_proj_visible.shape[1]):
                cv2.circle(color_image, (np.int32(pcd_proj_visible[0][j]), np.int32(pcd_proj_visible[1][j])), 2, (255, 0, 0), -1)
            plt.imshow(color_image)
            plt.show()
        if total_visible is None:
            total_visible = visible_mask
        else:
            total_visible |= visible_mask
        
    # save points cam
    pcd_save = o3d.geometry.PointCloud()
    pcd_cam = pcd_cam[:,total_visible].T
    rng = np.random.default_rng()
    pcd_cam = rng.choice(pcd_cam, 20000)# sample 20000
    pcd_save.points = o3d.utility.Vector3dVector(pcd_cam)
    save_folder = os.path.join(root_folder, log_id, 'pcd')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    o3d.io.write_point_cloud(os.path.join(save_folder,FL_image_names[i][:-5]+".pcd"), pcd_save)

