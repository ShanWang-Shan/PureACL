# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:44:19 2020

@author: loocy
"""

from pixloc.pixlib.datasets.base_dataset import BaseDataset
import numpy as np
import os
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
#import pixloc.pixlib.datasets.Kitti_gps_coord_func as gps_func
import ford_data_process.gps_coord_func as gps_func
import random
import cv2
from glob import glob
from pixloc.pixlib.datasets.transformations import quaternion_matrix, euler_matrix
from pixloc.pixlib.geometry import Camera, Pose
import open3d as o3d
import yaml
ImageFile.LOAD_TRUNCATED_IMAGES = True # add for 'broken data stream'

points_type = 0 # 0: use 3d points from map, 1: use 3d points from lidar blue
gt_from_gps = True #ture: pose gt from gps, False: pose gt from NED pose gt

pre_init = False
sat_dir = 'Satellite_Images_18'
sat_zoom = 18
log_id_train = "2017-08-04-V2-Log4"
log_id_val = "2017-07-24-V2-Log4"
log_id_test = "2017-10-26-V2-Log4"
map_points_dir = 'pcd'
lidar_dir = 'lidar_blue_pointcloud'
calib_dir = 'V2'
info_dir = 'info_files'
satellite_ori_size = 1280
query_size = [432, 816]
query_ori_size = [860, 1656]
# query_grd_height=0.335 #1.6-1.265

ToTensor = transforms.Compose([
    transforms.ToTensor()])

grd_trans = transforms.Compose([
    transforms.Resize(query_size),
    transforms.ToTensor()])

def homography_trans(image, I_tar, I_src, E, N, height):
    # inputs:
    #   image: src image
    #   I_tar,I_src:camera
    #   E: pose
    #   N: ground normal
    #   height: ground height
    # return:
    #   out: tar image

    w, h = I_tar.size

    # get back warp matrix
    i = torch.arange(0, h)
    j = torch.arange(0, w)
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w
    uv = torch.stack([jj, ii], dim=-1).float()  # shape = [h, w, 2]
    # ones = torch.ones_like(ii)
    # uv1 = torch.stack([jj, ii, ones], dim=-1).float()  # shape = [h, w, 3]

    p3D = I_tar.image2world(uv) # 2D->3D scale unknow

    depth = height / torch.einsum('...hwi,...i->...hw', p3D, N)
    depth = depth.clamp(0., 1000.)
    p3D_grd = depth[:,:,None] * p3D
    # each camera coordinate to 'query' coordinate
    p3d_ref = E * p3D_grd  # to sat
    uv,_ = I_src.world2image(p3d_ref)


    # lefttop to center
    uv_center = uv - I_src.size//2 #I_src.c  # shape = [h,w,2]
    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = I_src.size//2 #torch.max(I_src.size - I_src.c, I_src.c)
    uv_center /= scale

    out = torch.nn.functional.grid_sample(image.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
                        padding_mode='zeros')

    out = transforms.functional.to_pil_image(out.squeeze(0), mode='RGB')
    return out

def read_calib_yaml(calib_folder, file_name):
    with open(os.path.join(calib_folder, file_name), 'r') as stream:
        try:
            cur_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cur_yaml

def read_txt(root_folder, file_name):
    with open(os.path.join(root_folder, file_name)) as f:
        cur_file = f.readlines()
    return cur_file

def read_numpy(root_folder, file_name):
    try:
        with open(os.path.join(root_folder, file_name), 'rb') as f:
            cur_file = np.load(f, allow_pickle=True)
    except IOError:
        print("file open IO error, not exist?")
    return cur_file

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

class FordAV(BaseDataset):
    default_conf = {
        'dataset_dir': '/data/dataset/Ford_AV', #"/home/shan/data/FordAV", #'/data/FordAV', #
        'mul_query': 2
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        #assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(Dataset):
    def __init__(self, conf, split):
        self.root = conf.dataset_dir
        self.conf = conf
        if split == 'train':
            self.log_id = log_id_train
        # if split == 'val':
        #     self.log_id = log_id_val
        else:
            self.log_id = log_id_test

        # get calib infor
        calib_folder = os.path.join(self.root, calib_dir)

        # get intrinsic of front left camera and its RT to body
        FL_K_dict = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
        self.FL_k = np.asarray(FL_K_dict['K']).reshape(3, 3)
        # turn original img size to process img size
        self.FL_k[0] *= query_size[1] / query_ori_size[1]
        self.FL_k[1] *= query_size[0] / query_ori_size[0]

        FL2body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
        self.FL_relPose_body = quaternion_matrix(quat_from_pose(FL2body))
        FL_relTrans_body = trans_from_pose(FL2body)
        self.FL_relPose_body[0, 3] = FL_relTrans_body[0]
        self.FL_relPose_body[1, 3] = FL_relTrans_body[1]
        self.FL_relPose_body[2, 3] = FL_relTrans_body[2]

        # RR camera to body coordinate
        RR_K_dict = read_calib_yaml(calib_folder, "cameraRearRightIntrinsics.yaml")
        self.RR_k = np.asarray(RR_K_dict['K']).reshape(3, 3)
        # turn original img size to process img size
        self.RR_k[0] *= query_size[1] / query_ori_size[1]
        self.RR_k[1] *= query_size[0] / query_ori_size[0]
        RR2body = read_calib_yaml(calib_folder, "cameraRearRight_body.yaml")
        self.RR_relPose_body = quaternion_matrix(quat_from_pose(RR2body))
        RR_relTrans_body = trans_from_pose(RR2body)
        self.RR_relPose_body[0, 3] = RR_relTrans_body[0]
        self.RR_relPose_body[1, 3] = RR_relTrans_body[1]
        self.RR_relPose_body[2, 3] = RR_relTrans_body[2]

        # SL camera to body coordinate
        SL_K_dict = read_calib_yaml(calib_folder, "cameraSideLeftIntrinsics.yaml")
        self.SL_k = np.asarray(SL_K_dict['K']).reshape(3, 3)
        # turn original img size to process img size
        self.SL_k[0] *= query_size[1] / query_ori_size[1]
        self.SL_k[1] *= query_size[0] / query_ori_size[0]
        SL2body = read_calib_yaml(calib_folder, "cameraSideLeft_body.yaml")
        self.SL_relPose_body = quaternion_matrix(quat_from_pose(SL2body))
        SL_relTrans_body = trans_from_pose(SL2body)
        self.SL_relPose_body[0, 3] = SL_relTrans_body[0]
        self.SL_relPose_body[1, 3] = SL_relTrans_body[1]
        self.SL_relPose_body[2, 3] = SL_relTrans_body[2]

        # SR camera to body coordinate
        SR_K_dict = read_calib_yaml(calib_folder, "cameraSideRightIntrinsics.yaml")
        self.SR_k = np.asarray(SR_K_dict['K']).reshape(3, 3)
        # turn original img size to process img size
        self.SR_k[0] *= query_size[1] / query_ori_size[1]
        self.SR_k[1] *= query_size[0] / query_ori_size[0]
        SR2body = read_calib_yaml(calib_folder, "cameraSideRight_body.yaml")
        self.SR_relPose_body = quaternion_matrix(quat_from_pose(SR2body))
        SR_relTrans_body = trans_from_pose(SR2body)
        self.SR_relPose_body[0, 3] = SR_relTrans_body[0]
        self.SR_relPose_body[1, 3] = SR_relTrans_body[1]
        self.SR_relPose_body[2, 3] = SR_relTrans_body[2]

        log_folder = os.path.join(self.root, self.log_id )
        info_folder = os.path.join(log_folder, info_dir )

        # get original image & location information
        self.file_name = read_txt(info_folder, self.log_id  + '-FL-names.txt')
        self.file_name.pop(0)

        # get the satellite images
        satellite_folder = os.path.join(log_folder, sat_dir)
        satellite_names = glob(satellite_folder + '/*.png')
        nb_satellite_images = len(satellite_names)
        self.satellite_dict = {}
        for i in range(nb_satellite_images):
            cur_sat = int(os.path.split(satellite_names[i])[-1].split("_")[1])
            self.satellite_dict[cur_sat] = satellite_names[i]

        self.groundview_yaws = read_numpy(info_folder, 'groundview_yaws_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_rolls = read_numpy(info_folder,  'groundview_rolls_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_pitchs = read_numpy(info_folder, 'groundview_pitchs_pose_gt.npy')  # 'groundview_yaws_pose.npy'
        self.groundview_gps = read_numpy(info_folder, 'groundview_gps.npy')
        if not gt_from_gps:
            self.groundview_ned = read_numpy(info_folder, "groundview_NED_pose_gt.npy")
        self.match_pair = read_numpy(info_folder,
                                'groundview_satellite_pair.npy')  # 'groundview_satellite_pair_2.npy'# 'groundview_gps_2.npy'

        if 0:  # only 1 item
            self.file_name = self.file_name[:1]

        if 0:  # for debug
            # can not random sample, have order in npy files
            if split != 'train':
                self.file_name = self.file_name[:len(self.file_name)//3]
            # else:
            #     self.file_name = random.sample(self.file_name, len(self.file_name)//2)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        ###############################
        satellite_img = os.path.split(self.satellite_dict[self.match_pair[idx]])[-1].split("_")
        satellite_gps = [float(satellite_img[3]), float(satellite_img[5])]

        # get the current resolution of satellite image
        # a scale at 2 when downloading the dataset
        meter_per_pixel = 156543.03392 * np.cos(satellite_gps[0] * np.pi / 180.0) / np.power(2, sat_zoom) / 2.0

        query_gps = self.groundview_gps[idx, :]
        # using the satellite image as the reference and calculate the offset of the ground-view query
        dx, dy = gps_func.angular_distance_to_xy_distance_v2(satellite_gps[0], satellite_gps[1], query_gps[0],
                                                             query_gps[1])
        # get the pixel offsets of car pose
        dx_pixel = dx / meter_per_pixel # along the east direction
        dy_pixel = -dy / meter_per_pixel # along the north direction

        if not gt_from_gps:
            query_ned = self.groundview_ned[idx, :]
            grdx, grdy = gps_func.angular_distance_to_xy_distance(query_gps[0],
                                                                  query_gps[1])
            dx = query_ned[1]-grdx+dx # long east
            dy = query_ned[0]-grdy+dy # lat north
            # get the pixel offsets of car pose
            dx_pixel = dx / meter_per_pixel # along the east direction
            dy_pixel = -dy / meter_per_pixel # along the north direction

        heading = self.groundview_yaws[idx] * np.pi / 180.0
        roll = self.groundview_rolls[idx] * np.pi / 180.0
        pitch = self.groundview_pitchs[idx] * np.pi / 180.0

        # sat
        with Image.open(self.satellite_dict[self.match_pair[idx]], 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # ned: x: north, y: east, z:down
        ned2sat_r = np.array([[0,1,0],[-1,0,0],[0,0,1]]) #ned y->sat x; ned -x->sat y, ned z->sat z
        # to pose
        ned2sat = Pose.from_Rt(ned2sat_r, np.array([0.,0,0])).float() # shift in K
        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, dx_pixel+satellite_ori_size / 2.0, dy_pixel+satellite_ori_size / 2.0, 0,0,0,0,np.infty),#np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))

        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float()  # grd 2 sat in q2r, so just eye(4)
        }

        # grd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        log_folder = os.path.join(self.root, self.log_id)
        if self.conf['mul_query']>0:
            # ground images, rear right camera
            query_image_folder = os.path.join(log_folder, self.log_id + "-RR")
            name = os.path.join(query_image_folder, self.file_name[idx][:-1])
            with Image.open(name, 'r') as GrdImg:
                grd = GrdImg.convert('RGB')
                grd = grd_trans(grd)

            camera_para = (self.RR_k[0,0],self.RR_k[1,1],self.RR_k[0,2],self.RR_k[1,2])
            camera = Camera.from_colmap(dict(
                model='PINHOLE', params=camera_para,
                width=int(query_size[1]), height=int(query_size[0])))
            #FL2RR = inverse_pose(self.RR_relPose_body) @ self.FL_relPose_body
            # body2RR = inverse_pose(self.RR_relPose_body)
            RR_image = {
                # to array, when have multi query
                'image': grd.float(),
                'camera': camera.float(),
                'T_w2cam': Pose.from_4x4mat(self.RR_relPose_body).inv().float(), # body2camera
                'camera_h': torch.tensor(1.623)
            }

            if self.conf['mul_query'] > 1:
                # ground images, side left camera
                query_image_folder = os.path.join(log_folder, self.log_id + "-SL")
                name = os.path.join(query_image_folder, self.file_name[idx][:-1])
                with Image.open(name, 'r') as GrdImg:
                    grd = GrdImg.convert('RGB')
                    grd = grd_trans(grd)

                camera_para = (self.SL_k[0,0],self.SL_k[1,1],self.SL_k[0,2],self.SL_k[1,2])
                camera = Camera.from_colmap(dict(
                    model='PINHOLE', params=camera_para,
                    width=int(query_size[1]), height=int(query_size[0])))
                #FL2SL = inverse_pose(self.SL_relPose_body) @ self.FL_relPose_body
                # body2SL = inverse_pose(self.SL_relPose_body)
                SL_image = {
                    # to array, when have multi query
                    'image': grd.float(),
                    'camera': camera.float(),
                    'T_w2cam': Pose.from_4x4mat(self.SL_relPose_body).inv().float(),
                    'camera_h': torch.tensor(1.545)
                }

                # ground images, side right camera
                query_image_folder = os.path.join(log_folder, self.log_id + "-SR")
                name = os.path.join(query_image_folder, self.file_name[idx][:-1])
                with Image.open(name, 'r') as GrdImg:
                    grd = GrdImg.convert('RGB')
                    grd = grd_trans(grd)

                camera_para = (self.SR_k[0,0],self.SR_k[1,1],self.SR_k[0,2],self.SR_k[1,2])
                camera = Camera.from_colmap(dict(
                    model='PINHOLE', params=camera_para,
                    width=int(query_size[1]), height=int(query_size[0])))
                #FL2SR = inverse_pose(self.SR_relPose_body) @ self.FL_relPose_body
                # body2SR = inverse_pose(self.SR_relPose_body)
                SR_image = {
                    # to array, when have multi query
                    'image': grd.float(),
                    'camera': camera.float(),
                    'T_w2cam': Pose.from_4x4mat(self.SR_relPose_body).inv().float(),
                    'camera_h': torch.tensor(1.527)
                }

        # ground images, front left color camera
        query_image_folder = os.path.join(log_folder, self.log_id + "-FL")
        name = os.path.join(query_image_folder, self.file_name[idx][:-1])
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = grd_trans(grd)

        camera_para = (self.FL_k[0,0],self.FL_k[1,1],self.FL_k[0,2],self.FL_k[1,2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_size[1]), height=int(query_size[0])))
        # body2FL = inverse_pose(self.FL_relPose_body)
        FL_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(self.FL_relPose_body).inv().float(),
            'camera_h': torch.tensor(1.60)
        }

        normal = torch.tensor([[0., 0, 1]])  # down, z axis of body coordinate
        # # calculate road Normal for key point from camera 2D to 3D, in query coordinate
        # normal = torch.tensor([0.,0, 1]) # down, z axis of body coordinate
        # # ignore roll angle
        # ignore_roll = Pose.from_4x4mat(euler_matrix(-roll, 0, 0)).float()
        # normal = ignore_roll * normal

        # gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # query is body, ref is NED
        body2ned = Pose.from_4x4mat(euler_matrix(roll, pitch, heading)).float()
        body2sat = ned2sat@body2ned

        if not pre_init:
            # init and gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ramdom shift translation and rotation on yaw
            YawShiftRange = 15 * np.pi / 180 #error degree
            yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
            # R_yaw = torch.tensor([[np.cos(yaw),-np.sin(yaw),0],  [np.sin(yaw),np.cos(yaw),0], [0, 0, 1]])
            TShiftRange = 5
            T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
            T[2] = 0  # no shift on height
            #print(f'in dataset: yaw:{yaw/np.pi*180},t:{T}')

            # add random yaw and t to init pose
            R_yaw = euler_matrix(0, 0, yaw)
            init_shift = Pose.from_Rt(R_yaw[:3,:3], T).float()
            body2sat_init = init_shift@body2sat
        else:
            # use previous pose as initial pose
            pre_gps = self.groundview_gps[idx-1, :]
            # shift of previous gps
            dx, dy = gps_func.angular_distance_to_xy_distance_v2(query_gps[0],
                                                                 query_gps[1], pre_gps[0], pre_gps[1])
            if dx > 15 or dy > 15:
                # not coutinue frames
                body2sat_init = body2sat
            else:
                heading_pre = self.groundview_yaws[idx-1] * np.pi / 180.0
                roll_pre = self.groundview_rolls[idx-1] * np.pi / 180.0
                pitch_pre = self.groundview_pitchs[idx-1] * np.pi / 180.0
                body2ned_pre = Pose.from_4x4mat(euler_matrix(roll_pre, pitch_pre, heading_pre)).float()
                # add ne shift
                # get the pixel offsets of car pose
                de_pixel = dx / meter_per_pixel # along the east direction
                dn_pixel = dy / meter_per_pixel # along the north direction
                ned_shift = Pose.from_Rt(np.eye(3),np.array([dn_pixel,de_pixel,0])).float()
                body2sat_init = ned2sat@ned_shift@body2ned_pre

        data = {
            'ref': sat_image,
            'query': FL_image,
            'T_q2r_init': body2sat_init,
            'T_q2r_gt': body2sat,
            'normal': normal,
            #'grd_ratio': torch.tensor(0.65)
        }
        if self.conf['mul_query'] > 0:
            data['query_1'] = RR_image
        if self.conf['mul_query'] > 1:
            data['query_2'] = SL_image
            data['query_3'] = SR_image


        if 0:
            #show sat imge
            color_image0 = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
            color_image0 = np.array(color_image0)
            plt.imshow(color_image0)
            plt.show()

            # show grd image
            color_image1 = transforms.functional.to_pil_image(data['query']['image'], mode='RGB')
            color_image1 = np.array(color_image1)
            plt.imshow(color_image1)
            plt.show()
            color_image2 = transforms.functional.to_pil_image(data['query_1']['image'], mode='RGB')
            color_image2 = np.array(color_image2)
            plt.imshow(color_image2)
            plt.show()
            color_image3 = transforms.functional.to_pil_image(data['query_2']['image'], mode='RGB')
            color_image3 = np.array(color_image3)
            plt.imshow(color_image3)
            plt.show()
            color_image4 = transforms.functional.to_pil_image(data['query_3']['image'], mode='RGB')
            color_image4 = np.array(color_image4)
            plt.imshow(color_image4)
            plt.show()

        # debug
        if 0:
            color_image = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
            color_image = np.array(color_image)
            plt.imshow(color_image)

            # camera position
            # camera gt position
            origin = torch.zeros(3)
            origin_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * origin)
            origin_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * origin)
            # direct = torch.tensor([6.,0,0])
            direct = torch.tensor([0, 0, 0.])
            direct = data['query_2']['T_w2cam'].inv()*direct
            direct_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * direct)
            direct_2d_init, _ = data['ref']['camera'].world2image(data['T_q2r_init'] * direct)
            origin_2d_gt = origin_2d_gt.squeeze(0)
            origin_2d_init = origin_2d_init.squeeze(0)
            direct_2d_gt = direct_2d_gt.squeeze(0)
            direct_2d_init = direct_2d_init.squeeze(0)

            # plot the init direction of the body frame
            plt.scatter(x=origin_2d_init[0], y=origin_2d_init[1], c='r', s=5)
            plt.quiver(origin_2d_init[0], origin_2d_init[1], direct_2d_init[0] - origin_2d_init[0],
                       origin_2d_init[1] - direct_2d_init[1], color=['r'], scale=None)
            # plot the gt direction of the body frame
            plt.scatter(x=origin_2d_gt[0], y=origin_2d_gt[1], c='g', s=5)
            plt.quiver(origin_2d_gt[0], origin_2d_gt[1], direct_2d_gt[0] - origin_2d_gt[0],
                       origin_2d_gt[1] - direct_2d_gt[1], color=['g'], scale=None)

            plt.show()

            # grd images
            fig = plt.figure(figsize=plt.figaspect(0.5))

            if self.conf['mul_query'] == 2:
                ax1 = fig.add_subplot(2, 2, 1)
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
                ax4 = fig.add_subplot(2, 2, 4)
            elif self.conf['mul_query'] == 1:
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
            else:
                ax1 = fig.add_subplot(1, 1, 1)

            color_image1 = transforms.functional.to_pil_image(data['query']['image'], mode='RGB')
            color_image1 = np.array(color_image1)
            ax1.imshow(color_image1)

            if self.conf['mul_query'] > 0:
                color_image2 = transforms.functional.to_pil_image(data['query_1']['image'], mode='RGB')
                color_image2 = np.array(color_image2)
                ax2.imshow(color_image2)

            if self.conf['mul_query'] > 1:
                color_image3 = transforms.functional.to_pil_image(data['query_2']['image'], mode='RGB')
                color_image3 = np.array(color_image3)
                ax3.imshow(color_image3)

                color_image4 = transforms.functional.to_pil_image(data['query_3']['image'], mode='RGB')
                color_image4 = np.array(color_image4)
                ax4.imshow(color_image4)

            plt.show()
            print(self.file_name[idx][:-1])

        # debug projection
        if 0:#idx % 50 == 0:
            if self.conf['mul_query'] > 1:
                query_list = ['query','query_1','query_2','query_3']
            elif self.conf['mul_query'] > 0:
                query_list = ['query', 'query_1']
            else:
                query_list = ['query']
            # query_list = ['query']
            # project ground to sat
            for q in query_list:
                E = data['T_q2r_gt']@data[q]['T_w2cam'].inv()
                N = torch.einsum('...ij,...cj->...ci', data[q]['T_w2cam'].R, data['normal'])
                tran_sat = homography_trans(data['ref']['image'], data[q]['camera'], data['ref']['camera'], E, N.squeeze(0), data[q]['camera_h'])
                fig = plt.figure(figsize=plt.figaspect(1.))
                ax1 = fig.add_subplot(2, 2, 1)
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
                ax4 = fig.add_subplot(2, 2, 4)
                ax1.imshow(tran_sat)
                q_img = transforms.functional.to_pil_image(data[q]['image'], mode='RGB')
                ax2.imshow(q_img)
                fusion = Image.blend(q_img.convert("RGBA"), tran_sat.convert("RGBA"), alpha=.6)
                ax4.imshow(fusion)
                sat_img = transforms.functional.to_pil_image(data['ref']['image'], mode='RGB')
                ax3.imshow(sat_img)
                # camera gt position
                origin = torch.zeros(3)
                origin_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * origin)
                direct = torch.tensor([0, 0, 20.])
                direct = data[q]['T_w2cam'].inv() * direct
                direct_2d_gt, _ = data['ref']['camera'].world2image(data['T_q2r_gt'] * direct)
                origin_2d_gt = origin_2d_gt.squeeze(0)
                direct_2d_gt = direct_2d_gt.squeeze(0)
                # plot the gt direction of the body frame
                ax3.scatter(x=origin_2d_gt[0], y=origin_2d_gt[1], c='r', s=5)
                ax3.quiver(origin_2d_gt[0], origin_2d_gt[1], direct_2d_gt[0] - origin_2d_gt[0],
                           origin_2d_gt[1] - direct_2d_gt[1], color=['r'], scale=None)

                plt.show()
                print(name, q)

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'dataset_dir': '/data/dataset/Ford_AV',  # root_dir = "/home/shan/data/FordAV"
        'batch_size': 1,
        'num_workers': 0,
        'mul_query': 2 # 0: FL; 1:FL+RR; 2:FL+RR+SL+SR
    }
    dataset = FordAV(conf)
    loader = dataset.get_data_loader('test', shuffle=False)  # or 'train' ‘val’

    for i, data in zip(range(1000), loader):
        print(i)


