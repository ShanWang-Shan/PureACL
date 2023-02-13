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
import robotcar_data_process.robotcar_gps_coord_func as gps_func
from pixloc.pixlib.datasets.transformations import euler_matrix
from pixloc.pixlib.geometry import Camera, Pose

from robotcar_data_process.camera_model import CameraModel
from robotcar_data_process.transform import build_se3_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True # add for 'broken data stream'

sat_dir = 'Satellite_Images'
sat_zoom = 18
satellite_ori_size = 1280

###############################
query_front_size = [400, 528] #[480, 640]
query_front_ori_size = [960, 1280]
query_size = [1024, 1024]
query_mono_ori_size = [1024, 1024]
# query height : 0.45m

ToTensor = transforms.Compose([
    transforms.ToTensor()])

grd_trans = transforms.Compose([
    transforms.Resize(query_size),
    transforms.ToTensor()])

grd_front_trans = transforms.Compose([
    transforms.Resize(query_front_size),
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

def camera_in_ex(root, camera_model):
    # get camera model
    model_dir = os.path.join(root, "camera-models")
    extrinsics_dir = os.path.join(root, "extrinsics")
    camera = CameraModel(model_dir, camera_model)
    extrinsics_path = os.path.join(extrinsics_dir, camera.camera + '.txt')
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]
    G_camera_vehicle = build_se3_transform(extrinsics) # camera 0 to caemra n

    # rtk to camera 0
    with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_rtk = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])

    # camera face x-> face z
    # camera_Rzx = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
    camera_Rzx = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    # scale of process and original images
    if 'mono' in camera_model:
        scale_x =  query_size[1]/query_mono_ori_size[1]
        scale_y = query_size[0] / query_mono_ori_size[0]
    else:
        scale_x =  query_front_size[1]/query_front_ori_size[1]
        scale_y = query_front_size[0] / query_front_ori_size[0]
    camera_K = np.eye(3)
    camera_K[0, 0] = camera.focal_length[0] * scale_x
    camera_K[0, 2] = camera.principal_point[0] * scale_x
    camera_K[1, 1] = camera.focal_length[1] * scale_y
    camera_K[1, 2] = camera.principal_point[1] * scale_y

    return camera_K, camera_Rzx@np.array(G_camera_rtk)

class RobotCar(BaseDataset):
    default_conf = {
        'dataset_dir': "/home/shan/data/robotcar/", #'/data/dataset/robotcar/',
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
        self.files = np.genfromtxt(os.path.join(self.root, "split", split+'.csv'),
                                   dtype=[('day','U19'),('front_ts','i8'),('left_ts','i8'),('right_ts','i8'),
                                          ('rear_ts','i8'),('sat','U94'),('lat','f8'),('long','f8'),('alt','f8'),
                                          ('roll','f8'),('pitch','f8'),('yaw','f8')], delimiter=',')
        # get camera intrinsics and extrinsics
        self.front_camera, self.frontC_vehicle = camera_in_ex(self.root, 'stereo/centre')
        self.left_camera, self.leftC_vehicle = camera_in_ex(self.root, 'mono_left')
        self.right_camera, self.rightC_vehicle = camera_in_ex(self.root, 'mono_right')
        self.rear_camera, self.rearC_vehicle = camera_in_ex(self.root, 'mono_rear')

        if 0: #debug find missing
            for i in range(self.files.shape[0]):
                log_folder = os.path.join(self.root, 'undistort', self.files['day'][i])
                # rear camera
                name = os.path.join(log_folder, "mono_rear", str(self.files['rear_ts'][i]) + '.png')
                if not os.path.exists(name):
                    print('no file: ', name)
                # left camera
                name = os.path.join(log_folder, "mono_left", str(self.files['left_ts'][i]) + '.png')
                if not os.path.exists(name):
                    print('no file: ', name)
                # right camera
                name = os.path.join(log_folder, "mono_right", str(self.files['right_ts'][i]) + '.png')
                if not os.path.exists(name):
                    print('no file: ', name)
                # front camera
                name = os.path.join(log_folder, "stereo/centre", str(self.files['front_ts'][i]) + '.png')
                if not os.path.exists(name):
                    print('no file: ', name)

        if 0:  # only 1 item
            self.files = self.files[:1]

        if 0:  # for debug
            self.files = self.files[:len(self.files)//3]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ###############################
        satellite_img = os.path.split(self.files['sat'][idx])[-1].split("_")
        satellite_gps = [float(satellite_img[3]), float(satellite_img[5])]

        # get the current resolution of satellite image
        # a scale at 2 when downloading the dataset
        meter_per_pixel = 156543.03392 * np.cos(satellite_gps[0] * np.pi / 180.0) / np.power(2, sat_zoom) / 2.0

        # using the satellite image as the reference and calculate the offset of the ground-view query
        dx, dy = gps_func.angular_distance_to_xy_distance_v2(satellite_gps[0], satellite_gps[1], self.files['lat'][idx],
                                                             self.files['long'][idx])
        # get the pixel offsets of car pose
        dx_pixel = dx / meter_per_pixel # along the east direction
        dy_pixel = -dy / meter_per_pixel # along the north direction

        # sat
        with Image.open(os.path.join(self.root,sat_dir,self.files['sat'][idx]), 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            sat_map = ToTensor(sat_map)

        # # ned: x: north, y: east, z:down
        # ned2sat_r = np.array([[0,1,0],[-1,0,0],[0,0,1]]) #ned y->sat x; ned -x->sat y, ned z->sat z
        # # to pose
        # ned2sat = Pose.from_Rt(ned2sat_r, np.array([0.,0,0])).float() # shift in K
        camera = Camera.from_colmap(dict(
            model='SIMPLE_PINHOLE', params=(1 / meter_per_pixel, dx_pixel+satellite_ori_size / 2.0, dy_pixel+satellite_ori_size / 2.0, 0,0,0,0,np.infty),#np.infty for parallel projection
            width=int(satellite_ori_size), height=int(satellite_ori_size)))

        sat_image = {
            'image': sat_map.float(),
            'camera': camera.float(),
            'T_w2cam': Pose.from_4x4mat(np.eye(4)).float(),  # grd 2 sat in q2r, so just eye(4)
        }

        # grd ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        log_folder = os.path.join(self.root, 'undistort', self.files['day'][idx])
        if self.conf['mul_query']>0:
            # ground images, rear camera
            query_image_folder = os.path.join(log_folder, "mono_rear")
            name = os.path.join(query_image_folder, str(self.files['rear_ts'][idx])+'.png')
            if not os.path.exists(name):
                print("no file ", name)
            with Image.open(name, 'r') as GrdImg:
                grd = GrdImg.convert('RGB')
                grd = grd_trans(grd)


            camera_para = (self.rear_camera[0,0],self.rear_camera[1,1],self.rear_camera[0,2],self.rear_camera[1,2])
            camera = Camera.from_colmap(dict(
                model='PINHOLE', params=camera_para,
                width=int(query_size[1]), height=int(query_size[0])))
            body2rear = Pose.from_4x4mat(self.rearC_vehicle)
            R_image = {
                # to array, when have multi query
                'image': grd.float(),
                'camera': camera.float(),
                'T_w2cam': body2rear.float(), # body2camera
                'camera_h': torch.tensor(1.44)
            }

            if self.conf['mul_query'] > 1:
                # ground images, side left camera
                query_image_folder = os.path.join(log_folder, "mono_left")
                name = os.path.join(query_image_folder, str(self.files['left_ts'][idx]) + '.png')
                if not os.path.exists(name):
                    print("no file ", name)
                with Image.open(name, 'r') as GrdImg:
                    grd = GrdImg.convert('RGB')
                    grd = grd_trans(grd)

                camera_para = (self.left_camera[0, 0], self.left_camera[1, 1], self.left_camera[0, 2], self.left_camera[1, 2])
                camera = Camera.from_colmap(dict(
                    model='PINHOLE', params=camera_para,
                    width=int(query_size[1]), height=int(query_size[0])))
                body2left = Pose.from_4x4mat(self.leftC_vehicle)
                SL_image = {
                    # to array, when have multi query
                    'image': grd.float(),
                    'camera': camera.float(),
                    'T_w2cam': body2left.float(),  # body2camera
                    'camera_h': torch.tensor(1.36)
                }

                # ground images, side right camera
                query_image_folder = os.path.join(log_folder, "mono_right")
                name = os.path.join(query_image_folder, str(self.files['right_ts'][idx]) + '.png')
                if not os.path.exists(name):
                    print("no file ", name)
                with Image.open(name, 'r') as GrdImg:
                    grd = GrdImg.convert('RGB')
                    grd = grd_trans(grd)

                camera_para = (self.right_camera[0, 0], self.right_camera[1, 1], self.right_camera[0, 2], self.right_camera[1, 2])
                camera = Camera.from_colmap(dict(
                    model='PINHOLE', params=camera_para,
                    width=int(query_size[1]), height=int(query_size[0])))
                body2right = Pose.from_4x4mat(self.rightC_vehicle)
                SR_image = {
                    # to array, when have multi query
                    'image': grd.float(),
                    'camera': camera.float(),
                    'T_w2cam': body2right.float(),  # body2camera
                    'camera_h': torch.tensor(1.36)
                }

        # ground images, front color camera
        query_image_folder = os.path.join(log_folder, "stereo", "centre")
        name = os.path.join(query_image_folder, str(self.files['front_ts'][idx]) + '.png')
        if not os.path.exists(name):
            print("no file ", name)
        with Image.open(name, 'r') as GrdImg:
            grd = GrdImg.convert('RGB')
            grd = grd_front_trans(grd)

        camera_para = (self.front_camera[0, 0], self.front_camera[1, 1], self.front_camera[0, 2], self.front_camera[1, 2])
        camera = Camera.from_colmap(dict(
            model='PINHOLE', params=camera_para,
            width=int(query_front_size[1]), height=int(query_front_size[0])))
        body2front = Pose.from_4x4mat(self.frontC_vehicle)
        F_image = {
            # to array, when have multi query
            'image': grd.float(),
            'camera': camera.float(),
            'T_w2cam': body2front.float(),  # body2camera
            'camera_h': torch.tensor(1.52)
        }

        normal = torch.tensor([[0., 0., 1]])  # down, z axis of body coordinate
        # # query is body, ref is NED
        # # calculate road Normal for key point from camera 2D to 3D, in query coordinate
        # normal = torch.tensor([0.,0.,1]) # down, z axis of body coordinate
        # # ignore roll angle
        # ignore_roll = Pose.from_4x4mat(euler_matrix(-self.files['roll'][idx], 0, 0)).float()
        # normal = ignore_roll * normal

        # gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        body2wnd = Pose.from_4x4mat(euler_matrix(self.files['roll'][idx], self.files['pitch'][idx], self.files['yaw'][idx])).float()
        wnd2sat = Pose.from_4x4mat(np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])).float()
        body2sat = wnd2sat@body2wnd
        # body2sat = ned2sat@body2ned

        # normal = torch.tensor([0.,0.,1]) # down, z axis of sat coordinate
        # normal = body2sat.inv()*normal

        # init and gt pose~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ramdom shift translation and rotation on yaw
        YawShiftRange = 15 * np.pi / 180 #error degree
        yaw = 2 * YawShiftRange * np.random.random() - YawShiftRange
        TShiftRange = 5 
        T = 2 * TShiftRange * np.random.rand((3)) - TShiftRange
        T[2] = 0  # no shift on height
        #print(f'in dataset: yaw:{yaw/np.pi*180},t:{T}')

        # add random yaw and t to init pose
        R_yaw = euler_matrix(0, 0, yaw)
        init_shift = Pose.from_Rt(R_yaw[:3,:3], T).float()
        body2sat_init = init_shift@body2sat

        data = {
            'ref': sat_image,
            'query': F_image,
            'T_q2r_init': body2sat_init,
            'T_q2r_gt': body2sat,
            'normal': normal,
            #'grd_ratio': torch.tensor(0.45)
        }
        if self.conf['mul_query'] > 0:
            data['query_1'] = R_image
        if self.conf['mul_query'] > 1:
            data['query_2'] = SL_image
            data['query_3'] = SR_image

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
            # direct = torch.tensor([6.,0, 0])
            direct = torch.tensor([0,0, 6.])
            direct = data['query']['T_w2cam'].inv() * direct
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
            print(self.files['front_ts'][idx])

        # debug projection
        if 0:#idx % 30 == 0:
            if self.conf['mul_query'] > 1:
                query_list = ['query','query_1','query_2','query_3']
            elif self.conf['mul_query'] > 0:
                query_list = ['query', 'query_1']
            else:
                query_list = ['query']
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
                print(self.files['lat'][idx], self.files['long'][idx],q)

        return data

if __name__ == '__main__':
    # test to load 1 data
    conf = {
        'dataset_dir': "/data/dataset/robotcar/",
        'batch_size': 1,
        'num_workers': 0,
        'mul_query': 2 # 0: FL; 1:FL+RR; 2:FL+RR+SL+SR
    }
    dataset = RobotCar(conf)
    loader = dataset.get_data_loader('train', shuffle=False)  # or 'train' ‘val’ 'test'

    for i, data in zip(range(8000), loader):
        print(i)

    # for i, data in zip(range(1000), loader):
    #     print(i)


