#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:37:35 2021

@author: shan
"""
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
from scipy import interpolate
import cv2
import collections
import matplotlib as mpl
import matplotlib.cm as cm
from input_libs import *

depth_map_dir = '../LEAStereo/predict/sceneflow/images/Ford'



grd_img_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0~1 to -1~1
            ])

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def fillMissingValues(target_for_interp, invalide_pixel, copy=False,
                      interpolator=interpolate.LinearNDInterpolator):
    # tensor to np
    target_for_interp = np.array(target_for_interp)
    if copy:
        target_for_interp = target_for_interp.copy()

    def getPixelsForInterp(img):
        """
        Calculates a mask of pixels neighboring invalid values -
           to use for interpolation.
        """
        # mask invalid pixels
        invalid_mask = np.isnan(img) + (img == invalide_pixel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        #dilate to mark borders around invalid regions
        dilated_mask = cv2.dilate(invalid_mask.astype('uint8'), kernel,
                          borderType=cv2.BORDER_CONSTANT, borderValue=int(0))

        # pixelwise "and" with valid pixel mask (~invalid_mask)
        masked_for_interp = dilated_mask *  ~invalid_mask
        return masked_for_interp.astype('bool'), invalid_mask

    # Mask pixels for interpolation
    mask_for_interp, invalid_mask = getPixelsForInterp(target_for_interp)

    # Interpolate only holes, only using these pixels
    points = np.argwhere(mask_for_interp)
    values = target_for_interp[mask_for_interp]
    interp = interpolator(points, values)

    target_for_interp[invalid_mask] = interp(np.argwhere(invalid_mask))

    # np to tensor
    target_for_interp = torch.from_numpy(target_for_interp)
    return target_for_interp


def lin_interp(shape, xdy):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xdy[:, 0::2], xdy[:, 1]
    f = interpolate.LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def get_height(depth, satmap_edge, heading, camera_k, meter_per_pixel):

    #depth = torch.where(depth >= 80., torch.tensor(1000.), depth)
    # real word coordinate
    # meshgrid the depth pannel
    H,W = depth.shape
    i = torch.arange(0, H)
    j = torch.arange(0, W)  # W +- 15 degree
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w
    uv = torch.stack([jj, ii], dim=-1).float()

    cx = camera_k[0, 2]
    cy = camera_k[1, 2]
    fx = camera_k[0, 0]
    fy = camera_k[1, 1]

    uv_center = uv - torch.tensor([cx, cy])

    # X = (uv_center[:, :, 0]) * depth / fx
    # Y = (uv_center[:, :, 1]) * depth / fy
    # Z = depth
    x_over_z = (uv_center[:, :, 0]) / fx
    y_over_z = (uv_center[:, :, 1]) / fy
    Z = depth / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    X = x_over_z * Z
    Y = y_over_z * Z

    depth_XYZ = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], dim=-1)
    # camera only have height offset

    depth_XYZ -= torch.tensor([0,0,0])

    cos = torch.cos(heading)
    sin = torch.sin(heading)
    R_depth = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])  # shape = [9]
    depth_XYZ = torch.einsum('ij, hwj -> hwi', R_depth, depth_XYZ).float() # shape = [H,W,3]

    depth_XYZ = depth_XYZ[:, :, :].view(H*W,-1)
    depth_XYZ = np.array(depth_XYZ)
    depth_XYZ[:,0] = np.floor(depth_XYZ[:,0] / meter_per_pixel + satmap_edge / 2)
    depth_XYZ[:,2] = np.floor(depth_XYZ[:,2] / meter_per_pixel + satmap_edge / 2)
    val_inds = (depth_XYZ[:,0] >= 0) & (depth_XYZ[:,2] >= 0)
    val_inds = val_inds & (depth_XYZ[:,0] < satmap_edge) & (depth_XYZ[:,2] < satmap_edge)
    #val_inds = val_inds & (depth_XYZ[:, 1] < 0) # height only >0
    depth_XYZ = depth_XYZ[val_inds, :]


    height = np.ones([satmap_edge,satmap_edge])*80
    #height[depth_XYZ[:,0].astype(np.int), depth_XYZ[:,2].astype(np.int)] = torch.from_numpy(depth_XYZ[:,1])

    # find the duplicate points and choose the closest depth
    inds = sub2ind(height.shape, depth_XYZ[:, 0], depth_XYZ[:, 2])
    dupe_inds = [item for item, count in collections.Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(depth_XYZ[pts[0], 0])
        z_loc = int(depth_XYZ[pts[0], 2])
        height[x_loc, z_loc] = depth_XYZ[pts, 1].max()
    height = torch.from_numpy(height).float()

    # turn 0 height(no data) out of camera display range (under ground 100m to remove it)
    #vmin = torch.min(height)
    height = torch.where(height < -7., torch.tensor(80.), height)

    #interpolate the depth map to fill in holes
    #height = fillMissingValues(height, -20)

    #debug
    normalizer = mpl.colors.Normalize(vmin=torch.min(height), vmax=0)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    colormapped_im = (mapper.to_rgba(height)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save("height_map.png")

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(cm.ScalarMappable(norm=normalizer, cmap='magma_r'),
                 cax=ax, orientation='horizontal', label='Some Units')
    fig.show()

    # height_img = transforms.functional.to_pil_image(height, mode='L')
    # height_img.save("height_map.png")

    return height


def get_warp_sat2real(satmap_edge, depth, heading, camera_k, lat):
    # satellite: u:east , v:south from topleft and u_center: east; v_center: north from center
    # realword: X: south, Y:down, Z: east
    
    # meshgrid the sat pannel
    i = j = torch.arange(0, satmap_edge)
    ii,jj = torch.meshgrid(i, j) # i:h,j:w
    
    # uv is coordinate from top/left, v: south, u:east
    uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_edge, satmap_edge, 2] 
     
    # sat map from top/left to center coordinate
    u0 = v0 = satmap_edge//2
    uv_center = uv-torch.tensor([u0,v0])

    # affine matrix: scale*R
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / np.power(2, 20) / 2.0
    R = torch.tensor([[0,1],[1,0]]).float()
    Aff_sat2real = meter_per_pixel*R # shape = [2,2]
    
    # Trans matrix from sat to realword
    XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real, uv_center) # shape = [satmap_edge, satmap_edge, 2]

    # # velodyne to get height
    # # velo to realword where original at camera ground position
    # camera_height = utils.get_camera_height()
    # cos = np.cos(heading)
    # sin = np.sin(heading)
    # P = np.array([[-sin,-cos,0,0.06],[0,0,-1,-camera_height],[cos,-sin,0,-0.27]])
    # velo_real = P@(velodyne.T) # [3,n]
    #
    # # Filter lidar points to be within realword FOV
    # # turn xz meter to pixel
    # velo_real[0,:] /= meter_per_pixel
    # velo_real[2,:] /= meter_per_pixel
    # velo_real[0,:] = np.round(velo_real[0, :]) + satmap_edge/2 - 1
    # velo_real[2,:] = np.round(velo_real[2, :]) + satmap_edge/2 - 1
    # val_inds = (velo_real[0,:] >= 0) & (velo_real[2,:] >= 0)
    # val_inds = val_inds & (velo_real[0,:] < satmap_edge) & (velo_real[2,:] < satmap_edge) #[0]
    # velo_real = velo_real[:, val_inds]
    #
    # # project to Y/height image
    # Y = np.zeros((satmap_edge,satmap_edge))
    # Y[velo_real[0, :].astype(np.int), velo_real[2, :].astype(np.int)] = velo_real[1, :]
    #
    # # find the duplicate points and choose the closest depth
    # inds = sub2ind(Y.shape, velo_real[0, :], velo_real[2, :])
    # dupe_inds = [item for item, count in collections.Counter(inds).items() if count > 1]
    # for dd in dupe_inds:
    #     pts = np.where(inds == dd)[0]
    #     x_loc = int(velo_real[0,pts[0]])
    #     z_loc = int(velo_real[2,pts[0]])
    #     Y[x_loc, z_loc] = velo_real[1, pts].min()
    #
    #
    # # turn 0 height(no data) out of camera display range (under ground 100m to remove it)
    #Y = np.where(Y<-8, 100., Y)
    #
    # # interpolate the depth map to fill in holes
    # #Y = lin_interp((satmap_edge,satmap_edge), velo_real.T)
    # #Y = fillMissingValues(Y, 0)
    #
    # #debug
    # Y = torch.from_numpy(Y).float()
    # Y_img = transforms.functional.to_pil_image(Y, mode='L')
    # Y_img.save("height_map.png")

    # Y = get_height(depth, satmap_edge, heading, camera_k, meter_per_pixel)
    # Y = Y.unsqueeze(-1)

    # X+Y+Z+1: shape=[satmap_edge, satmap_edge, 4] homogeneous coordinates of realword
    Y = torch.zeros((satmap_edge, satmap_edge, 1))
    ones = torch.ones((satmap_edge, satmap_edge, 1))
    return torch.cat([XZ[:,:,:1], Y, XZ[:,:,1:], ones], dim=-1) # [H,W,4]

def get_warp_ned2camera(XYZ_1, camera_R, camera_k, shift=None):
    # realword: X: north, Y:east, Z: down
    # camera: u:south, v: up from center (when heading east, need rotate heading angle)
    
    # cos = np.cos(-heading)
    # sin = np.sin(-heading)
    # R = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]) # shape = [9]
    
    
    # camera_height = 1.265415#utils_pro.get_camera_height()
    # camera only have height offset
    # T = np.array([[0],[camera_height], [0]]) # shape = [3,1]
    # T = np.einsum('ij, jk -> ik', camera_R, T) # shape = [3,1]
    T = np.zeros((3,1))
    # P = K[R|T]
    P = np.einsum('ij, jk -> ik', camera_k, np.hstack((camera_R, T))) # shape = [3,4]
    P = torch.from_numpy(P)

    # project ground pannel to ground camera coordinates
    uv_1 = torch.einsum('ij, hwj -> hwi', P.float(), XYZ_1.float()) # shape = [H, W,3]
    # only need view in front of camera ,Epsilon = 1e-6
    uv_1_last = torch.maximum(uv_1[:,:,2:], torch.ones_like(uv_1[:,:,2:]) * 1e-6)
    uv = uv_1[:,:,:2]/uv_1_last # shape = [H, W,2]

    return uv

def project_grd_to_map( grd, depth, heading, camera_k, satmap_edge, lat):
    # inputs:
    #   grd: ground image: C*H*W
    #   velodyne: 3D points: np.array[n,3]
    #   heading: heading angle
    #   camera_k: 3*3 K maheight_maptrix of left color camera : 3*3
    # return:
    #   grd_trans: C*satmap_edge*satmap_edge
    grd = grd_img_transform(grd)
    depth = grd_img_transform(depth)
    
    C,H,W = grd.size()
    
    # get back warp matrix
    XYZ_1 = get_warp_sat2real(satmap_edge, depth, heading, camera_k, lat) #  [edge,edge,4]
    uv = get_warp_ned2camera(XYZ_1, heading, camera_k) #  [edge, edge,2]
    
    # lefttop to center
    uv_center = uv-torch.tensor([W//2,H//2])# shape = [H, W,2] 

    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = torch.tensor([W//2, H//2])
    uv_center /= scale
        
    grd_trans = F.grid_sample(grd.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear', padding_mode='zeros') #[C,edge,edge]

    return grd_trans.squeeze(0)


def get_warp_sat2ned_grd(satmap_edge, lat, zoom):
    # satellite: u:east , v:south from topleft and u_center: east; v_center: north from center
    # realword: X: north, Y:east, Z: down

    # meshgrid the sat pannel
    i = j = torch.arange(0, satmap_edge)
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w

    # uv is coordinate from top/left, v: south, u:east
    uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_edge, satmap_edge, 2]

    # sat map from top/left to center coordinate
    u0 = v0 = satmap_edge // 2
    uv_center = uv - torch.tensor([u0, v0])

    # affine matrix: scale*R
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / np.power(2, zoom) / 2.0
    R = torch.tensor([[0, -1], [1, 0]]).float()
    Aff_sat2ned = meter_per_pixel * R  # shape = [2,2]

    # Trans matrix from sat to realword
    XY = torch.einsum('ij, hwj -> hwi', Aff_sat2ned, uv_center)  # shape = [satmap_edge, satmap_edge, 2]

    Z = torch.ones((satmap_edge, satmap_edge, 1))*1.55 # try 1.6
    ones = torch.ones((satmap_edge, satmap_edge, 1))
    return torch.cat([XY[:, :], Z, ones], dim=-1)  # [H,W,4]

def project_grd_to_map_grd( grd, camera_R, camera_k, satmap_edge, lat, zoom):
    grd = grd_img_transform(grd)
    C, H, W = grd.size()

    # get back warp matrix
    XYZ_1 = get_warp_sat2ned_grd(satmap_edge, lat, zoom)  # [edge,edge,4]
    uv = get_warp_ned2camera(XYZ_1, camera_R, camera_k)  # [edge, edge,2]

    # lefttop to center
    uv_center = uv - torch.tensor([W // 2, H // 2])  # shape = [H, W,2]

    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = torch.tensor([W // 2, H // 2])
    uv_center /= scale

    grd_trans = F.grid_sample(grd.unsqueeze(0), uv_center.unsqueeze(0), mode='bilinear',
                              padding_mode='zeros')  # [C,edge,edge]

    return grd_trans.squeeze(0)


def fuse(image1, image2):
    # image 1 from np to pil
    img1 = transforms.functional.to_pil_image(image1, mode='RGB')
    img1 = img1.convert("RGBA")

    img2 = transforms.functional.to_pil_image(grd_img_transform(image2), mode = 'RGBA')

    out_img = Image.blend(img1, img2, alpha=.2)

    if 1:
        img1.save("project.png")
        out_img.save("merge.png")

    return np.array(out_img)



