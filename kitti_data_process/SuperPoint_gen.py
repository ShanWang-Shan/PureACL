# check the matching relationship between cross-view images

import cv2 as cv
from ford_data_process.superpoint import SuperPoint
import torch
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

root_dir = "/home/shan/Dataset/Kitti"
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'

grd_pad_size = (384, 1248)
start_ratio = 0.45

ToTensor = transforms.Compose([
    transforms.ToTensor()])

# init super point------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 256
}

superpoint = SuperPoint(config).eval().to(device)

# read form txt files
for split in ('train', 'val', 'test'):
    grd_dict = {}
    txt_file_name = os.path.join(root_dir, grdimage_dir, 'kitti_split', split + '_files.txt')
    with open(txt_file_name, "r") as txt_f:
        lines = txt_f.readlines()
        for line in lines:
            line = line.strip()
            # check grb file exist
            grb_file_name = os.path.join(root_dir, grdimage_dir, line[:38], left_color_camera_dir,
                                         line[38:].lower())
            if not os.path.exists(grb_file_name):
                # ignore frames with out velodyne
                print(grb_file_name + ' do not exist!!!')
                continue

            file_name = line
            day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]

            # ground images, left color camera
            left_img_name = os.path.join(root_dir, grdimage_dir, drive_dir, left_color_camera_dir, image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img = GrdImg.convert('L')
                grd_ori_H = grd_img.size[1]
                grd_ori_W = grd_img.size[0]

                # crop
                H_start = int(grd_ori_H*start_ratio)
                H_end = grd_ori_H

                # trun np to tensor
                img = ToTensor(grd_img)[:,H_start:H_end]
                img = img.unsqueeze(0).to(device) # add b

                pred = superpoint({'image': img})
                key_points = pred['keypoints'][0].detach().cpu().numpy() #[n,2]
                key_points[:, 1] += H_start
                if 0:  # debug:
                    grd_img = cv.imread(left_img_name)
                    for j in range(key_points.shape[0]):
                        cv.circle(grd_img, (np.int32(key_points[j,0]), np.int32(key_points[j,1])), 2, (255, 0, 0),
                                   -1)
                    plt.imshow(grd_img), plt.show()
                # save kp
                grd_dict[file_name] = key_points
        with open(os.path.join(root_dir, grdimage_dir,split+'_kp.npy'), 'wb') as f:
            np.save(f, grd_dict)
        print(split+'_kp.npy saved')



