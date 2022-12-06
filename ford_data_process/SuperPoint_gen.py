# check the matching relationship between cross-view images

from input_libs import *
import cv2 as cv
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy, read_csv, write_numpy
from superpoint import SuperPoint
import torch
from torchvision import transforms

root_folder = "/data/dataset/Ford_AV"

log_id = "2017-10-26-V2-Log5"#"2017-08-04-V2-Log5"#
# size of the satellite image and ground-view query image (left camera)
# satellite_size = 1280
query_size = [1656, 860]
start_ratio = 0.6


log_folder = os.path.join(root_folder , log_id, 'info_files')
FL_image_names = read_txt(log_folder, log_id + '-FL-names.txt')
FL_image_names.pop(0)
nb_query_images = len(FL_image_names)

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

#-----------------------------------------------------------

# # get the satellite images
# satellite_folder = os.path.join( root_folder, log_id, "Satellite_Images")
# satellite_names = glob.glob(satellite_folder + '/*.png')
# nb_satellite_images = len(satellite_names)

# satellite_dict = {}
# for i in range(nb_satellite_images):
#     sate_img = cv.imread(satellite_names[i])
#     # Initiate ORB detector
#     orb = cv.ORB_create(nfeatures=512*4)
#     # find the keypoints with ORB
#     kp = orb.detect(sate_img, None)
#
#     if 0: #debug:
#         # draw only keypoints location,not size and orientation
#         img2 = cv.drawKeypoints(sate_img, kp, None, color=(0, 255, 0), flags=0)
#         plt.imshow(img2), plt.show()
#     # only save kp
#     kp_list = []
#     for p in range(len(kp)):
#         kp_list.append(kp[p].pt)
#
#     sat_file_name = satellite_names[i].split('/')
#     satellite_dict[sat_file_name[-1]] = np.array(kp_list)
# write_numpy(log_folder, 'satellite_kp.npy', satellite_dict)
# print('satellite_kp.npy saved')

# # 3. read the matching pair
# match_pair = read_numpy(log_folder , 'groundview_satellite_pair.npy') # 'groundview_satellite_pair_2.npy'

for grd_folder in ('-FL','-RR','-SL','-SR'):
    query_image_folder = os.path.join(root_folder, log_id, log_id + grd_folder)

    # crop
    H_start = int(query_size[1]*start_ratio)
    H_end = query_size[1]

    grd_dict = {}
    for i in range(nb_query_images):
        grd_img = cv.imread(os.path.join(query_image_folder, FL_image_names[i][:-1]), cv.IMREAD_GRAYSCALE)
        if grd_img is None:
            print(os.path.join(query_image_folder, FL_image_names[i][:-1]))

        # trun np to tensor
        img = ToTensor(grd_img[H_start:H_end])
        img = img.unsqueeze(0).to(device) # add b

        pred = superpoint({'image': img})
        key_points = pred['keypoints'][0].detach().cpu().numpy() #[n,2]
        key_points[:, 1] += H_start
        if 0:  # debug:
            grd_img = cv.imread(os.path.join(query_image_folder, FL_image_names[i][:-1]))
            for j in range(key_points.shape[0]):
                cv.circle(grd_img, (np.int32(key_points[j,0]), np.int32(key_points[j,1])), 2, (255, 0, 0),
                           -1)
            plt.imshow(grd_img), plt.show()
        # save kp
        grd_dict[FL_image_names[i][:-1]] = key_points
    write_numpy(log_folder, grd_folder[1:]+'_kp.npy', grd_dict)
    print(grd_folder[1:]+'_kp.npy saved')



