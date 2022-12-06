# check the matching relationship between cross-view images

from input_libs import *
import cv2 as cv
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy, read_csv, write_numpy

root_folder = "../"

log_id = "2017-10-26-V2-Log1"#"2017-08-04-V2-Log1"#
# size of the satellite image and ground-view query image (left camera)
satellite_size = 1280
query_size = [1656, 860]


log_folder = os.path.join(root_folder , log_id, 'info_files')
FL_image_names = read_txt(log_folder, log_id + '-FL-names.txt')
FL_image_names.pop(0)
nb_query_images = len(FL_image_names)

# get the satellite images
satellite_folder = os.path.join( root_folder, log_id, "Satellite_Images")
satellite_names = glob.glob(satellite_folder + '/*.png')

nb_satellite_images = len(satellite_names)

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

# 3. read the matching pair
match_pair = read_numpy(log_folder , 'groundview_satellite_pair.npy') # 'groundview_satellite_pair_2.npy'

for grd_folder in ('-SR','-SL'):#('-FL','-RR','-SL','-SR'):
    query_image_folder = os.path.join(root_folder, log_id, log_id + grd_folder)

    if grd_folder == '-FL':
        H_start = query_size[1]*62//100
        H_end = query_size[1]*95//100
    elif grd_folder == '-RR':
        H_start = query_size[1]*62//100
        H_end = query_size[1]*88//100
    else:
        H_start = query_size[1]*62//100
        H_end = query_size[1]

    grd_dict = {}
    for i in range(nb_query_images):
        grd_img = cv.imread(os.path.join(query_image_folder, FL_image_names[i][:-1]))
        if grd_img is None:
            print(os.path.join(query_image_folder, FL_image_names[i][:-1]))
        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=512)
        # find the keypoints with ORB
        # turn RGB to HSV--------------
        detect_img = cv.cvtColor(grd_img[H_start:H_end], cv.COLOR_BGR2HSV)
        # v to range 50~150
        # detect_img[:,:,-1] = np.where(detect_img[:,:,-1]>150, 150, detect_img[:,:,-1])
        # detect_img[:, :, -1] = np.where(detect_img[:, :, -1] < 100, 100, detect_img[:, :, -1])
        detect_img[:, :, -1] = np.clip(detect_img[:, :, -1], 50, 170)
        detect_img = cv.cvtColor(detect_img, cv.COLOR_HSV2BGR)
        #-------------------------------------

        #detect_img = grd_img[query_size[1]*64//100:]
        kp = orb.detect(detect_img, None)
        #assert len(kp) >0, f'no orb points detected on {FL_image_names[i][:-1]}'
        if len(kp) <=0:
            print(f'no orb points detected on {FL_image_names[i][:-1]}')
            continue
        if 0:  # debug:
            # draw only keypoints location,not size and orientation
            img2 = cv.drawKeypoints(detect_img, kp, None, color=(0, 255, 0), flags=0)
            plt.imshow(img2), plt.show()
        # only save kp
        kp_list = []
        for p in range(len(kp)):
            ori_pt = [kp[p].pt[0], kp[p].pt[1]+H_start]
            kp_list.append(ori_pt)

        grd_dict[FL_image_names[i][:-1]] = kp_list
    write_numpy(log_folder, grd_folder[1:]+'_kp.npy', grd_dict)
    print(grd_folder[1:]+'_kp.npy saved')



