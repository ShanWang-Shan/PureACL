import cv2
import os

# root_folder = "/home/users/u7094434/projects/SIDFM/visual_kitti/confidence_maps_1"
#
# image_dir = os.path.join(root_folder)
#
# # video parameter
# fps = 30
# size = (708, 218) #(411, 218)
#
# for dir in os.listdir(image_dir):
#     # FL/RR~
#     subdir = os.path.join(image_dir, dir)
#     if not os.path.isdir(subdir):
#         continue
#
#     if ('fl' in dir) or ('rr' in dir) or ('sl' in dir) or ('sr' in dir) or ('sat' in dir):
#         print('process '+dir)
#     else:
#         continue
#
#     if 'sat' in dir:
#         size = (218, 218)
#
#     file_list = os.listdir(subdir)
#     num_list = [int(i.split('.')[0]) for i in file_list]
#     num_list.sort()
#     #file_list.sort()
#
#     video = cv2.VideoWriter(dir+".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
#     for num in num_list:
#         img = cv2.imread(os.path.join(subdir, str(num)+'.png'))
#         video.write(img)
#
#     video.release()
#     cv2.destroyAllWindows()


root_folder = "/home/users/u7094434/projects/SIDFM/visual_ford/pose_refine"
fps = 3
size = (436, 436)
file_list = os.listdir(root_folder)
file_list.sort()

video = cv2.VideoWriter("pose_refine.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
for file in file_list:
    img = cv2.imread(os.path.join(root_folder, file))
    video.write(img)

video.release()
cv2.destroyAllWindows()