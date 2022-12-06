# check the matching relationship between cross-view images

from input_libs import *
from angle_func import convert_body_yaw_to_360
from pose_func import quat_from_pose, read_calib_yaml, read_txt, read_numpy
import gps_coord_func as gps_func
from project_grd_images import project_grd_to_map, project_grd_to_map_grd, fuse
from transformations import quaternion_matrix,rotation_matrix

root_folder = "/data/dataset/Ford_AV"

drive_id = "20171026"
log_id = "2017-10-26-V2-Log4"
type = '-SL'#'-RR'#'-FL'
# size of the satellite image and ground-view query image (left camera)
satellite_size = 1280
query_size = [1656, 860]
zoom = 18


calib_folder = os.path.join(root_folder, "V2")
#log_folder = os.path.join(root_folder , drive_id , log_id)
log_folder = os.path.join(root_folder , log_id)

if type == '-FL':
    camera_body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
    cameraIntrinsics = read_calib_yaml(calib_folder, "cameraFrontLeftIntrinsics.yaml")
elif type == '-RR':
    camera_body = read_calib_yaml(calib_folder, "cameraRearRight_body.yaml")
    cameraIntrinsics = read_calib_yaml(calib_folder, "cameraRearRightIntrinsics.yaml")
elif type == '-SL':
    camera_body = read_calib_yaml(calib_folder, "cameraSideLeft_body.yaml")
    cameraIntrinsics = read_calib_yaml(calib_folder, "cameraSideLeftIntrinsics.yaml")
elif type == '-SR':
    camera_body = read_calib_yaml(calib_folder, "cameraSideRight_body.yaml")
    cameraIntrinsics = read_calib_yaml(calib_folder, "cameraSideRightIntrinsics.yaml")

FL_image_names = read_txt( os.path.join(log_folder,'info_files'), log_id + '-FL-names.txt')
FL_image_names.pop(0)
nb_query_images = len(FL_image_names)

groundview_gps = read_numpy(os.path.join(log_folder,'info_files'), 'groundview_gps.npy') # 'groundview_gps_2.npy'
groundview_yaws = read_numpy(os.path.join(log_folder,'info_files'), 'groundview_yaws_pose_gt.npy') # 'groundview_yaws_pose.npy'
groundview_rolls = read_numpy(os.path.join(log_folder,'info_files'), 'groundview_rolls_pose_gt.npy')  # 'groundview_yaws_pose.npy'
groundview_pitchs = read_numpy(os.path.join(log_folder,'info_files'), 'groundview_pitchs_pose_gt.npy')  # 'groundview_yaws_pose.npy'

query_image_folder = os.path.join(log_folder , log_id + type)
# get the satellite images
satellite_folder = os.path.join(log_folder , "Satellite_Images_18")
satellite_names = glob.glob(satellite_folder + '/*.png')

nb_satellite_images = len(satellite_names)

satellite_dict = {}
for i in range(nb_satellite_images):
    cur_sat = int(os.path.split(satellite_names[i])[-1].split("_")[1])
    satellite_dict[cur_sat] = satellite_names[i]

# 3. read the matching pair
match_pair = read_numpy(os.path.join(log_folder,'info_files') , 'groundview_satellite_pair.npy') # 'groundview_satellite_pair_2.npy'

# 4. for each matching pair, set a local coordinate system on the satellite image
# and project the satellite points to the ground-view query image


query_y_coords, query_x_coords =  np.meshgrid(np.arange(query_size[1]), np.arange(query_size[0]), indexing='ij')
query_y_coords, query_x_coords = query_y_coords.reshape( -1), query_x_coords.reshape( -1)
query_xy_hom = np.stack ((query_x_coords, query_y_coords, np.ones_like(query_x_coords)), axis=0)

sate_y_coords, sate_x_coords =  np.meshgrid(np.arange(satellite_size), np.arange(satellite_size), indexing='ij')
sate_y_coords, sate_x_coords = sate_y_coords.reshape( -1), sate_x_coords.reshape( -1)

def trans_from_pose(trans):

    x = trans['transform']['translation']['x']
    y = trans['transform']['translation']['y']
    z = trans['transform']['translation']['z']

    return [x,y,z]


# for i in range(7520, nb_query_images, 20):
# for i in range(0, nb_query_images, 2out_img0):
for i in range(1000, nb_query_images, 100):
# for i in range(9660, nb_query_images, 20):
# for i in range(10000, nb_query_images, 20):
    query_gps = groundview_gps[i,:]
    # print(satellite_dict[match_pair[i]])
    # print(query_gps)
    # print(groundview_yaws[i])
    satellite_img = os.path.split(satellite_dict[match_pair[i]])[-1].split("_")
    satellite_gps = [float(satellite_img[3]), float(satellite_img[5])]

    # using the satellite image as the reference and calculate the offset of the ground-view query
    dx, dy = gps_func.angular_distance_to_xy_distance_v2(satellite_gps[0], satellite_gps[1], query_gps[0], query_gps[1])

    # get the current resolution of satellite image
    # a scale at 2 when downloading the dataset
    sat_res = 156543.03392 * np.cos(satellite_gps[0] * np.pi / 180.0) / np.power(2,zoom) / 2.0

    # get the pixel offsets
    # along the east direction
    dx_pixel = dx / sat_res
    # along the north direction
    dy_pixel = -dy / sat_res

    # left corner as the origin
    # let's project this satellite point on the query image
    #yaw = convert_body_yaw_to_360(groundview_yaws[i])* np.pi / 180.0
    yaw = groundview_yaws[i] * np.pi / 180.0
    roll = groundview_rolls[i] * np.pi / 180.0
    pitch = groundview_pitchs[i] * np.pi / 180.0

    cos = np.cos(pitch)
    sin = np.sin(pitch)
    R_y = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]) # shape = [9]
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    R_z = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]) # shape = [9]
    cos = np.cos(roll)
    sin = np.sin(roll)
    R_x = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]]) # shape = [9]
    R_body2ned =R_z@R_y@R_x
    # FL2body = read_calib_yaml(calib_folder, "cameraFrontLeft_body.yaml")
    camera_relPose_body = quaternion_matrix(quat_from_pose(camera_body))

    # add rotation for left & right
    if type == '-SL' or type == '-SR':
        add = -3*np.pi/180. #SR:3;SL:
        R_x = rotation_matrix(add, [1,0,0])
        camera_relPose_body = R_x@camera_relPose_body


    camera_relTrans_body = trans_from_pose(camera_body)
    # ned->body->cam
    R_ned2cam = (camera_relPose_body[:3,:3].T)@(R_body2ned.T)

    K_mat = np.asarray(cameraIntrinsics['K']).reshape(3,3)
    grd_query = mpimg.imread(os.path.join(query_image_folder , FL_image_names[i][:-1]))
    grd_proj = project_grd_to_map_grd(grd_query, R_ned2cam, K_mat, 960, satellite_gps[0], zoom)

    # transform the x axis in the body to the camera (corresponds to z axis)
    cam_shift = R_body2ned @ camera_relTrans_body

    query_pixel_x = dx_pixel + satellite_size / 2.0 + cam_shift[1] / sat_res
    query_pixel_y = dy_pixel + satellite_size / 2.0 - cam_shift[0] / sat_res
    start_x = int(np.round(query_pixel_x-960/2))
    start_y = int(np.round(query_pixel_y-960/2))
    sate_img = mpimg.imread(satellite_dict[match_pair[i]])
    croped_sat = sate_img[start_y:start_y+960,start_x:start_x+960,:]
    fusion = fuse(grd_proj, croped_sat)
    plt.imshow(fusion)
    #plt.imshow(grd_proj.permute(1, 2, 0))
    plt.show()

#
# # --------------------------------------------------------------------
#     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     # currently, the miss-alignment between the body yaw and cam yaw is not considered!
#     # correct it
#     FL_relPose_body = transformations.quaternion_matrix(quat_from_pose(FL_cameraFrontLeft_body))
#     # transform the x axis in the body to the camera (corresponds to z axis)
#     body_x_axis_in_cam = FL_relPose_body[:3,:3].T @ np.array([1, 0, 0])
#     yaw_cam_body = np.arctan2(body_x_axis_in_cam[0], body_x_axis_in_cam[2]) * 180 / np.pi
#
#     offset_body = [FL_cameraFrontLeft_body['transform']['translation']['x'], FL_cameraFrontLeft_body['transform']['translation']['y']]
# # --------------------------------------------------------------------
#     # add the offset between camera and body to shift the center to query camera
#     dx_cam = offset_body[1] / sat_res
#     dy_cam = -offset_body[0] / sat_res
#     # convert the offset to ploar coordinate
#     tan_theta = -offset_body[1]/ offset_body[0]
#     length_dxy = np.sqrt(offset_body[0] * offset_body[0] + offset_body[1] * offset_body[1])
#     offset_cam_theta = np.arctan(tan_theta)
#     yaw_FL = offset_cam_theta + body_yaws
#     dx_cam_pixel = np.cos(yaw_FL) * length_dxy / sat_res
#     dy_cam_pixel = -np.sin(yaw_FL) * length_dxy / sat_res
# # --------------------------------------------------------------------
#
#     yaw_local_camera_normlaized = np.arctan2(ray_local_camera_normalized[0], ray_local_camera_normalized[2]) * 180 / np.pi
#     # yaw_local_body_normlaized = - np.arctan2(ray_local_camera_normalized[2], ray_local_camera_normalized[0]) * 180 / np.pi + 90.0
#     yaw_local_camera_normlaized = yaw_local_camera_normlaized
#     yaw_max = np.amax(yaw_local_camera_normlaized)
#     yaw_min = np.amin(yaw_local_camera_normlaized)
#     yaw_local_camera_normlaized_image = yaw_local_camera_normlaized.reshape(query_size[1], query_size[0])
#
# # -------------------------------------------------------
# #     # show the query postion (camera) on the satellite map
#     query_pixel_x = dx_pixel + satellite_size / 2.0 + dx_cam_pixel
#     query_pixel_y = dy_pixel + satellite_size / 2.0 + dy_cam_pixel
#     # given the projection center, calculate the yaw angle of the satellite map with respect to the center
#     yaw_sate = (np.arctan2((-sate_y_coords + query_pixel_y) , (sate_x_coords - query_pixel_x)) % (2.0*np.pi)) * 180.0 / np.pi
#     yaw_sate_image = yaw_sate.reshape(satellite_size, satellite_size)
#     # using the body direction as the reference, and warp the satellite yaw angle
#     yaw_sate_image = body_yaws * 180.0 / np.pi - yaw_sate_image
#     yaw_sate_image = np.where(yaw_sate_image > 180.0, yaw_sate_image - 360.0, yaw_sate_image)
#     yaw_sate_max = np.amax(yaw_sate)
#     yaw_sate_min = np.amin(yaw_sate)
#
#
# # -------------------------------------------------------
#     grd_query = mpimg.imread(os.path.join(query_image_folder , FL_image_names[i][:-1]))
#     sate_img = mpimg.imread(satellite_dict[match_pair[i]])
#     # draw the query position on the satellite image
#     fig = plt.figure(figsize=plt.figaspect(0.5))
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax4 = fig.add_subplot(2, 2, 4)
#
#     ax1.imshow(grd_query)
#     ax2.imshow(sate_img)
#
#     grd_proj = project_grd_to_map_grd(grd_query, R_mat, K_mat, 960, satellite_gps[0], zoom) # yaw_body or FL
#
#     ax3.imshow(grd_proj.permute(1, 2, 0))
#     # --------------------------------------------------------
#     # fusion project grd and sat
#     start_x = int(np.round(query_pixel_x-960/2))
#     start_y = int(np.round(query_pixel_y-960/2))
#     croped_sat = sate_img[start_y:start_y+960,start_x:start_x+960,:]
#     fusion = fuse(grd_proj, croped_sat)
#     ax4.imshow(fusion)
#
#     # camera position
#     ax2.scatter(x=query_pixel_x, y=query_pixel_y, c='r', s=20)
#     # body position
#     ax2.scatter(x=query_pixel_x - dx_cam_pixel, y=query_pixel_y - dy_cam_pixel, c='g', s=20)
#
#
#     # given the body yaws, let's calculate the fovs of camera
#     camera_yaws_left = body_yaws  - yaw_min* np.pi / 180.0
#     camera_yaws_right = body_yaws  - yaw_max* np.pi / 180.0
# # -------------------------------------------------------
#     # plot the direction of the body frame
#     length_xy = 200.0
#     origin = np.array([[query_pixel_x],[query_pixel_y]]) # origin point
#     dx_east = length_xy * np.cos(body_yaws)
#     dy_north = length_xy * np.sin(body_yaws)
#     # x = [query_pixel_x, query_pixel_x + dx_east]
#     # y = [query_pixel_y, query_pixel_y - dy_north]
#     V = np.array([[dx_east,dy_north]])
#     ax2.quiver(*origin, V[:,0], V[:,1], color=['r'], scale = 1000)
#
#     # plot the direction of the left-fov of camera
#     dx_east = length_xy * np.cos(camera_yaws_left)
#     dy_north = length_xy * np.sin(camera_yaws_left)
#     V = np.array([[dx_east, dy_north]])
#     ax2.quiver(*origin, V[:, 0], V[:, 1], color=['g'], scale=1000)
#     # plot the direction of the right-fov of camera
#     dx_east = length_xy * np.cos(camera_yaws_right)
#     dy_north = length_xy * np.sin(camera_yaws_right)
#     V = np.array([[dx_east, dy_north]])
#     ax2.quiver(*origin, V[:, 0], V[:, 1], color=['b'], scale=1000)
#
#     plt.show()
