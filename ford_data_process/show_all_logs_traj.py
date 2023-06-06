# this script shows the trajectories of all logs
from input_libs import *
import gps_coord_func as gps_func
from pose_func import read_numpy
from transformations import euler_matrix

root_dir = "/data/dataset/Ford_AV/"
Logs = ["2017-10-26-V2-Log4",]
        # "2017-10-26-V2-Log1",
        # # "2017-10-26-V2-Log2",
        # # "2017-10-26-V2-Log3",
        # # "2017-10-26-V2-Log4",
        # "2017-08-04-V2-Log1",
        # "2017-10-26-V2-Log5",
        # "2017-08-04-V2-Log5"]
        # # "2017-10-26-V2-Log6"]

nb_logs = len(Logs)

all_gps_neds = []
all_slam_neds = []

for log_iter in range(nb_logs):

    cur_log_dir = root_dir + Logs[log_iter]
    # read the raw gps data
    # gps_file = cur_log_dir + "/info_files/gps.csv"
    # with open(gps_file, newline='') as f:
    #     reader = csv.reader(f)
    #     gps_data = list(reader)
    # # remove the headlines
    # gps_data.pop(0)
    #
    # cur_nb_gps = len(gps_data)
    #
    # gps_geodestic = np.zeros((cur_nb_gps, 3))
    #
    # # convert to NED position
    # gps_ned = np.zeros((cur_nb_gps, 3))
    #
    # for i, line in zip(range(cur_nb_gps), gps_data):
    #     gps_geodestic[i, 0] = float(line[10])
    #     gps_geodestic[i, 1] = float(line[11])
    #     gps_geodestic[i, 2] = float(line[12])
    #     x, y, z = gps_func.GeodeticToEcef(gps_geodestic[i, 0] * np.pi / 180.0, gps_geodestic[i, 1] * np.pi / 180.0,
    #                                       gps_geodestic[i, 2])
    #     xEast, yNorth, zUp = gps_func.EcefToEnu(x, y, z, gps_func.gps_ref_lat, gps_func.gps_ref_long,
    #                                             gps_func.gps_ref_height)
    #     gps_ned[i, 0] = yNorth
    #     gps_ned[i, 1] = xEast
    #     gps_ned[i, 2] = -zUp

    # read the ned gt pose
    query_ned = read_numpy(cur_log_dir+"/info_files", "groundview_NED_pose_gt.npy")

    # all_gps_neds.append(gps_ned)
    all_slam_neds.append(query_ned)

    # # plot the gps trajectories
    # color_list = ['g', 'b', 'r', 'k', 'c', 'm', 'y']
    #
    # for log_iter in range(nb_logs):
    #     plt.plot(all_gps_neds[log_iter][:, 1], all_gps_neds[log_iter][:, 0], color=color_list[log_iter], linewidth=1,
    #              label='log_{}'.format(log_iter))
    #     # plt.plot(all_slam_neds[log_iter][:, 1], all_slam_neds[log_iter][:, 0], color=color_list[log_iter], linewidth=1, linestyle="--",
    #     #          label='log_{}'.format(log_iter))
    #
    # plt.xlabel("East")
    # plt.ylabel("North")
    # plt.legend()
    # plt.show()

    # read the shift pose
    shift_dir = '/home/users/u7094434/projects/SIDFM/pixloc/ford_shift_100/'
    shift_R = read_numpy(shift_dir, "pred_R.np") #  pre->gt
    shift_T = read_numpy(shift_dir, "pred_T.np") #  pre->gt
    yaws = read_numpy(cur_log_dir+"/info_files", 'groundview_yaws_pose_gt.npy')  # 'groundview_yaws_pose.npy'
    yaws = yaws*np.pi / 180.0
    rolls = read_numpy(cur_log_dir+"/info_files", 'groundview_rolls_pose_gt.npy')  # 'groundview_yaws_pose.npy'
    rolls = rolls * np.pi / 180.0
    pitchs = read_numpy(cur_log_dir+"/info_files", 'groundview_pitchs_pose_gt.npy')  # 'groundview_yaws_pose.npy'
    pitchs = pitchs * np.pi / 180.0

    shift_ned = []
    for i, line in zip(range(len(query_ned)), query_ned):
        body2ned = euler_matrix(rolls[i], pitchs[i], yaws[i])
        shift_ned.append(query_ned[i] + body2ned[:3,:3]@shift_T[i])
    shift_ned = np.array(shift_ned)


    # plot the trajectories
    Min_Pose = 1275#627
    Max_Pose = 1430#2596#-1 #2596
    plt.plot(query_ned[Min_Pose:Max_Pose, 1], query_ned[Min_Pose:Max_Pose, 0], alpha=0.5, color='r', marker='o', markersize=2) #linewidth=1)
    plt.plot(shift_ned[Min_Pose:Max_Pose, 1], shift_ned[Min_Pose:Max_Pose, 0], alpha=0.5, color='b', marker='o', markersize=2)# linewidth=1)

    plt.xlabel("East(m)")
    plt.ylabel("North(m)")
    plt.legend()
    plt.show()

    temp = 0
