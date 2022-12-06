
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from omegaconf import OmegaConf
import os
import math
from scipy.io import savemat

Ford_dataset = True
exp = 'ford'

from pixloc.pixlib.utils.tensor import batch_to_device, map_tensor
from pixloc.pixlib.utils.tools import set_seed
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.visualization.viz_2d import (
    plot_images, plot_keypoints, plot_matches, cm_RdGn,
    features_to_RGB, add_text, save_plot)
from pixloc.pixlib.models.utils import merge_confidence_map

data_conf = {
    'train_batch_size': 1,
    'test_batch_size': 1,
    'num_workers': 0,
    'mul_query': 0,
}


if Ford_dataset:
    from pixloc.pixlib.datasets.ford import FordAV
    dataset = FordAV(data_conf)
else:
    from pixloc.pixlib.datasets.kitti import Kitti
    dataset = Kitti(data_conf)

torch.set_grad_enabled(False);
mpl.rcParams['image.interpolation'] = 'bilinear'

loader = dataset.get_data_loader('train', shuffle=True)  # or 'train' ‘val’
val_loader = dataset.get_data_loader('val', shuffle=True)  # or 'train' ‘val’
test_loader = dataset.get_data_loader('test', shuffle=False) #shuffle=True)

# Name of the example experiment. Replace with your own training experiment.
device = 'cuda'
conf = {
    'normalize_dt': False,
    'optimizer': {'num_iters': 20,},
}
refiner = load_experiment(exp, conf,get_last=True).to(device)
print(OmegaConf.to_yaml(refiner.conf))

class Logger:
    def __init__(self, optimizers=None):
        self.costs = []
        self.dt = []
        #self.p2D_trajectory = []
        self.t = []
        self.camera_trajectory = []
        self.yaw_trajectory = []
        self.pre_q2r = None

        if optimizers is not None:
            for opt in optimizers:
                opt.logging_fn = self.log

    def log(self, **args):
        if args['i'] == 0:
            self.costs.append([])
            # add init and gt camera
            camera_3D = torch.zeros(1,3).to(args['T_delta'].device)
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_gt'] * camera_3D)
            self.camera_gt = camera_2D[0].cpu().numpy()
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_init'] * camera_3D)
            self.camera_trajectory.append((camera_2D[0].cpu().numpy(), valid[0].cpu().numpy()))
            camera_3D[:, 0] = 2
            camera_2D, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_gt'] * camera_3D)
            self.camera_gt_yaw = camera_2D[0].cpu().numpy()
            camera_yaw, valid = self.data['ref']['camera'].world2image(self.data['T_q2r_init'] * camera_3D)
            self.yaw_trajectory.append((camera_yaw[0].cpu().numpy(), valid[0].cpu().numpy()))
        self.costs[-1].append(args['cost'].mean(-1).cpu().numpy())
        self.dt.append(args['T_delta'].magnitude()[1].cpu().numpy())
        self.t.append(args['T'].cpu())

        camera_3D = torch.zeros(1,3).to(args['T_delta'].device)
        camera_2D, valid = self.data['ref']['camera'].world2image(args['T'] * camera_3D)
        camera_3D[:, 0] = 2
        camera_yaw, valid = self.data['ref']['camera'].world2image(args['T'] * camera_3D)
        self.camera_trajectory.append((camera_2D[0].cpu().numpy(), valid[0].cpu().numpy()))
        self.yaw_trajectory.append((camera_yaw[0].cpu().numpy(), valid[0].cpu().numpy()))

        self.pre_q2r = args['T'].cpu()

    def clear_trajectory(self):
        self.camera_trajectory = []
        self.yaw_trajectory = []
        self.t = []

    def set(self, data):
        self.data = data


logger = Logger(refiner.optimizer)
set_seed(20)

def min_max_norm(confidence):
    max= torch.max(confidence)
    min= torch.min(confidence)
    normed = (confidence - min) / (max - min + 1e-8)
    return normed

#val
def Val(refiner, val_loader, save_path, best_result):
    refiner.eval()
    acc = 0
    cnt = 0
    for idx, data in zip(range(2959), val_loader):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        cam_r = data['ref']['camera']
        p3D_q = pred['query']['grd_key_3d']
        p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam']*p3D_q)
        if data_conf['mul_query'] > 0:
            p2D_q_1, valid_q_1 = data['query_1']['camera'].world2image(data['query_1']['T_w2cam'] * p3D_q)
        if data_conf['mul_query'] > 1:
            p2D_q_2, valid_q_2 = data['query_2']['camera'].world2image(data['query_2']['T_w2cam'] * p3D_q)
            p2D_q_3, valid_q_3 = data['query_3']['camera'].world2image(data['query_3']['T_w2cam'] * p3D_q)
        p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        p2D_q_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        p2D_q_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)
        if data_conf['mul_query']==2:
            valid = (valid_q | valid_q_1 | valid_q_2 | valid_q_3) & valid_r
        elif data_conf['mul_query']==1:
            valid = (valid_q | valid_q_1 ) & valid_r
        else:
            valid = valid_q & valid_r

        losses = refiner.loss(pred_, data_)
        mets = refiner.metrics(pred_, data_)
        errP = f"ΔP {losses['reprojection_error/init'].item():.2f} -> {losses['reprojection_error'].item():.3f} px; "
        errR = f"ΔR {mets['R_error/init'].item():.2f} -> {mets['R_error'].item():.3f} deg; "
        errt = f"Δt {mets['t_error/init'].item():.2f} -> {mets['t_error'].item():.3f} m"
        errlat = f"Δlat {mets['lat_error/init'].item():.2f} -> {mets['lat_error'].item():.3f} m"
        errlong = f"Δlong {mets['long_error/init'].item():.2f} -> {mets['long_error'].item():.3f} m"
        print(errP, errR, errt, errlat,errlong)

        if mets['t_error'].item() < 1 and mets['R_error'].item() < 2:
            acc += 1
        cnt += 1

        # for debug
        if 1:
            confidence_map_count = 2
            if pred['ref']['confidences'][0].size(0) == 1:
                confidence_map_count = 1

            # plot points
            imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)
            # plot satellite
            plot_images([imr],
                        dpi=50,  # set to 100-200 for higher res
                        titles=[errP + errt])
            axes = plt.gcf().axes
            plot_keypoints([p2D_q_init[valid]], colors='red')
            start, _ = logger.camera_trajectory[0]
            end, _ = logger.yaw_trajectory[0]
            axes[0].quiver(start[:, 0], start[:, 1], end[:, 0] - start[:, 0], start[:, 1] - end[:, 1],
                           color='r')

            plot_keypoints([p2D_r_gt[valid]], colors='lime')
            axes[0].quiver(logger.camera_gt[:, 0], logger.camera_gt[:, 1],
                           logger.camera_gt_yaw[:, 0] - logger.camera_gt[:, 0],
                           logger.camera_gt[:, 1] - logger.camera_gt_yaw[:, 1], color='lime')

            plot_keypoints([p2D_q_opt[valid]], colors='blue')
            start, _ = logger.camera_trajectory[-1]
            end, _ = logger.yaw_trajectory[-1]
            axes[0].quiver(start[:, 0], start[:, 1], end[:, 0] - start[:, 0], start[:, 1] - end[:, 1],
                           color='b')
            add_text(0, 'reference')
            plt.show()

            if data_conf['mul_query'] > 0:
                imq_1  = data['query_1']['image'].permute(1, 2, 0)
            if data_conf['mul_query'] > 1:
                imq_2, imq_3  = data['query_2']['image'].permute(1, 2, 0), data['query_3']['image'].permute(1, 2, 0)

            # plt grd together
            if data_conf['mul_query'] > 1:
                plot_images([imq, imq_1, imq_2, imq_3],
                            dpi=50,  # set to 100-200 for higher res
                            titles=[valid_q.sum().item(), valid_q_1.sum().item(),valid_q_2.sum().item(),valid_q_3.sum().item()])
                plot_keypoints([p2D_q[valid_q],p2D_q_1[valid_q_1],p2D_q_2[valid_q_2],p2D_q_3[valid_q_3]], colors='lime')
            elif data_conf['mul_query'] > 0:
                plot_images([imq, imq_1],
                            dpi=50,  # set to 100-200 for higher res
                            titles=[valid_q.sum().item(), valid_q_1.sum().item()])
                plot_keypoints([p2D_q[valid_q],p2D_q_1[valid_q_1]], colors='lime')
            else:
                plot_images([imq],
                            dpi=50,  # set to 100-200 for higher res
                            titles=[valid_q.sum().item()])
                plot_keypoints([p2D_q[valid_q]], colors='lime')

            # add merged confidence map
            C_sat = merge_confidence_map(pred_['ref']['confidences'], confidence_map_count)  # [B,C,H,W]
            C_sat = min_max_norm(C_sat)
            C_sat = C_sat.cpu().numpy()[0, 0]
            plot_images([C_sat], cmaps=mpl.cm.gnuplot2, dpi=50)
            axes = plt.gcf().axes
            axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)
            plt.show()

            C_q = merge_confidence_map(pred_['query']['confidences'], confidence_map_count)  # [B,C,H,W]
            C_q = min_max_norm(C_q)
            C_q = C_q.cpu().numpy()[0, 0]
            if 'query_1' in pred_.keys():
                C_q1 = merge_confidence_map(pred_['query_1']['confidences'], confidence_map_count)  # [B,C,H,W]
                C_q1 = min_max_norm(C_q1)
                C_q1 = C_q1.cpu().numpy()[0, 0]
            if 'query_3' in pred_.keys():
                C_q2 = merge_confidence_map(pred_['query_2']['confidences'], confidence_map_count)  # [B,C,H,W]
                C_q2 = min_max_norm(C_q2)
                C_q2 = C_q2.cpu().numpy()[0, 0]
                C_q3 = merge_confidence_map(pred_['query_3']['confidences'], confidence_map_count)  # [B,C,H,W]
                C_q3 = min_max_norm(C_q3)
                C_q3 = C_q3.cpu().numpy()[0, 0]
            if 'query_3' in pred_.keys():
                plot_images([C_q, C_q1, C_q2, C_q3], cmaps=mpl.cm.gnuplot2, dpi=50)
            elif 'query_1' in pred_.keys():
                plot_images([C_q, C_q1], cmaps=mpl.cm.gnuplot2, dpi=50)
            else:
                plot_images([C_q], cmaps=mpl.cm.gnuplot2, dpi=50)
            axes = plt.gcf().axes
            axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)
            if 'query_1' in pred_.keys():
                axes[1].imshow(imq_1, alpha=0.2, extent=axes[1].images[0]._extent)
            if 'query_3' in pred_.keys():
                axes[2].imshow(imq_2, alpha=0.2, extent=axes[2].images[0]._extent)
                axes[3].imshow(imq_3, alpha=0.2, extent=axes[3].images[0]._extent)
            plt.show()

            # plot feature & confidence in each level
            for i in range(len(pred['ref']['feature_maps'])):  # level from fine to coarse
                # ref
                plot_images(features_to_RGB(pred['ref']['feature_maps'][i].numpy(), skip=1), dpi=50)
                add_text(0, f'ref_f_Level {i}')
                plt.show()
                for confi in range(confidence_map_count):
                    C_r = pred['ref']['confidences'][i][confi]
                    C_r = min_max_norm(C_r)
                    plot_images([C_r], cmaps=mpl.cm.gnuplot2, dpi=50)
                    add_text(0, f'ref_c_Level {i},{confi}')
                    axes = plt.gcf().axes
                    axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)
                    plt.show()

                # query
                if data_conf['mul_query'] > 1:
                    plot_images(features_to_RGB(pred['query']['feature_maps'][i].numpy(),
                                                pred['query_1']['feature_maps'][i].numpy(),
                                                pred['query_2']['feature_maps'][i].numpy(),
                                                pred['query_3']['feature_maps'][i].numpy(), skip=1), dpi=50)
                elif data_conf['mul_query'] > 0:
                    plot_images(features_to_RGB(pred['query']['feature_maps'][i].numpy(),
                                                pred['query_1']['feature_maps'][i].numpy(), skip=1), dpi=50)
                else:
                    plot_images(features_to_RGB(pred['query']['feature_maps'][i].numpy(), skip=1), dpi=50)
                add_text(0, f'query_f_Level {i}')
                plt.show()
                for confi in range(confidence_map_count):
                    C_q = pred['query']['confidences'][i][confi]
                    C_q = min_max_norm(C_q)
                    if data_conf['mul_query'] > 0:
                        C_q1 = pred['query_1']['confidences'][i][confi]
                        C_q1 = min_max_norm(C_q1)
                    if data_conf['mul_query'] > 1:
                        C_q2, C_q3 = pred['query_2']['confidences'][i][confi], pred['query_3']['confidences'][i][confi]
                        C_q2 = min_max_norm(C_q2)
                        C_q3 = min_max_norm(C_q3)
                    if data_conf['mul_query'] > 1:
                        plot_images([C_q, C_q1, C_q2, C_q3], cmaps=mpl.cm.gnuplot2, dpi=50)
                        add_text(0, f'query_c_Level {i},{confi}')
                        axes = plt.gcf().axes
                        axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)
                        axes[1].imshow(imq_1, alpha=0.2, extent=axes[1].images[0]._extent)
                        axes[2].imshow(imq_2, alpha=0.2, extent=axes[2].images[0]._extent)
                        axes[3].imshow(imq_3, alpha=0.2, extent=axes[3].images[0]._extent)
                    elif data_conf['mul_query'] > 0:
                        plot_images([C_q, C_q1], cmaps=mpl.cm.gnuplot2, dpi=50)
                        add_text(0, f'query_c_Level {i},{confi}')
                        axes = plt.gcf().axes
                        axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)
                        axes[1].imshow(imq_1, alpha=0.2, extent=axes[1].images[0]._extent)
                    else:
                        plot_images([C_q], cmaps=mpl.cm.gnuplot2, dpi=50)
                        add_text(0, f'query_c_Level {i},{confi}')
                        axes = plt.gcf().axes
                        axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)
                    plt.show()

            costs = logger.costs
            fig, axes = plt.subplots(1, len(costs), figsize=(len(costs)*4.5, 4.5))
            for i, (ax, cost) in enumerate(zip(axes, costs)):
                ax.plot(cost) if len(cost)>1 else ax.scatter(0., cost)
                ax.set_title(f'({i}) Scale {i//3} Level {i%3}')
                ax.grid()
            plt.show()

            colors = mpl.cm.cool(1 - np.linspace(0, 1, len(logger.camera_trajectory)))[:, :3]
            plot_images([imr])
            axes = plt.gcf().axes
            for i,(p2s, _), (p2e, _), T, c in zip(range(len(logger.camera_trajectory)), logger.camera_trajectory, logger.yaw_trajectory, logger.t, colors):
                # plot the direction of the body frame
                if i == 0:
                    start_0 = p2s
                    end_0 = p2e
                    axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color='r')
                elif i == len(logger.camera_trajectory)-1:
                    axes[0].quiver(p2s[:, 0], p2s[:, 1], p2e[:, 0] - p2s[:, 0], p2s[:, 1] - p2e[:, 1], color='b')
                    axes[0].quiver(start_0[:, 0], start_0[:, 1], end_0[:, 0] - start_0[:, 0], start_0[:, 1] - end_0[:, 1], color='r')
                else:
                    axes[0].quiver(p2s[:,0], p2s[:,1], p2e[:,0]-p2s[:,0], p2s[:,1]-p2e[:,1], color=c[None])
            axes[0].quiver(logger.camera_gt[:, 0], logger.camera_gt[:, 1], logger.camera_gt_yaw[:, 0]-logger.camera_gt[:, 0],
                           logger.camera_gt[:, 1]-logger.camera_gt_yaw[:, 1], color='lime')
            logger.clear_trajectory()
            plt.show()

    acc = acc/cnt
    print('acc of a epoch:#####',acc)
    if acc > best_result:
        print('best acc:@@@@@', acc)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(refiner.state_dict(), save_path + 'Model_best.pth')
    return acc

def test(refiner, test_loader):
    refiner.eval()
    errR = torch.tensor([])
    errlong = torch.tensor([])
    errlat = torch.tensor([])
    for idx, data in enumerate(test_loader):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred_ = refiner(data_)
        #losses = refiner.loss(pred_, data_)
        metrics = refiner.metrics(pred_, data_)

        errR = torch.cat([errR, metrics['R_error'].cpu().data], dim=0)
        errlong = torch.cat([errlong, metrics['long_error'].cpu().data], dim=0)
        errlat = torch.cat([errlat, metrics['lat_error'].cpu().data], dim=0)

        del pred_, data_

    # if lat <= 0.2 and long <= 0.4 and R < 1: #requerment of Ford
    print(f'acc of lat<=0.25:{torch.sum(errlat <= 0.25) / errlat.size(0)}')
    print(f'acc of lat<=0.5:{torch.sum(errlat <= 0.5) / errlat.size(0)}')
    print(f'acc of lat<=1:{torch.sum(errlat <= 1) / errlat.size(0)}')
    print(f'acc of lat<=2:{torch.sum(errlat <= 2) / errlat.size(0)}')

    print(f'acc of long<=0.25:{torch.sum(errlong <= 0.25) / errlong.size(0)}')
    print(f'acc of long<=0.5:{torch.sum(errlong <= 0.5) / errlong.size(0)}')
    print(f'acc of long<=1:{torch.sum(errlong <= 1) / errlong.size(0)}')
    print(f'acc of long<=2:{torch.sum(errlong <= 2) / errlong.size(0)}')

    # print(f'acc of R<=0.5:{torch.sum(errR <= 0.5) / errR.size(0)}')
    print(f'acc of R<=1:{torch.sum(errR <= 1) / errR.size(0)}')
    print(f'acc of R<=2:{torch.sum(errR <= 2) / errR.size(0)}')
    print(f'acc of R<=4:{torch.sum(errR <= 4) / errR.size(0)}')

    print(f'mean errR:{torch.mean(errR)},errlat:{torch.mean(errlat)},errlong:{torch.mean(errlong)}')
    print(f'var errR:{torch.var(errR)},errlat:{torch.var(errlat)},errlong:{torch.var(errlong)}')
    print(f'median errR:{torch.median(errR)},errlat:{torch.median(errlat)},errlong:{torch.median(errlong)}')

    return

def angle_from_Rmatrix(R):
    '''

    :param R:
    :return: roll x, pitch:y, yaw:z
    '''
    if R[2,0] != 1 and R[2,0] != -1:
         pitch_1 = -1*math.asin(R[2,0])
         pitch_2 = math.pi - pitch_1
         roll_1 = math.atan2( R[2,1] / math.cos(pitch_1) , R[2,2] /math.cos(pitch_1) )
         roll_2 = math.atan2( R[2,1] / math.cos(pitch_2) , R[2,2] /math.cos(pitch_2) )
         yaw_1 = math.atan2( R[1,0] / math.cos(pitch_1) , R[0,0] / math.cos(pitch_1) )
         yaw_2 = math.atan2( R[1,0] / math.cos(pitch_2) , R[0,0] / math.cos(pitch_2) )

         # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
         # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info).
         sol_1 = np.array([roll_1, pitch_1, yaw_1])
         sol_2 = np.array([roll_2, pitch_2, yaw_2])
    else:
         yaw = 0 # anything (we default this to zero)
         if R[2,0] == -1:
            pitch = math.pi/2
            roll = yaw + math.atan2(R[0,1],R[0,2])
         else:
            pitch = -match.pi/2
            roll = -1*yaw + math.atan2(-1*R[0,1],-1*R[0,2])
         sol_1 = np.array([roll, pitch, yaw])
         sol_2 = None

    return sol_1, sol_2

# save_root = "/data/dataset/Ford_AV/2017-10-26-V2-Log4"
save_root = '../visual_kitti'
def SaveConfidencePose(refiner, test_loader):
    shift_pose = []
    refiner.eval()
    for idx, data in enumerate(test_loader):
        data_ = batch_to_device(data, device)
        logger.set(data_)
        pred = refiner(data_)
        #pred = map_tensor(pred_, lambda x: x[0].cpu())
        data = map_tensor(data, lambda x: x[0].cpu())
        # cam_r = data['ref']['camera']
        # if 'points3D' in data['query'].keys():
        #     p3D_q = data['query']['points3D']
        # elif 'grd_key_3d' in pred['query'].keys():
        #     # grd keypoints detection
        #     p3D_q = pred['query']['grd_key_3d']
        # else:
        #     p3D_q = None
        #
        # p2D_q, valid_q = data['query']['camera'].world2image(data['query']['T_w2cam'] * p3D_q)
        # if data_conf['mul_query']:
        #     p2D_q_1, valid_q_1 = data['query_1']['camera'].world2image(data['query_1']['T_w2cam'] * p3D_q)
        #     p2D_q_2, valid_q_2 = data['query_2']['camera'].world2image(data['query_2']['T_w2cam'] * p3D_q)
        #     p2D_q_3, valid_q_3 = data['query_3']['camera'].world2image(data['query_3']['T_w2cam'] * p3D_q)
        # p2D_r_gt, valid_r = cam_r.world2image(data['T_q2r_gt'] * p3D_q)
        # p2D_q_init, _ = cam_r.world2image(data['T_q2r_init'] * p3D_q)
        # p2D_q_opt, _ = cam_r.world2image(pred['T_q2r_opt'][-1] * p3D_q)
        # if data_conf['mul_query']:
        #     valid = (valid_q | valid_q_1 | valid_q_2 | valid_q_3) & valid_r
        # else:
        #     valid = valid_q & valid_r

        # save pose
        T_r2q_gt = data['T_q2r_gt'].inv()
        T_q2r = logger.pre_q2r
        diff_sat_pose = T_q2r@T_r2q_gt
        shift_north, shift_east = diff_sat_pose.shift_NE()
        # get yaw on axis z
        sat_R, _ = angle_from_Rmatrix(diff_sat_pose.R[0])
        yaw = sat_R[-1]
        # yaw must in +-30 degree
        if math.fabs(yaw) > 30./math.pi:
            yaw = yaw - np.sign(yaw)*math.pi
        #print(yaw, shift_north.item(), shift_east.item() )
        shift_pose.append([yaw, shift_north.item(), shift_east.item()])

        confidence_map_count = 2
        if 'points3D_type' in data['query'].keys():
            if data['query']['points3D_type'] == ['lidar']:
                confidence_map_count = 1

        imr, imq = data['ref']['image'].permute(1, 2, 0), data['query']['image'].permute(1, 2, 0)

        # save confidence map
        # satellite map
        C_sat = merge_confidence_map(pred['ref']['confidences'], confidence_map_count) #[B,C,H,W]
        C_sat = C_sat.cpu().numpy()[0,0]
        plot_images([C_sat], cmaps=mpl.cm.gnuplot2, dpi=50)
        axes = plt.gcf().axes
        axes[0].imshow(imr, alpha=0.2, extent=axes[0].images[0]._extent)

        if not os.path.exists(os.path.join(save_root,'confidence_maps')):
            os.makedirs(os.path.join(save_root,'confidence_maps'))
        if not os.path.exists(os.path.join(save_root,'confidence_maps/sat')):
            os.makedirs(os.path.join(save_root,'confidence_maps/sat'))
        save_plot(os.path.join(save_root, 'confidence_maps/sat', str(idx) + '.png'))

        # query images
        C_q = merge_confidence_map(pred['query']['confidences'], confidence_map_count)  # [B,C,H,W]
        C_q = C_q.cpu().numpy()[0, 0]
        plot_images([C_q], cmaps=mpl.cm.gnuplot2, dpi=50)
        axes = plt.gcf().axes
        axes[0].imshow(imq, alpha=0.2, extent=axes[0].images[0]._extent)

        if not os.path.exists(os.path.join(save_root,'confidence_maps/fl')):
            os.makedirs(os.path.join(save_root,'confidence_maps/fl'))
        save_plot(os.path.join(save_root, 'confidence_maps/fl', str(idx) + '.png'))

        if data_conf['mul_query']:
            imq_1, imq_2, imq_3 = data['query_1']['image'].permute(1, 2, 0), data['query_2']['image'].permute(1, 2, 0), \
                                  data['query_3']['image'].permute(1, 2, 0)

            # rr images
            C_q1 = merge_confidence_map(pred['query_1']['confidences'], confidence_map_count)  # [B,C,H,W]
            C_q1 = C_q1.cpu().numpy()[0, 0]
            plot_images([C_q1], cmaps=mpl.cm.gnuplot2, dpi=50)
            axes = plt.gcf().axes
            axes[0].imshow(imq_1, alpha=0.2, extent=axes[0].images[0]._extent)
            if not os.path.exists(os.path.join(save_root, 'confidence_maps/rr')):
                os.makedirs(os.path.join(save_root, 'confidence_maps/rr'))
            save_plot(os.path.join(save_root, 'confidence_maps/rr', str(idx) + '.png'))

            # sl images
            C_q2 = merge_confidence_map(pred['query_2']['confidences'], confidence_map_count)  # [B,C,H,W]
            C_q2 = C_q2.cpu().numpy()[0, 0]
            plot_images([C_q2], cmaps=mpl.cm.gnuplot2, dpi=50)
            axes = plt.gcf().axes
            axes[0].imshow(imq_2, alpha=0.2, extent=axes[0].images[0]._extent)
            if not os.path.exists(os.path.join(save_root, 'confidence_maps/sl')):
                os.makedirs(os.path.join(save_root, 'confidence_maps/sl'))
            save_plot(os.path.join(save_root, 'confidence_maps/sl', str(idx) + '.png'))

            # sr images
            C_q3 = merge_confidence_map(pred['query_3']['confidences'], confidence_map_count)  # [B,C,H,W]
            C_q3 = C_q3.cpu().numpy()[0, 0]
            plot_images([C_q3], cmaps=mpl.cm.gnuplot2, dpi=50)
            axes = plt.gcf().axes
            axes[0].imshow(imq_3, alpha=0.2, extent=axes[0].images[0]._extent)
            if not os.path.exists(os.path.join(save_root, 'confidence_maps/sr')):
                os.makedirs(os.path.join(save_root, 'confidence_maps/sr'))
            save_plot(os.path.join(save_root, 'confidence_maps/sr', str(idx) + '.png'))

    with open(os.path.join(save_root, 'shift_pose.npy'), 'wb') as f:
        np.save(f, shift_pose)
    return

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    save_path = 'parameter'

    if 0: # test
        #test(refiner, test_loader)
        test(refiner, val_loader)
    if 1: # val
        Val(refiner, val_loader, save_path, 0)
    if 0: # save confidence map
        SaveConfidencePose(refiner, test_loader)
