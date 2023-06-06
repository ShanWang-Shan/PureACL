"""
Flexible UNet model which takes any Torchvision backbone as encoder.
Predicts multi-level feature and uncertainty maps
and makes sure that they are well aligned.
"""

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .utils import checkpointed
from copy import deepcopy
from sidfm.pixlib.models.utils import camera_to_onground

# for 1 unet test
two_confidence = False # False when only grd
max_dis = 200
debug_pe = False
visualize = False

updown_fusion = 1 #0: without 1:transformer, 2: linear, 3:nerf
if updown_fusion == 1:
    from .fusion_topdown import Fusion_topdown

if debug_pe:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl
    from sidfm.visualization.viz_2d import plot_images

class DecoderBlock(nn.Module):
    def __init__(self, previous, skip, out, num_convs=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous+skip if i == 0 else out, out,
                kernel_size=3, padding=1, bias=norm is None)
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

        # norm is instanceNorm2d when batch is 1
        #self.norm = nn.InstanceNorm2d(out)

    def forward(self, previous, skip):
        upsampled = self.upsample(previous)
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        assert (hu <= hs) and (wu <= ws), 'Using ceil_mode=True in pooling?'
        # assert (hu == hs) and (wu == ws), 'Careful about padding'
        skip = skip[:, :, :hu, :wu]

        # norm is instanceNorm2d when batch is 1
        return self.layers(torch.cat([upsampled, skip], dim=1))
        # if previous.size(0) == 1:
        #     if len(self.layers) >= 3:
        #         # bn is in layers[1]
        #         out = self.layers[0](torch.cat([upsampled, skip], dim=1))
        #         if torch.isnan(out).any():
        #             print('nan in decoder conv')
        #         out = self.norm(out)
        #         if torch.isnan(out).any():
        #             print('nan in decoder inorm')
        #         return self.layers[2:](out)
        # else:
        #     return self.layers(torch.cat([upsampled, skip], dim=1))

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)


class UNet(BaseModel):
    default_conf = {
        'output_scales': [0, 2, 4],  # what scales to adapt and output
        'output_dim': [32, 128, 128],  # # of channels in output feature maps
        'encoder': 'vgg16',  # string (torchvision net) or list of channels
        'num_downsample': 4,  # how many downsample block (if VGG-style net)
        'decoder': [64, 64, 64, 32],  # list of channels of decoder
        'decoder_norm': 'nn.BatchNorm2d',  # normalization ind decoder blocks
        'do_average_pooling': False,
        'compute_uncertainty': True,
        'checkpointed': False,  # whether to use gradient checkpointing
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def build_encoder(self, conf):
        assert isinstance(conf.encoder, str)
        Encoder = getattr(torchvision.models, conf.encoder)
        encoder = Encoder(pretrained=True)
        Block = checkpointed(torch.nn.Sequential, do=conf.checkpointed)

        if conf.encoder.startswith('vgg'):
            # Parse the layers and pack them into downsampling blocks
            # It's easy for VGG-style nets because of their linear structure.
            # This does not handle strided convs and residual connections
            assert max(conf.output_scales) <= conf.num_downsample
            skip_dims = []
            previous_dim = None
            blocks = [[]]
            for i, layer in enumerate(encoder.features):
                if isinstance(layer, torch.nn.Conv2d):
                    previous_dim = layer.out_channels
                elif isinstance(layer, torch.nn.MaxPool2d):
                    assert previous_dim is not None
                    skip_dims.append(previous_dim)
                    if (conf.num_downsample + 1) == len(blocks):
                        break
                    blocks.append([])
                    if conf.do_average_pooling:
                        assert layer.dilation == 1
                        layer = torch.nn.AvgPool2d(
                            kernel_size=layer.kernel_size, stride=layer.stride,
                            padding=layer.padding, ceil_mode=layer.ceil_mode,
                            count_include_pad=False)
                blocks[-1].append(layer)
            assert (conf.num_downsample + 1) == len(blocks)
            encoder = [Block(*b) for b in blocks]
        elif conf.encoder.startswith('resnet'):
            # Manually define the splits - this could be improved
            assert conf.encoder[len('resnet'):] in ['18', '34', '50', '101']
            block1 = torch.nn.Sequential(encoder.conv1, encoder.bn1,
                                         encoder.relu)
            block2 = torch.nn.Sequential(encoder.maxpool, encoder.layer1)
            block3 = encoder.layer2
            block4 = encoder.layer3
            blocks = [block1, block2, block3, block4]
            encoder = [torch.nn.Identity()] + [Block(b) for b in blocks]
            skip_dims = [3, 64, 256, 512, 1024]
        else:
            raise NotImplementedError(conf.encoder)

        encoder = nn.ModuleList(encoder)
        return encoder, skip_dims

    def _init(self, conf):
        # Encoder
        self.encoder, skip_dims = self.build_encoder(conf)
        self.add_extra_input()  # add for pose enbedding

        # Decoder
        if conf.decoder is not None:
            assert len(conf.decoder) == (len(skip_dims) - 1)
            Block = checkpointed(DecoderBlock, do=conf.checkpointed)
            norm = eval(conf.decoder_norm) if conf.decoder_norm else None

            previous = skip_dims[-1]
            decoder = []
            for out, skip in zip(conf.decoder, skip_dims[:-1][::-1]):
                decoder.append(Block(previous, skip, out, norm=norm))
                previous = out
            self.decoder = nn.ModuleList(decoder)

        # Adaptation layers
        adaptation = []
        if conf.compute_uncertainty:
            uncertainty = []
        if updown_fusion:
            fuse_net= []
        for idx, i in enumerate(conf.output_scales):
            if conf.decoder is None or i == (len(self.encoder) - 1):
                input_ = skip_dims[i]
            else:
                input_ = conf.decoder[-1-i]

            if updown_fusion == 1:
                fuse_net.append(Fusion_topdown(input_))
            elif updown_fusion == 2:
                fuse_net.append(WeightGenerateBlock(input_ + 3, 1))
            elif updown_fusion == 3:
                fuse_net.append(WeightGenerateBlock(input_ + 3, 2))

            # out_dim can be an int (same for all scales) or a list (per scale)
            dim = conf.output_dim
            if not isinstance(dim, int):
                dim = dim[idx]

            block = AdaptationBlock(input_, dim)
            adaptation.append(block)
            if conf.compute_uncertainty:
                if two_confidence:
                    uncertainty.append(AdaptationBlock(input_, 2))
                else:
                    uncertainty.append(AdaptationBlock(input_, 1))
        self.adaptation = nn.ModuleList(adaptation)

        self.scales = [2**s for s in conf.output_scales]
        if conf.compute_uncertainty:
            self.uncertainty = nn.ModuleList(uncertainty)

        if updown_fusion:
            self.topdown_fusion = nn.ModuleList(fuse_net)

    def _forward(self, data):
        image = data['image']
        # mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        # image = (image - mean[:, None, None]) / std[:, None, None]

        # embedding height & distance & angle
        b, _, h, w = image.shape
        device = image.device
        vv, uu = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        uv = torch.stack([uu, vv], dim=-1)
        uv = uv[None, :, :, :].repeat(b, 1, 1, 1)  # shape = [b, h, w, 2]
        p3d_c = data['camera'].image2world(uv)  # [b, h, w, 3]
        if data['type'] == 'grd':
            p3d_q = torch.einsum('bij,bhwj->...bhwi', data['T_w2cam'].inv().R, p3d_c)  # query world coordinate
            height = p3d_q[..., 2]
            angle = p3d_q[..., 0] / torch.sqrt(p3d_q[..., 0] ** 2 + p3d_q[..., 1] ** 2)  # cos -1~1
            p3d_ongrd_q = camera_to_onground(p3d_c, data['T_w2cam'], data['camera_h'], data['normal'])
            dis = torch.sqrt((p3d_ongrd_q[..., 0]) ** 2 + (p3d_ongrd_q[..., 1]) ** 2) / max_dis
            dis = torch.where(dis > 1.2, torch.tensor(-1.).to(dis), dis)
            #dis = torch.where(torch.logical_or(dis>1.2, ~valid), torch.tensor(-1.).to(dis), dis) # dis/max_dis, igonore far than max_dis
        else:
            p3d_q = torch.einsum('bij,bhwj->...bhwi', data['q2r'].inv().R, p3d_c)  # query world coordinate
            angle = p3d_q[..., 0] / torch.sqrt(p3d_q[..., 0] ** 2 + p3d_q[..., 1] ** 2)  # cos -1~1
            dis = torch.sqrt(p3d_q[..., 0] ** 2 + p3d_q[..., 1] ** 2)/max_dis
            height = -0.6 * torch.ones_like(p3d_q[..., 2]) # all -0.6 as max height

        if debug_pe:
            plt.imshow(image[0].permute(1, 2, 0).detach().cpu())
            plt.axis("off")
            plt.margins(0, 0)
            plt.show()
            for img in (angle[0], dis[0], height[0]):
                # fig = plt.figure(figsize=plt.figaspect(1.))
                # ax1 = fig.add_subplot(1, 1, 1)
                # ax1.axis("off")
                # im1 = ax1.imshow(img, vmin=-1, vmax=1, cmap='jet', aspect='auto')
                # divider = make_axes_locatable(ax1)
                # cax = divider.append_axes('right', size='5%', pad=0.1)
                # fig.colorbar(im1, cax=cax, orientation='vertical')
                # plt.show()
                plot_images([img.detach().cpu()], cmaps=mpl.cm.gnuplot2, dpi=50)
                axes = plt.gcf().axes
                axes[0].imshow(image[0].permute(1,2,0).detach().cpu(), alpha=0.2, extent=axes[0].images[0]._extent)
                plt.show()

        extr = torch.stack([angle, height, dis], dim=1).to(device)  # shape = [b, 3, h, w]
        image = torch.cat([image, extr], dim=1) # shape = [b, 6, h, w]

        skip_features = []
        features = image
        for block in self.encoder:
            features = block(features)
            skip_features.append(features)


        if self.conf.decoder:
            pre_features = [skip_features[-1]]
            for block, skip in zip(self.decoder, skip_features[:-1][::-1]):
                pre_features.append(block(pre_features[-1], skip))
            pre_features = pre_features[::-1]  # fine to coarse
        else:
            pre_features = skip_features

        # up-down feature fusion
        if updown_fusion and data['type'] == 'grd':
            if updown_fusion == 3:
                raw2alpha = lambda raw, act_fn=F.relu: 1. - torch.exp(-act_fn(raw))  # dists is fix to 1
            for layer, i in zip(self.topdown_fusion, self.conf.output_scales):
                # pose embedding, query world coordinate
                if extr.shape[-2:] != pre_features[i].shape[-2:]:
                    pe = F.interpolate(extr, size=pre_features[i].shape[-2:])
                else:
                    pe = extr

                # # embedding uv
                #vv, uu = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                #pe = torch.stack([uu, vv], dim=0)  # shape = [2, h, w]
                # pe = pe[None, :, :, :].repeat(b, 1, 1, 1)

                # transformer fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                if updown_fusion == 1:
                    v_start_ratio = data['camera'].c[0, 1] / data['camera'].size[0, 1] + 0.05
                    v_start = int(h * v_start_ratio)
                    if visualize: # visualize
                        fused_feature = layer(pre_features[i], pe, v_start, visualize, image) #[b,c,h-v_start,w]
                    else:
                        fused_feature = layer(pre_features[i], pe, v_start)#[b,c,h-v_start,w]
                    fused_feature = torch.cat([pre_features[i][:, :, :v_start], fused_feature], dim=2)
                # end transformer fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif updown_fusion == 2:
                    # linear fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # generate fusion weight
                    f_pe = torch.cat([pre_features[i], pe], dim=1)  # [b,c+3,h,w]
                    fuse_weight = layer(f_pe)  # [b,1,h,w]

                    fused_feature = fuse_weight * pre_features[i]
                    sum_feature = torch.cumsum(fused_feature, dim=2) # [b,c,h,w]
                    sum_weight = torch.cumsum(fuse_weight, dim=2) # [b,1,h,w]
                    fused_feature = sum_feature/(sum_weight+1e-7)
                    # end linear fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif updown_fusion == 3:
                    #nerf fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    f_pe = torch.cat([pre_features[i], pe], dim=1)  # [b,c+3,h,w]
                    out = layer(f_pe)  # [b,2,h,w]
                    # beta = self.beta_net(density)  # [b,1,h,w]
                    density = out[:,:1]  # [b,1,h,w]
                    beta = torch.sigmoid(-out[:,1:]) #0~1

                    # visualize density & beta
                    if visualize:
                        density_img = density[0].permute(1, 2, 0).cpu().detach()
                        deta_img = beta[0].permute(1, 2, 0).cpu().detach()
                        ori_img = image[0].permute(1, 2, 0).cpu()
                        plot_images([density_img, deta_img], cmaps=mpl.cm.gnuplot2, dpi=50)
                        add_text(0, f'Level {i}')
                        axes = plt.gcf().axes
                        axes[0].imshow(ori_img, alpha=0.2, extent=axes[0].images[0]._extent)
                        axes[1].imshow(ori_img, alpha=0.2, extent=axes[1].images[0]._extent)
                        plt.show()

                    alpha = raw2alpha(density)  # [b,1,h(samples), w(rays)]
                    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[:,:,:1]), 1. - alpha + 1e-10], 2),
                                                    -1)[:, :, :-1]
                    fused_feature = weights * pre_features[i]
                    sum_feature = torch.cumsum(fused_feature, dim=2)  # [b,c,h,w]
                    sum_weight = torch.cumsum(weights, dim=2)  # [b,1,h,w]
                    fused_feature = sum_feature / (sum_weight + 1e-7)
                    fused_feature = beta * pre_features[i] + (1-beta) * fused_feature
                    #end nerf fusion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                pre_features[i] = fused_feature

        out_features = []
        for adapt, i in zip(self.adaptation, self.conf.output_scales):
            out_features.append(adapt(pre_features[i]))
        pred = {'feature_maps': out_features}

        if self.conf.compute_uncertainty:
            confidences = []
            for layer, i in zip(self.uncertainty, self.conf.output_scales):
                unc = layer(pre_features[i])
                conf = torch.sigmoid(-unc)
                confidences.append(conf)
            pred['confidences'] = confidences

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

    # def add_sat_unet(self):
    #     self.sat_encoder = deepcopy(self.encoder)
    #     self.sat_decoder = deepcopy(self.decoder)
    #     self.sat_adaptation = deepcopy(self.adaptation)
    #
    # def add_sat_branch(self):
    #     # high_encoder = self.encoder[2:]
    #     # sat_low_decoder = deepcopy(self.encoder[:2])
    #     # only not share weight in last layer of encoder
    #     high_encoder = deepcopy(self.encoder[self.sat_start_layer:])
    #     # sat_low_encoder = self.encoder[:-1]
    #     blocks = []
    #     # for block in sat_low_encoder:
    #     #     blocks.append(block)
    #     for block in high_encoder:
    #         blocks.append(block)
    #     self.sat_encoder = nn.ModuleList(blocks)

    def add_grd_confidence(self):
        for old_uncertainty in self.uncertainty:
            in_ch = old_uncertainty[0].weight.shape[1]
            # add 1ch
            # new_uncertainty = nn.Conv2d(in_ch, 2, kernel_size=1).to(old_uncertainty[0].weight)
            # new_weight = torch.cat([old_uncertainty[0].weight.clone(), new_uncertainty.weight[1:].clone()], dim=0)
            # remove 1ch
            new_uncertainty = nn.Conv2d(in_ch, 1, kernel_size=1).to(old_uncertainty[0].weight)
            new_weight = old_uncertainty[0].weight[:1].clone()
            new_uncertainty.weight = nn.Parameter(new_weight)
            old_uncertainty[0] = new_uncertainty

    # fix parameter for feature extractor, only need gradiant of confidence
    def fix_parameter_of_feature(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.adaptation.parameters():
            param.requires_grad = False
        if updown_fusion:
            for param in self.topdown_fusion.parameters():
                param.requires_grad = False
        # if updown_fusion == 3:
        #     for param in self.beta_net.parameters():
        #         param.requires_grad = False

    def add_weight_generator(self):
        fuse_net = []
        for input_ in (32, 64, 512):
            fuse_net.append(Fusion_topdown(input_)) # trans
            #fuse_net.append(WeightGenerateBlock(input_+3, 1)) # linear
            #fuse_net.append(WeightGenerateBlock(input_ + 3, 2)) # nerf
        self.topdown_fusion = nn.ModuleList(fuse_net).cuda()

    def add_extra_input(self):
        layer = self.encoder[0][0]
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(6,64,kernel_size=3,padding=1).to(layer.weight)
        new_weight = torch.cat([layer.weight.clone(), new_layer.weight[:,3:].clone()], dim=1)
        new_layer.weight = nn.Parameter(new_weight)
        self.encoder[0][0] = new_layer

        # for old_topdown_fusion, input_ in zip(self.topdown_fusion, (32, 64, 512)):
        #     old_topdown_fusion.add_extra_embed(input_)
