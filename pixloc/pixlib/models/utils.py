import torch


def masked_mean(x, mask, dim):
    mask = mask.float()
    return (mask * x).sum(dim) / mask.sum(dim).clamp(min=1)


def checkpointed(cls, do=True):
    '''Adapted from the DISK implementation of Michał Tyszkiewicz.'''
    assert issubclass(cls, torch.nn.Module)

    class Checkpointed(cls):
        def forward(self, *args, **kwargs):
            super_fwd = super(Checkpointed, self).forward
            if any((torch.is_tensor(a) and a.requires_grad) for a in args):
                return torch.utils.checkpoint.checkpoint(
                        super_fwd, *args, **kwargs)
            else:
                return super_fwd(*args, **kwargs)

    return Checkpointed if do else cls

# shan add for key points extraction, from super point
def merge_confidence_map(confidence, number):
    """extrac key ponts from confidence map.
    Args:
        confidence: torch.Tensor with size (B,C,H,W).
        number: number of confidence map
    Returns:
        merged confidence map: torch.Tensor with size (B,H,W).
    """
    B,C,H,W = confidence[0].size()
    for level in range(len(confidence)):
        if number == 2:
            c_cur = confidence[level][:,:1]*confidence[level][:,1:]
        else:
            c_cur = confidence[level][:,:1]
        if level > 0:
            c_cur = torch.nn.functional.interpolate(c_cur, size=(H,W), mode='bilinear')
            max, _ = torch.max(c_cur.flatten(-2), dim=-1)
            c_cur = c_cur / (max[:,:,None,None] + 1e-8)
            #c_cur = torch.nn.functional.normalize(c_cur.flatten(-2), p=float('inf'), dim=-1)  # normalize in 2d Plane #[b,c,H,W]
            #c_cur = c_cur.view(B, 1, H, W)
            c_last = 0.8*c_last + c_cur
        else:
            max, _ = torch.max(c_cur.flatten(-2), dim=-1)
            c_cur = c_cur / (max[:,:,None,None] + 1e-8)
            #c_cur = torch.nn.functional.normalize(c_cur.flatten(-2), p=float('inf'), dim=-1)  # normalize in 2d Plane #[b,c,H,W]
            #c_cur = c_cur.view(B, 1, H, W)
            c_last = c_cur
    return c_last

# shan add for key points extraction, from super point
def extract_keypoints(confidence, topk=256, start_ratio=0.65):
    """extrac key ponts from confidence map.
    Args:
        confidence: torch.Tensor with size (B,C,H,W).
        topk: extract topk points each confidence map
        start_ratio: extract close to ground part (start_ratio*H:)
    Returns:
        A torch.Tensor of index where the key points are.
    """
    assert (start_ratio < 1 and start_ratio >= 0)


    w_end = -1
    h_end = -1
    radius = 4
    if confidence.size(3) > 1224:
        # fix here, need move to config ---------------
        # kitti 375×1242, 370×1224,374×1238, and376×1241 -> 384, 1248
        start_ratio = 0.55
        radius = 6
        #---------------------------------------------

    #  only extract close to ground part (start_H:)
    start_H = int(confidence.size(2)*start_ratio)
    confidence = confidence[:,:,start_H:h_end,:w_end].detach().clone()

    # fast Non-maximum suppression to remove nearby points
    def max_pool(x):
        return torch.nn.functional.max_pool2d(x, kernel_size=radius*2+1, stride=1, padding=radius)

    max_mask = (confidence == max_pool(confidence))
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_confidence = torch.where(supp_mask, torch.zeros_like(confidence), confidence)
        new_max_mask = (supp_confidence == max_pool(supp_confidence))
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    confidence = torch.where(max_mask, confidence, torch.zeros_like(confidence))

    # remove borders
    border = radius
    confidence[:, :, :border] = 0.
    confidence[:, :, -border:] = 0.
    confidence[:, :, :, :border] = 0.
    confidence[:, :, :, -border:] = 0.

    # confidence topk
    _, index = confidence.flatten(1).topk(topk, dim=1, largest=True, sorted=True)

    index_v = torch.div(index, confidence.size(-1) , rounding_mode='trunc')
    index_u = index % confidence.size(-1)
    # back to original index
    index_v += start_H

    return torch.cat([index_u.unsqueeze(-1),index_v.unsqueeze(-1)],dim=-1)
