import math

import torch
import torchvision
from tqdm import tqdm
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from model.FLAME.FLAME import FLAME_MP

def optimize_camera(emoca_params, gt_landmarks, frames, image_size=512, steps=1000, device='cuda'):
    # build params
    batch_size = emoca_params['shape'].shape[0]
    for key in emoca_params:
        emoca_params[key] = emoca_params[key].to(device).float()
    for key in gt_landmarks:
        gt_landmarks[key] = gt_landmarks[key].to(device).float()
    # build flame
    flame = FLAME_MP(flame_path = './assets/FLAME', n_shape=100, n_exp=50).to(device)
    _, pred_lmk_68, pred_lmk_dense = flame(
        shape_params=emoca_params['shape'], expression_params=emoca_params['exp'], pose_params=emoca_params['pose']
    )
    flame_scale = (emoca_params['cam'][:, 0]*(emoca_params['crop_box'][:, 2]/image_size)).mean().item()
    # build camera
    screen_size = torch.tensor([image_size, image_size]).float().repeat(batch_size, 1).to(device)
    initial_focal_length = torch.tensor([[5000.0 / image_size]]).to(device)
    cameras = PerspectiveCameras(device=device, image_size=screen_size)
    # build initilization params
    R, T = look_at_view_transform(dist=initial_focal_length)
    R = R.to(device).repeat(batch_size, 1, 1); T = T.to(device).repeat(batch_size, 1)
    normed_lmks = cameras.transform_points_screen(
        pred_lmk_dense*flame_scale, R=R, T=T, focal_length=initial_focal_length
    )
    shifts = (normed_lmks.mean(dim=1)[..., :2] - gt_landmarks['lmks_dense'].mean(dim=1)) / image_size
    T[:, :2] = shifts * 2
    # build trainable params
    camera_T = torch.nn.Parameter(T, requires_grad=True)
    camera_R = torch.nn.Parameter(matrix_to_rotation_6d(R), requires_grad=True)
    focal_length = torch.nn.Parameter(initial_focal_length, requires_grad=True)
    principal_point = torch.nn.Parameter(torch.zeros(1, 2).to(device), requires_grad=True)
    # optimizer
    params = [{'params': [camera_R, camera_T, focal_length, principal_point], 'lr': 0.05}]
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)

    tqdm_queue = tqdm(range(steps), desc='', leave=True, miniters=100, ncols=120, colour='#95bb72')
    for k in tqdm_queue:
        points_68 = cameras.transform_points_screen(
            pred_lmk_68*flame_scale,
            R=rotation_6d_to_matrix(camera_R), T=camera_T, principal_point=principal_point, focal_length=focal_length
        )[..., :2]
        points_dense = cameras.transform_points_screen(
            pred_lmk_dense*flame_scale,
            R=rotation_6d_to_matrix(camera_R), T=camera_T, principal_point=principal_point, focal_length=focal_length
        )[..., :2]
        losses = {}
        losses['pp_reg'] = torch.sum(principal_point ** 2)
        losses['lmk68'] = lmk_loss(points_68, gt_landmarks['lmks'], image_size) * 65
        losses['lmkMP'] = lmk_loss(points_dense, gt_landmarks['lmks_dense'][:, flame.mediapipe_idx], image_size) * 65
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        scheduler.step()
        loss = all_loss.item()
        tqdm_queue.set_description(f'Loss for camera {loss:.4f}')
    # visualization
    visualization = []
    for idx, frame in enumerate(frames):
        vis_i = torchvision.utils.draw_keypoints(frame.to(torch.uint8), points_dense[idx:idx+1], colors="green", radius=1.5)
        vis_i = torchvision.utils.draw_keypoints(vis_i, points_68[idx:idx+1], colors="white", radius=1.5)
        # vis_i = torchvision.utils.draw_bounding_boxes(vis_i, emoca_params['crop_box'][idx:idx+1])
        visualization.append(vis_i.float())
    camera_params = {
        'focal_length': focal_length.detach().cpu()[0], 'principal_point': principal_point.detach().cpu()[0],
        'fov': 2 * torch.arctan(1 / focal_length.detach().cpu()[0]) / math.pi * 360, 'flame_scale': flame_scale
    }
    print(camera_params)
    return camera_params, torchvision.utils.make_grid(visualization[:8], nrow=4)


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()
