import os
import math
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

from model.FLAME.FLAME import FLAME_MP

class Lightning_Engine:
    def __init__(self, flame_model_path, device='cuda', lazy_init=True):
        self._device = device
        self._flame_model_path = flame_model_path

    def init_model(self, camera_params, image_size=512):
        print('Initializing lightning models...')
        # camera params
        self.image_size = image_size
        self.flame_scale = camera_params['flame_scale']
        self.focal_length = camera_params['focal_length'].to(self._device)
        self.principal_point = camera_params['principal_point'].to(self._device)
        # build flame
        self.flame_model = FLAME_MP(self._flame_model_path, 100, 50).to(self._device)
        print('Done.')

    def _build_cameras_kwargs(self, batch_size):
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': self.principal_point.repeat(batch_size, 1), 'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def flame_to_camera(self, cameras, flame_pose, pred_lmks, gt_lmks):
        # rotation
        flame_pose[:, 1] += math.pi
        flame_pose[:, 0] *= -1
        rotation_matrix = euler_angles_to_matrix(flame_pose[..., :3], 'XYZ').permute(0, 2, 1)
        # translation
        translation = rotation_matrix.new_zeros(rotation_matrix.shape[0], 3)
        translation[..., 2] = self.focal_length
        cameras.get_camera_center(R=rotation_matrix, T=translation)
        pred_lmks = cameras.transform_points_screen(pred_lmks)
        translation[..., :2] = (pred_lmks.mean(dim=1)[..., :2] - gt_lmks.mean(dim=1)[..., :2]) * 2 / self.image_size
        return rotation_matrix, translation

    def lightning_optimize(self, batch_data, steps=200):
        # ['frame_names', 'frames', 'emoca', 'lmks', 'shape']
        batch_size = len(batch_data['frame_names'])
        flame_pose = batch_data['emoca']['pose'].clone()
        batch_data['emoca']['pose'][..., :3] *= 0
        batch_data['frames'] = batch_data['frames'] / 255.0
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        # flame params
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=batch_data['shape_code'][None].expand(batch_size, -1), 
            expression_params=batch_data['emoca']['exp'],
            pose_params=batch_data['emoca']['pose']
        )
        flame_verts = flame_verts * self.flame_scale
        pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        # build params
        cameras = PerspectiveCameras(**cameras_kwargs)
        rotation, translation = self.flame_to_camera(
            cameras, flame_pose, pred_lmk_68, batch_data['emoca']['lmks']
        )
        translation = torch.nn.Parameter(translation)
        rotation = torch.nn.Parameter(matrix_to_rotation_6d(rotation))
        params = [{'params': [rotation, translation], 'lr': 0.02}]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        # run
        for idx in range(steps):
            points_68 = cameras.transform_points_screen(pred_lmk_68, R=rotation_6d_to_matrix(rotation), T=translation)[..., :2]
            points_dense = cameras.transform_points_screen(pred_lmk_dense, R=rotation_6d_to_matrix(rotation), T=translation)[..., :2]
            loss_lmk_68 = lmk_loss(points_68, batch_data['emoca']['lmks'], self.image_size)
            loss_lmk_dense = lmk_loss(points_dense, batch_data['emoca']['lmks_dense'][:, self.flame_model.mediapipe_idx], self.image_size)
            all_loss = (loss_lmk_68 + loss_lmk_dense) * 65
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
        # gather results
        lightning_results = {}
        transform_matrix = torch.cat([rotation_6d_to_matrix(rotation), translation[:, :, None]], dim=-1)
        for idx, name in enumerate(batch_data['frame_names']):
            lightning_results[name] = {
                'flame_pose': batch_data['emoca']['pose'][idx].half().cpu(),
                'expression': batch_data['emoca']['exp'][idx].half().cpu(),
                'face_box': batch_data['emoca']['face_box'][idx].half().cpu(),
                'transform_matrix': transform_matrix[idx].half().cpu()
            }
        return lightning_results


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()
