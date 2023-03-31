import os
import math
import torch
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix

from model.FLAME.FLAME import FLAME_MP
from utils.renderer import Mesh_Renderer

FLAME_MODEL_PATH = './assets/FLAME'

class Lightning_Engine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device

    def init_model(self, batch_size, camera_params, image_size=512):
        print('Initializing lightning models...')
        # camera params
        self.flame_scale = camera_params['flame_scale']
        self.focal_length = camera_params['focal_length'].to(self._device)
        self.principal_point = camera_params['principal_point'].to(self._device).repeat(batch_size, 1)
        # build camera
        screen_size = torch.tensor([image_size, image_size], device=self._device).float()[None].repeat(batch_size, 1)
        self.cameras = PerspectiveCameras(
            principal_point=self.principal_point, focal_length=self.focal_length, 
            image_size=screen_size, device=self._device,
        )
        # build flame
        self.flame_model = FLAME_MP(FLAME_MODEL_PATH, 100, 50).to(self._device)
        self.mesh_render = Mesh_Renderer(
            512, obj_filename=os.path.join(FLAME_MODEL_PATH, 'head_template_mesh.obj'), device=self._device
        )
        print('Done.')

    def flame_to_camera(self, flame_pose, pred_lmks, gt_lmks):
        # rotation
        flame_pose[:, 1] += math.pi
        flame_pose[:, 0] *= -1
        rotation_matrix = euler_angles_to_matrix(flame_pose[..., :3], 'XYZ').permute(0, 2, 1)
        # translation
        translation = rotation_matrix.new_zeros(rotation_matrix.shape[0], 3)
        translation[..., 2] = self.focal_length 
        self.cameras.get_camera_center(R=rotation_matrix, T=translation)
        pred_lmks = self.cameras.transform_points_screen(pred_lmks)
        translation[..., :2] = (pred_lmks.mean(dim=1)[..., :2] - gt_lmks.mean(dim=1)[..., :2]) * 2
        return rotation_matrix, translation

    def lightning_optimize_camera(self, emoca_params, landmarks):
        flame_shape = emoca_params['shape'].float().to(self._device)
        flame_exp = emoca_params['exp'].float().to(self._device)
        flame_pose = emoca_params['pose'].clone().float().to(self._device)
        flame_pose[..., :3] *= 0
        flame_verts, pred_lmk_68, pred_lmk_media = self.flame_model(
            shape_params=flame_shape, expression_params=flame_exp, pose_params=flame_pose
        )
        flame_verts = flame_verts * self.flame_scale
        pred_lmk_68, pred_lmk_media = pred_lmk_68 * self.flame_scale, pred_lmk_media * self.flame_scale
        rotation_matrix, translation = self.flame_to_camera(
            emoca_params['pose'].float(), pred_lmk_media, landmarks['lmks_dense']
        )
        self.cameras.get_camera_center(R=rotation_matrix, T=translation)
        images, _ = self.mesh_render(flame_verts, self.cameras)
        print(images.shape)
        raise Exception

