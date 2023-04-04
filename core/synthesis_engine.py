import os
import math
import torch
import numpy as np
import torchvision
from tqdm.rich import tqdm
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

from utils.renderer import Texture_Renderer
from model.FLAME.FLAME import FLAME_MP, FLAME_Tex

FLAME_MODEL_PATH = './assets/FLAME'

class Synthesis_Engine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device

    def init_model(self, camera_params, image_size=512):
        print('Initializing synthesis models...')
        # camera params
        self.image_size = image_size
        self.flame_scale = camera_params['flame_scale']
        self.focal_length = camera_params['focal_length'].to(self._device)
        self.principal_point = camera_params['principal_point'].to(self._device)
        # build flame
        self.flame_model = FLAME_MP(FLAME_MODEL_PATH, 100, 50).to(self._device)
        self.flame_texture = FLAME_Tex(FLAME_MODEL_PATH, image_size=512).to(self._device)
        self.flame_face_mask = self.flame_texture.masks.face
        self.mesh_render = Texture_Renderer(
            512, flame_path=FLAME_MODEL_PATH, flame_mask=self.flame_face_mask, device=self._device
        )
        print('Done.')

    def _build_cameras_kwars(self, batch_size):
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': self.principal_point.repeat(batch_size, 1), 'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def optimize_camera(self, rotation, translation, pred_lmks_68, pred_lmks_dense, gt_lmks_68, gt_lmks_dense, steps=200):
        # build trainable params
        camera_T = torch.nn.Parameter(translation, requires_grad=True)
        camera_R = torch.nn.Parameter(matrix_to_rotation_6d(rotation), requires_grad=True)
        # optimizer
        params = [{'params': [camera_R, camera_T], 'lr': 0.05}]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        for k in range(steps):
            points_68 = self.cameras.transform_points_screen(
                pred_lmks_68, R=rotation_6d_to_matrix(camera_R), T=camera_T
            )[..., :2]
            points_dense = self.cameras.transform_points_screen(
                pred_lmks_dense, R=rotation_6d_to_matrix(camera_R), T=camera_T
            )[..., :2]
            losses = {}
            losses['lmk_68'] = lmk_loss(points_68, gt_lmks_68, self.image_size) * 65
            losses['lmk_dense'] = lmk_loss(points_dense, gt_lmks_dense[:, self.flame_model.mediapipe_idx], self.image_size) * 65
            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
            loss = all_loss.item()
        return rotation_6d_to_matrix(camera_R).detach(), camera_T.detach()

    def optimize_texture(self, frames, lightning_params, steps=1000):
        # build camera
        cameras_kwargs = self._build_cameras_kwars(frames.shape[0])
        transform_matrix = lightning_params['transform_matrix'].float().to(self._device)
        rotation, translation = transform_matrix[:, :3, :3], transform_matrix[..., :3, 3]
        cameras = PerspectiveCameras(R=rotation, T=translation, **cameras_kwargs)
        # flame params
        flame_shape = lightning_params['shape'].float().to(self._device)
        flame_exp = lightning_params['expression'].float().to(self._device)
        flame_pose = lightning_params['flame_pose'].clone().float().to(self._device)
        flame_pose[..., :3] *= 0
        flame_verts, _, _ = self.flame_model(
            shape_params=flame_shape, expression_params=flame_exp, pose_params=flame_pose
        )
        flame_verts = flame_verts * self.flame_scale
        frames = frames.to(self._device) / 255.0
        # optimize
        texture_params = torch.nn.Parameter(torch.rand(1, 140).to(self._device))
        params = [
            {'params': [texture_params], 'lr': 0.005, 'name': ['tex']},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        tqdm_queue = tqdm(range(steps), desc='', leave=True, miniters=10)
        for k in tqdm_queue:
            albedos = self.flame_texture(texture_params)
            pred_images, masks_all, masks_face = self.mesh_render(flame_verts, albedos, cameras)
            loss_head = pixel_loss(pred_images, frames, mask=masks_all)
            loss_face = pixel_loss(pred_images, frames, mask=masks_face)
            all_loss = (loss_head + loss_face) * 350
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
            tqdm_queue.set_description(f'Loss for tex {all_loss.item():.4f}')
        results = {'texture_params': texture_params.detach().cpu()}
        return results, torch.cat([frames[:4], pred_images[:4]]).cpu()

    def synthesis_optimize(self, frame_names, frames, lightning_params, landmarks, steps=500):
        # build params
        cameras_kwargs = self._build_cameras_kwars(frames.shape[0])
        texture_params = torch.nn.Parameter(lightning_params['texture'].to(self._device), requires_grad=False)
        light_params = torch.nn.Parameter(lightning_params['light'].to(self._device), requires_grad=False)
        transform_matrix = lightning_params['transform_matrix'].float().to(self._device)
        rotation, translation = transform_matrix[:, :3, :3], transform_matrix[..., :3, 3]
        rotation = torch.nn.Parameter(matrix_to_rotation_6d(rotation))
        translation = torch.nn.Parameter(translation)
        params = [
            # {'params': [texture_params], 'lr': 0.005, 'name': ['tex']},
            # {'params': [light_params], 'lr': 0.01, 'name': ['sh']},
            {'params': [rotation], 'lr': 0.005, 'name': ['r']},
            {'params': [translation], 'lr': 0.005, 'name': ['t']},
        ]
        # flame params
        flame_shape = lightning_params['shape'].float().to(self._device)
        flame_exp = lightning_params['expression'].float().to(self._device)
        flame_pose = lightning_params['flame_pose'].clone().float().to(self._device)
        flame_pose[..., :3] *= 0
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=flame_shape, expression_params=flame_exp, pose_params=flame_pose
        )
        flame_verts = flame_verts * self.flame_scale
        # pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        frames = frames.to(self._device) / 255.0
        
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        for idx in range(steps):
            self.cameras = PerspectiveCameras(R=rotation_6d_to_matrix(rotation), T=translation, **cameras_kwargs)
            albedos = self.flame_texture(texture_params)
            pred_images, mask_all, mask_face = self.mesh_render(flame_verts, albedos, light_params, self.cameras)
            losses = {}
            losses['face'] = pixel_loss(pred_images, frames, mask=mask_face) * 350
            losses['head'] = pixel_loss(pred_images, frames, mask=mask_all) * 350
            all_loss = losses['face'] + losses['head']
            optimizer.zero_grad()
            all_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            loss = all_loss.item()
            # print(rotation[0], translation[0])
            print(idx, loss)

        return synthesis_results


def lmk_loss(opt_lmks, target_lmks, image_size, lmk_mask=None):
    size = torch.tensor([1 / image_size, 1 / image_size], device=opt_lmks.device).float()[None, None, ...]
    diff = torch.pow(opt_lmks - target_lmks, 2)
    if lmk_mask is None:
        return (diff * size).mean()
    else:
        return (diff * size * lmk_mask).mean()

def pixel_loss(opt_img, target_img, mask=None):
    if mask is None:
        mask = torch.ones_like(opt_img).type_as(opt_img)
    n_pixels = torch.sum((mask[:, 0, ...] > 0).int()).detach().float()
    loss = (mask * (opt_img - target_img)).abs()
    loss = torch.sum(loss) / n_pixels
    return loss
