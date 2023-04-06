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

    def _build_cameras_kwargs(self, batch_size):
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': self.principal_point.repeat(batch_size, 1), 'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def optimize_texture(self, frames, lightning_params, steps=1000):
        # build camera
        cameras_kwargs = self._build_cameras_kwargs(frames.shape[0])
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

    def synthesis_optimize(self, batch_data, steps=30):
        # ['frame_names', 'frames', 'lightning', 'landmarks', 'texture_code', 'shape_code']
        batch_size = len(batch_data['frame_names'])
        batch_data['lightning']['flame_pose'][..., :3] *= 0
        batch_data['frames'] = batch_data['frames'] / 255.0
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        # build flame params
        # flame params
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=batch_data['shape_code'][None].expand(batch_size, -1), 
            expression_params=batch_data['lightning']['expression'],
            pose_params=batch_data['lightning']['flame_pose']
        )
        flame_verts = flame_verts * self.flame_scale
        # pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        # build params
        transform_matrix = batch_data['lightning']['transform_matrix']
        rotation, translation = transform_matrix[:, :3, :3], transform_matrix[..., :3, 3]
        translation = torch.nn.Parameter(translation)
        rotation = torch.nn.Parameter(matrix_to_rotation_6d(rotation))
        texture_params = torch.nn.Parameter(batch_data['texture_code'])
        params = [
            # {'params': [texture_params], 'lr': 0.005, 'name': ['tex']},
            {'params': [rotation], 'lr': 0.005, 'name': ['r']},
            {'params': [translation], 'lr': 0.005, 'name': ['t']},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        # run        
        for idx in range(steps):
            self.cameras = PerspectiveCameras(
                R=rotation_6d_to_matrix(rotation), T=translation, **cameras_kwargs
            )
            albedos = self.flame_texture(texture_params)
            pred_images, mask_all, mask_face = self.mesh_render(flame_verts, albedos, self.cameras)
            losses = {}
            losses['face'] = pixel_loss(pred_images, batch_data['frames'], mask=mask_face) * 350
            losses['head'] = pixel_loss(pred_images, batch_data['frames'], mask=mask_all) * 350
            all_loss = losses['face'] + losses['head']
            optimizer.zero_grad()
            all_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            loss = all_loss.item()
            # ### DEBUG
            # print(idx, rotation[0], translation[0], loss)
            # torchvision.utils.save_image(
            #     torch.cat([batch_data['frames'][:4], pred_images[:4]]).cpu(),
            #     './debug.jpg', nrow=4
            # )
        # gather results
        synthesis_results = {}
        transform_matrix = torch.cat(
            [rotation_6d_to_matrix(rotation), translation[:, :, None]], dim=-1
        )
        
        for idx, name in enumerate(batch_data['frame_names']):
            synthesis_results[name] = {
                'flame_pose': batch_data['lightning']['flame_pose'][idx].half().cpu(),
                'expression': batch_data['lightning']['expression'][idx].half().cpu(),
                'bbox': batch_data['lightning']['bbox'][idx].half().cpu(),
                'transform_matrix': transform_matrix[idx].half().cpu()
            }
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
