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
        self.mesh_render = Texture_Renderer(
            512, flame_path=FLAME_MODEL_PATH, device=self._device
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
        translation[..., :2] = (pred_lmks.mean(dim=1)[..., :2] - gt_lmks.mean(dim=1)[..., :2]) * 2 / self.image_size
        return rotation_matrix, translation

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
        light_params = torch.nn.Parameter(torch.rand(1, 9, 3).to(self._device))
        params = [
            {'params': [texture_params], 'lr': 0.005, 'name': ['tex']},
            {'params': [light_params], 'lr': 0.01, 'name': ['sh']},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps, gamma=0.1)
        tqdm_queue = tqdm(range(steps), desc='', leave=True, miniters=10)
        for k in tqdm_queue:
            albedos = self.flame_texture(texture_params)
            pred_images, mask_all, mask_face = self.mesh_render(flame_verts, albedos, light_params, cameras)
            losses = {}
            losses['face'] = pixel_loss(pred_images, frames, mask=mask_face) * 350
            losses['head'] = pixel_loss(pred_images, frames, mask=mask_all) * 350
            all_loss = losses['face'] + losses['head']
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            scheduler.step()
            loss = all_loss.item()
            tqdm_queue.set_description(f'Loss for tex {loss:.4f}')
        results = {
            'texture_params': texture_params.detach().cpu(), 'light_params': light_params.detach().cpu(), 
        }
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

    def render(self, frames, lightning_params, shape_code):
        # flame
        flame_shape = shape_code.float().to(self._device)[None].repeat(frames.shape[0], 1)
        flame_exp = lightning_params['expression'].float().to(self._device)
        flame_pose = lightning_params['flame_pose'].clone().float().to(self._device)
        transform_matrix = lightning_params['transform_matrix'].clone().float().to(self._device)
        flame_pose[..., :3] *= 0
        # run
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=flame_shape, expression_params=flame_exp, pose_params=flame_pose
        )
        flame_verts = flame_verts * self.flame_scale
        pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        # render
        cameras_kwargs = self._build_cameras_kwars(frames.shape[0])
        rotation = transform_matrix[:, :3, :3]
        translation = transform_matrix[:, :3, 3]
        cameras = PerspectiveCameras(R=rotation, T=translation, **cameras_kwargs)
        images, alpha_images = self.mesh_render(flame_verts, cameras)
        images = images.cpu(); alpha_images = alpha_images.cpu(); alpha_images = alpha_images.repeat(1, 3, 1, 1)
        pred_lmk_68 = cameras.transform_points_screen(pred_lmk_68)[..., :2]
        pred_lmk_dense = cameras.transform_points_screen(pred_lmk_dense)[..., :2]
        # gather
        vis_images = []
        for idx, frame in enumerate(frames):
            vis_i = frame.clone().float()
            vis_i[alpha_images[idx]>0.5] *= 0.3
            vis_i[alpha_images[idx]>0.5] += (images[idx, alpha_images[idx]>0.5] * 0.7)
            vis_i = torchvision.utils.draw_keypoints(vis_i.to(torch.uint8), pred_lmk_dense[idx:idx+1], colors="white", radius=1.5)
            vis_i = torchvision.utils.draw_keypoints(vis_i, pred_lmk_68[idx:idx+1], colors="red", radius=1.5)
            # vis_i = torchvision.utils.draw_bounding_boxes(vis_i, mini_batch['bbox'][idx:idx+1])
            # vis_image = torchvision.utils.make_grid([vis_i, mp_images[idx], frame], nrow=3)
            vis_image = torchvision.utils.make_grid([frame, images[idx], vis_i], nrow=4)
            # torchvision.utils.save_image(vis_image/255, './debug.jpg')
            vis_images.append(vis_image)
        return vis_images


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
