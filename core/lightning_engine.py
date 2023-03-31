import os
import math
import torch
import numpy as np
import torchvision
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix

from model.FLAME.FLAME import FLAME_MP
from utils.renderer import Mesh_Renderer

FLAME_MODEL_PATH = './assets/FLAME'

class Lightning_Engine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device

    def init_model(self, camera_params, image_size=512):
        print('Initializing lightning models...')
        # camera params
        self.image_size = image_size
        self.flame_scale = camera_params['flame_scale']
        self.focal_length = camera_params['focal_length'].to(self._device)
        self.principal_point = camera_params['principal_point'].to(self._device)
        # build flame
        self.flame_model = FLAME_MP(FLAME_MODEL_PATH, 100, 50).to(self._device)
        self.mesh_render = Mesh_Renderer(
            512, obj_filename=os.path.join(FLAME_MODEL_PATH, 'head_template_mesh.obj'), device=self._device
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

    def lightning_optimize_camera(self, frame_names, emoca_params, landmarks, frames=None):
        # build camera
        cameras_kwargs = self._build_cameras_kwars(len(frame_names))
        self.cameras = PerspectiveCameras(**cameras_kwargs)
        # params
        landmarks['lmks'] = landmarks['lmks'].to(self._device).float()
        landmarks['lmks_dense'] = landmarks['lmks_dense'].to(self._device).float()
        emoca_params['exp'] = emoca_params['exp'].to(self._device).float()
        emoca_params['pose'] = emoca_params['pose'].to(self._device).float()
        flame_shape = emoca_params['shape'].float().to(self._device)
        flame_exp = emoca_params['exp'].float().to(self._device)
        flame_pose = emoca_params['pose'].clone().float().to(self._device)
        flame_pose[..., :3] *= 0
        # run
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=flame_shape, expression_params=flame_exp, pose_params=flame_pose
        )
        flame_verts = flame_verts * self.flame_scale
        pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        # get params
        rotation, translation = self.flame_to_camera(emoca_params['pose'], pred_lmk_68, landmarks['lmks'])
        # optimize
        rotation, translation = self.optimize_camera(
            rotation, translation, 
            pred_lmk_68, pred_lmk_dense, landmarks['lmks'], landmarks['lmks_dense']
        )
        # gather results
        lightning_results = {}
        transform_matrix = torch.cat([rotation, translation[:, :, None]], dim=-1)
        for idx, name in enumerate(frame_names):
            lightning_results[name] = {
                'flame_pose': emoca_params['pose'][idx].half().cpu(),
                'expression': emoca_params['exp'][idx].half().cpu(),
                'bbox': emoca_params['crop_box'][idx].half().cpu(),
                'transform_matrix': transform_matrix[idx].half().cpu()
            }
        # ### DEBUG
        # cameras_kwargs = self._build_cameras_kwars(len(frame_names))
        # images, _ = self.mesh_render(flame_verts, PerspectiveCameras(R=rotation, T=translation, **cameras_kwargs))
        # images = images * 0.5 + frames.to(self._device) * 0.5
        # torchvision.utils.save_image(images.cpu()[:8]/255, './debug.jpg', nrow=4)
        # raise Exception
        return lightning_results

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
            vis_image = torchvision.utils.make_grid([frame, vis_i], nrow=4)
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
