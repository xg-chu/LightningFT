import os
import math
import torch
import torchvision
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from model.FLAME.FLAME import FLAME_MP, FLAME_Tex
from utils.renderer import Mesh_Renderer, Texture_Renderer, Point_Renderer

class Render_Engine(torch.nn.Module):
    def __init__(self, camera_params, flame_model_path, image_size=512, with_texture=False, device='cuda'):
        super(Render_Engine, self).__init__()

        self._device = device
        self._with_texture = with_texture
        self.image_size = image_size
        self.flame_scale = camera_params['flame_scale']
        self.focal_length = camera_params['focal_length'].to(self._device)
        self.principal_point = camera_params['principal_point'].to(self._device)
        # build model
        self._with_texture = with_texture
        self.flame_model = FLAME_MP(flame_model_path, 100, 50).to(self._device)
        self.point_render = Point_Renderer(image_size=image_size, device=self._device)
        if not with_texture:
            self.mesh_render = Mesh_Renderer(
                512, obj_filename=os.path.join(
                    flame_model_path, 'FLAME_embedding', 'head_template_mesh.obj'
                ), device=self._device
            )
        else:
            self.flame_texture = FLAME_Tex(flame_model_path, image_size=image_size).to(self._device)
            self.mesh_render = Texture_Renderer(
                512, flame_path=flame_model_path, flame_mask=None, device=self._device
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

    @torch.no_grad()
    def forward(self, batch_data, anno_key):
        batch_size = len(batch_data['frame_names'])
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        # flame
        cameras = PerspectiveCameras(
            R=batch_data[anno_key]['transform_matrix'][:, :3, :3], 
            T=batch_data[anno_key]['transform_matrix'][:, :3, 3], 
            **cameras_kwargs
        )
        flame_verts, pred_lmk_68, pred_lmk_dense = self.flame_model(
            shape_params=batch_data['shape_code'][None].expand(batch_size, -1), 
            expression_params=batch_data[anno_key]['expression'],
            pose_params=batch_data[anno_key]['flame_pose']
        )
        flame_verts = flame_verts * self.flame_scale
        pred_lmk_68, pred_lmk_dense = pred_lmk_68 * self.flame_scale, pred_lmk_dense * self.flame_scale
        # render
        # points_image = self.point_render(torch.cat([pred_lmk_68, pred_lmk_dense], dim=1))
        points_image = self.point_render(flame_verts)
        if self._with_texture:
            if not hasattr(self, 'albedos'):
                self.albedos = self.flame_texture(batch_data['texture_code'])
            images, alpha_images, _ = self.mesh_render(flame_verts, self.albedos, cameras)
            images = (images * 255.0).clamp(0, 255)
        else:
            images, alpha_images = self.mesh_render(flame_verts, cameras)
        # gather
        points_68 = cameras.transform_points_screen(pred_lmk_68)[..., :2]
        points_dense = cameras.transform_points_screen(pred_lmk_dense)[..., :2]
        vis_images = []
        alpha_images = alpha_images.expand(-1, 3, -1, -1)
        for idx, frame in enumerate(batch_data['frames']):
            vis_i = frame.clone()
            vis_i[alpha_images[idx]>0.5] *= 0.3
            vis_i[alpha_images[idx]>0.5] += (images[idx, alpha_images[idx]>0.5] * 0.7)
            vis_i = torchvision.utils.draw_keypoints(vis_i.to(torch.uint8), points_dense[idx:idx+1], colors="red", radius=1.5)
            vis_i = torchvision.utils.draw_keypoints(vis_i.to(torch.uint8), points_68[idx:idx+1], colors="blue", radius=1.5)
            vis_i = torchvision.utils.draw_bounding_boxes(vis_i, batch_data[anno_key]['face_box'][idx:idx+1]*frame.shape[-1])
            vis_image = torchvision.utils.make_grid([frame.cpu(), images[idx].cpu(), vis_i.cpu(), points_image[idx].cpu()], nrow=4)
            vis_images.append(vis_image)
        return vis_images
