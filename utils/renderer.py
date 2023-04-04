import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import (
    Meshes, Pointclouds
)
from pytorch3d.renderer import (
    look_at_view_transform, RasterizationSettings, 
    FoVPerspectiveCameras, PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer,AlphaCompositor,
    PointLights, MeshRenderer, MeshRasterizer,  
    SoftPhongShader, TexturesVertex
)

from model.FLAME.masking import Masking

class Mesh_Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, device='cpu'):
        super(Mesh_Renderer, self).__init__()
        self.device = device
        verts, self.faces, aux = load_obj(obj_filename, load_textures=False)
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    def forward(self, vertices, cameras):
        faces = self.faces.verts_idx
        faces = faces[None].repeat(vertices.shape[0], 1, 1)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=faces.to(self.device),
            textures=textures
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=self.lights, device=self.device)
        )
        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        return images*255, alpha_images


class Point_Renderer:
    def __init__(self, image_size=256, device='cpu'):
        self.device = device
        R, T = look_at_view_transform(4, 30, 30) # d, e, a
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=1.0)
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, radius=0.005, points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        
    def render(self, points, D=3, E=15, A=30, coords=True, ex_points=None):
        if D !=8 or E != 30 or A != 30:
            R, T = look_at_view_transform(D, E, A) # d, e, a
            self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=1.0)
        verts = torch.Tensor(points).to(self.device)
        verts = verts[:, torch.randperm(verts.shape[1])[:10000]]
        if ex_points is not None:
            verts = torch.cat([verts, ex_points.expand(verts.shape[0], -1, -1)], dim=1)
        if coords:
            coords_size = verts.shape[1]//10
            cod = verts.new_zeros(coords_size*3, 3)
            li = torch.linspace(0, 1.0, steps=coords_size, device=cod.device)
            cod[:coords_size, 0], cod[coords_size:coords_size*2, 1], cod[coords_size*2:, 2] = li, li, li
            verts = torch.cat(
                [verts, cod.unsqueeze(0).expand(verts.shape[0], -1, -1)], dim=1
            )
        rgb = torch.Tensor(torch.rand_like(verts)).to(self.device)
        point_cloud = Pointclouds(points=verts, features=rgb)
        images = self.renderer(point_cloud, cameras=self.cameras).permute(0, 3, 1, 2)
        return images*255


class Texture_Renderer(nn.Module):
    def __init__(self, image_size, flame_path, device='cpu'):
        super(Texture_Renderer, self).__init__()
        self.device = device
        obj_filename = os.path.join(flame_path, 'head_template_mesh.obj')
        verts, self.faces, aux = load_obj(obj_filename, load_textures=False)
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = self.faces.textures_idx[None, ...]  # (N, F, 3)
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        self.face_uvcoords = face_vertices(uvcoords, uvfaces)
        from skimage.io import imread
        mask = torch.from_numpy(imread(os.path.join(flame_path, 'uv_mask_eyes.png')) / 255.).permute(2, 0, 1).to(device)[0:3, :, :]
        mask = mask > 0.
        self.mask = F.interpolate(mask[None].float(), [2048, 2048], mode='bilinear')

        self.rasterizer = TrackerRasterizer(image_size, None)
        self.masking = Masking(flame_path)

    def forward(self, vertices_world, texture_images, light_params, cameras):
        batch_size = vertices_world.shape[0]
        faces = self.faces.verts_idx
        faces = faces[None].expand(batch_size, -1, -1)
        meshes_world = Meshes(verts=vertices_world.to(self.device),faces=faces.to(self.device),)
        face_mask = face_vertices(self.masking.to_render_mask(self.masking.get_mask_face()).expand(batch_size, -1, -1), faces)
        face_normals = meshes_world.verts_normals_packed()[meshes_world.faces_packed()].reshape(batch_size, -1, 3, 3)
        uv = self.face_uvcoords.expand(batch_size, -1, -1, -1)
        # render
        params = [uv, face_normals, face_mask]
        for idx in range(len(params)):
            params[idx] = params[idx].to(self.device)
        attributes = torch.cat(params, -1)
        rendering, zbuffer = self.rasterizer(meshes_world, attributes, cameras=cameras)
        uvcoords_images = rendering[:, 0:3, :, :].detach()
        normal_images = rendering[:, 3:6, :, :].detach()
        mask_images_mesh = rendering[:, 6:9, :, :].detach()
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        # mask = self.mask.repeat(batch_size, 1, 1, 1)
        grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(texture_images.expand(batch_size, -1, -1, -1), grid, align_corners=False).float()
        # mask_images = F.grid_sample(mask, grid, align_corners=False).float()
        shading_images = self.add_SHlight(normal_images, light_params.to(self.device))
        images = albedo_images * shading_images
        
        mask_all = (alpha_images > 0).float().repeat(1, 3, 1, 1)
        mask_images_mesh = (mask_images_mesh > 0).float()
        return images * albedo_images, mask_all, mask_images_mesh

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        if not hasattr(self, 'constant_factor'):
            ## lighting
            pi = np.pi
            sh_const = torch.tensor(
                [
                    1 / np.sqrt(4 * pi),
                    ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                    ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                    ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                    (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                    (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                    (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                    (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                    (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
                ],
                dtype=torch.float32,
            )
            self.constant_factor = sh_const.to(self.device)

        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ], 1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class TrackerRasterizer(MeshRasterizer):
    def __init__(self, image_size, cameras) -> None:
        settings = RasterizationSettings()
        settings.image_size = (image_size, image_size)
        settings.perspective_correct = True
        settings.cull_backfaces = True

        super().__init__(cameras, settings)
        self.reset()

    def reset(self):
        self.bary_coords = None
        self.pix_to_face = None
        self.zbuffer = None

    def is_rasterize(self):
        return self.bary_coords is None and self.pix_to_face is None and self.zbuffer is None

    def forward(self, meshes, attributes, **kwargs):
        if self.is_rasterize():
            fragments = super().forward(meshes, **kwargs)
            self.bary_coords = fragments.bary_coords#.detach()
            self.pix_to_face = fragments.pix_to_face#.detach()
            self.zbuffer = fragments.zbuf.permute(0, 3, 1, 2)#.detach()

        vismask = (self.pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = self.bary_coords.shape
        mask = self.pix_to_face == -1
        pix_to_face = self.pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

        return pixel_vals, self.zbuffer
