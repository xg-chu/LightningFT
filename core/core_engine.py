import os
import sys
import math
import random
sys.path.append('./')

import lmdb
import torch
import mediapipe
import numpy as np
import torchvision
from tqdm.rich import tqdm
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


from .data_engine import DataEngine
from .calibration import optimize_camera
from .lmks_engine import LandmarksEngine
from .emoca_engine import Emoca_V2_Engine
from .lightning_engine import Lightning_Engine
# from utils.renderer import Mesh_Renderer, Point_Renderer
# from utils.utils import read_json, save_json
# from model.FLAME.FLAME import FLAME_MP
# from assets.FLAME.landmarks import get_mediapipe_indices

EMOCA_CKPT_PATH = './assets/EMOCA/EMOCA_v2_lr_mse_20/detail/checkpoints/deca-epoch=10-val_loss/dataloader_idx_0=3.25521111.ckpt'

class TrackEngine:
    def __init__(self, data_path, device='cuda'):
        self._debug = False
        self._device = device
        # paths and data engine
        path_dict = {
            'video_path': data_path, 'data_name': os.path.splitext(os.path.basename(data_path))[0],
            'output_path': os.path.join('outputs', os.path.splitext(os.path.basename(data_path))[0]),
        }
        self.data_engine = DataEngine(path_dict=path_dict)
        self.data_engine.build_data_lmdb()
        # lmks engine
        self.lmks_engine = LandmarksEngine(device=device, lazy_init=True)
        self.emoca_engine = Emoca_V2_Engine(device=device, lazy_init=True)
        self.lightning_engine = Lightning_Engine(device=device, lazy_init=True)

    def run(self, visualization=True):
        # landmarks
        if not self.data_engine.check_path('lmks_path'):
            landmarks = self.run_landmarks()
            self.data_engine.save(landmarks, 'lmks_path')
        # emoca
        if not self.data_engine.check_path('emoca_path'):
            emoca_results = self.run_emoca()
            self.data_engine.save(emoca_results, 'emoca_path')
        if not self.data_engine.check_path('camera_path'):
            cali_frames = random.choices(self.data_engine.frames(), k=32)
            batch_data = self.data_engine.get_frames(cali_frames, keys=['emoca', 'landmarks'])
            camera_params, calibration_image = optimize_camera(
                batch_data['emoca'], batch_data['landmarks'], batch_data['frames'], device=self._device
            )
            torchvision.utils.save_image(calibration_image/255.0, './debug.jpg')
            self.data_engine.save(camera_params, 'camera_path')
        # optimize landmarks
        if not self.data_engine.check_path('lightning_path'):
            lightning_results = self.run_lightning()
            self.data_engine.save(lightning_results, 'lightning_path')
        # smoothed landmarks
        if not self.data_engine.check_path('smoothed_path'):
            smoothed_results = self.run_smoothing()
            self.data_engine.save(smoothed_results, 'smoothed_path')
        # save video
        if visualization:
            render_images = self.render_video()
            self.data_engine.save(render_images, 'visul_path')

    def run_landmarks(self, ):
        all_landmarks = {}
        print('Annotating landmarks...')
        for frame_name in tqdm(self.data_engine.frames()):
            frame = self.data_engine.get_frame(frame_name)
            landmarks = self.lmks_engine.process_face(frame) # please input unnorm image
            all_landmarks[frame_name] = landmarks
        print('Done.')
        return all_landmarks

    def run_emoca(self, ):
        shape_codes, emoca_results = [], {}
        print('EMOCA encoding...')
        # processing
        for frame_name in tqdm(self.data_engine.frames()):
            frame = self.data_engine.get_frame(frame_name)
            landmarks = self.data_engine.get_landmarks(frame_name)['lmks_dense']
            emoca_res = self.emoca_engine.process_face(frame, landmarks) # please input unnorm image
            emoca_results[frame_name] = emoca_res
            shape_codes.append(emoca_res['shape'])
        shape_codes = torch.stack(shape_codes, dim=0).mean(dim=0)
        emoca_results['shape_code'] = shape_codes.cpu().half()
        print('Done.')
        return emoca_results

    def run_lightning(self, ):
        lightning_results = {}
        camera_params = self.data_engine.get_camera_params()
        self.lightning_engine.init_model(camera_params, image_size=512)
        mini_batchs = build_minibatch(self.data_engine.frames(), 128)
        print('Lightning tracking...')
        for batch_frames in tqdm(mini_batchs):
            batch_data = self.data_engine.get_frames(batch_frames, keys=['emoca', 'landmarks'])
            batch_data['emoca']['shape'] = self.data_engine.get_emoca_params('shape_code')[None].repeat(len(batch_frames), 1)
            lightning_res = self.lightning_engine.lightning_optimize_camera(
                batch_data['frame_names'], batch_data['emoca'], batch_data['landmarks'], batch_data['frames']
            )
            lightning_results.update(lightning_res)
        lightning_results['meta_info'] = camera_params
        lightning_results['meta_info']['shape_code'] = self.data_engine.get_emoca_params('shape_code')
        print('Done.')
        return lightning_results

    def render_video(self, ):
        from pytorch3d.renderer import look_at_view_transform
        print('Rendering...')
        anno_key = 'smoothed'
        camera_params = self.data_engine.get_camera_params()
        self.lightning_engine.init_model(camera_params, image_size=512)
        mini_batchs = build_minibatch(self.data_engine.frames()[:1000], 64)
        vis_images = []
        shape_code = self.data_engine.get_smoothed_params('meta_info')['shape_code']
        for batch_frames in tqdm(mini_batchs):
            batch_data = self.data_engine.get_frames(batch_frames, keys=[anno_key])
            render_result = self.lightning_engine.render(batch_data['frames'], batch_data[anno_key], shape_code)
            vis_images += render_result
        vis_images = torch.stack(vis_images, dim=0).permute(0, 2, 3, 1).to(torch.uint8).cpu()
        print('Done.')
        return vis_images

    def run_smoothing(self, ):
        def smooth_params(params):
            kf = KalmanFilter(initial_state_mean=params[0], n_dim_obs=params.shape[-1])
            smoothed_params = kf.em(params).smooth(params)[0]
            return smoothed_params

        from pykalman import KalmanFilter
        print('Running Kalman Smoother...')
        smoothed_results = {}
        bboxes, quaternions, translations = [], [], []
        for frame_name in self.data_engine.frames():
            # bboxes.append(abs_results[frame_name]['bbox'])
            smoothed_results[frame_name] = self.data_engine.get_lightning_params(frame_name)
            transform_matrix = smoothed_results[frame_name]['transform_matrix']
            quaternions.append(matrix_to_rotation_6d(transform_matrix[:3, :3]).numpy())
            translations.append(transform_matrix[:3, 3].numpy())
        # bboxes = smooth_params(np.array(bboxes))
        quaternions = smooth_params(np.array(quaternions))
        translations = smooth_params(np.array(translations))
        for idx, frame_name in enumerate(self.data_engine.frames()):
            # smoothed_results[frame_name]['bbox'] = bboxes[idx].tolist()
            rotation = rotation_6d_to_matrix(torch.tensor(quaternions[idx]))
            affine_matrix = torch.cat([rotation, torch.tensor(translations[idx])[:, None]], dim=-1).half().cpu()
            smoothed_results[frame_name]['transform_matrix'] = affine_matrix
        smoothed_results['meta_info'] = self.data_engine.get_lightning_params('meta_info')
        print('Done')
        return smoothed_results


def build_minibatch(all_frames, batch_size=32):
    all_mini_batch, mini_batch = [], []
    for frame_name in all_frames:
        mini_batch.append(frame_name)
        if len(mini_batch) % batch_size == 0:
            all_mini_batch.append(mini_batch)
            mini_batch = []
    if len(mini_batch):
        all_mini_batch.append(mini_batch)
    return all_mini_batch
