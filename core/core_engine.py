import os
import sys
import random
sys.path.append('./')

import torch
import numpy as np
from tqdm.rich import tqdm

from .data_engine import DataEngine
from .calibration import optimize_camera
from .detection_engine import Detection_Engine
from .emoca_engine import Emoca_V2_Engine
from .lightning_engine import Lightning_Engine
from .synthesis_engine import Synthesis_Engine
from .render_engine import Render_Engine

FLAME_MODEL_PATH = './assets/FLAME'
EMOCA_CKPT_PATH = './assets/EMOCA/EMOCA_v2_lr_mse_20/detail/checkpoints/deca-epoch=10-val_loss/dataloader_idx_0=3.25521111.ckpt'

class TrackEngine:
    def __init__(self, args_config, device='cuda'):
        self._debug = False
        self._device = device
        self._args_config = args_config
        # paths and data engine
        path_dict = {
            'video_path': self._args_config.data, 
            'data_name': os.path.splitext(os.path.basename(self._args_config.data))[0],
            'output_path': os.path.join('outputs', os.path.splitext(os.path.basename(self._args_config.data))[0]),
        }
        self.data_engine = DataEngine(path_dict=path_dict)
        self.data_engine.build_data_lmdb()
        # lmks engine
        self.lmks_engine = Detection_Engine(device=device, lazy_init=True)
        self.emoca_engine = Emoca_V2_Engine(EMOCA_CKPT_PATH, device=device, lazy_init=True)
        self.lightning_engine = Lightning_Engine(FLAME_MODEL_PATH, device=device, lazy_init=True)
        self.synthesis_engine = Synthesis_Engine(FLAME_MODEL_PATH, device=device, lazy_init=True)

    def run(self, ):
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
            batch_data = self.data_engine.get_frames(cali_frames, keys=['emoca', 'lmks'], device=self._device)
            camera_params, calibration_image = optimize_camera(
                batch_data['emoca'], batch_data['lmks'], batch_data['frames'], device=self._device
            )
            self.data_engine.save(camera_params, 'camera_path')
            self.data_engine.save(calibration_image/255.0, 'visul_calib_path')
        # optimize landmarks
        if not self.data_engine.check_path('lightning_path'):
            lightning_results = self.run_lightning()
            self.data_engine.save(lightning_results, 'lightning_path')
        # synthesis optimization
        if self._args_config.synthesis and not self.data_engine.check_path('synthesis_path'):
            synthesis_results = self.run_synthesis()
            self.data_engine.save(synthesis_results, 'synthesis_path')
        # # smoothed landmarks
        if not self._args_config.no_smooth and not self.data_engine.check_path('smoothed_path'):
            smoothed_results = self.run_smoothing(
                anno_key='synthesis' if self._args_config.synthesis else 'lightning', 
                type=self._args_config.smooth_type
            )
            self.data_engine.save(smoothed_results, 'smoothed_path')
        # save video
        if self._args_config.visualization:
            # render_images = self.render_video(anno_key='synthesis' if self._args_config.synthesis else 'lightning')
            render_images = self.render_video(anno_key='smoothed')
            self.data_engine.save(render_images, 'visul_path', fps=self._args_config.visualization_fps)

    def run_landmarks(self, ):
        all_landmarks = {}
        print('Annotating landmarks...')
        last_frame_name = None
        for frame_name in tqdm(self.data_engine.frames(), ncols=120, colour='#95bb72'):
            frame = self.data_engine.get_frame(frame_name)
            landmarks = self.lmks_engine.process_face(frame) # please input unnorm image
            if landmarks['lmks'] is None:
                landmarks['lmks'] = all_landmarks[last_frame_name]['lmks']
            if landmarks['lmks_dense'] is None:
                landmarks['lmks_dense'] = all_landmarks[last_frame_name]['lmks_dense']
            all_landmarks[frame_name] = landmarks
            last_frame_name = frame_name
        print('Done.')
        return all_landmarks

    def run_emoca(self, ):
        shape_codes, emoca_results = [], {}
        print('EMOCA encoding...')
        # processing
        for frame_name in tqdm(self.data_engine.frames(), ncols=120, colour='#95bb72'):
            frame = self.data_engine.get_frame(frame_name).float().to(self._device)
            landmarks = self.data_engine.get_data('lmks_path', query_name=frame_name, device=self._device)['lmks_dense']
            emoca_res = self.emoca_engine.process_face(frame, landmarks) # please input unnorm image
            emoca_results[frame_name] = emoca_res
            shape_codes.append(emoca_res['shape'])
        shape_codes = torch.stack(shape_codes, dim=0).mean(dim=0)
        emoca_results['shape_code'] = shape_codes.cpu().half()
        print('Done.')
        return emoca_results

    def run_lightning(self, ):
        lightning_results = {}
        camera_params = self.data_engine.get_data('camera_path', device=self._device)
        self.lightning_engine.init_model(camera_params, image_size=512)
        mini_batchs = build_minibatch(self.data_engine.frames(), 128)
        print('Lightning tracking...')
        for batch_frames in tqdm(mini_batchs, ncols=120, colour='#95bb72'):
            batch_data = self.data_engine.get_frames(batch_frames, keys=['emoca', 'lmks'], device=self._device)
            batch_data['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code', device=self._device)
            lightning_res = self.lightning_engine.lightning_optimize(batch_data)
            lightning_results.update(lightning_res)
        lightning_results['meta_info'] = camera_params
        lightning_results['meta_info']['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code').half()
        print('Done.')
        return lightning_results

    def run_synthesis(self, ):
        synthesis_results = {}
        camera_params = self.data_engine.get_data('camera_path', device=self._device)
        self.synthesis_engine.init_model(camera_params, image_size=512)
        if not self.data_engine.check_path('texture_path'):
            # optimize texture
            print('optimizing texture...')
            random_frames = random.choices(self.data_engine.frames(), k=32)
            batch_data = self.data_engine.get_frames(random_frames, keys=['lightning'], device=self._device)
            batch_data['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code', device=self._device)
            tex_params, tex_image = self.synthesis_engine.optimize_texture(batch_data)
            self.data_engine.save(tex_params, 'texture_path')
            self.data_engine.save(tex_image, 'visul_texture_path')
            tex_params = self.data_engine.get_data('texture_path', device=self._device)
            print('Done.')
        else:
            tex_params = self.data_engine.get_data('texture_path', device=self._device)

        mini_batchs = build_minibatch(self.data_engine.frames(), 64)
        print('Synthesis tracking...')
        for batch_frames in tqdm(mini_batchs, ncols=120, colour='#95bb72'):
            batch_data = self.data_engine.get_frames(batch_frames, keys=['lightning', 'lmks'], device=self._device)
            batch_data['texture_code'] = tex_params['texture_params'].clone()
            batch_data['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code', device=self._device)
            synthesis_res = self.synthesis_engine.synthesis_optimize(batch_data)
            synthesis_results.update(synthesis_res)
        synthesis_results['meta_info'] = camera_params
        synthesis_results['meta_info']['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code')
        print('Done.')
        return synthesis_results

    def render_video(self, anno_key='synthesis'):
        with_texture = self._args_config.synthesis
        print('Rendering...')
        camera_params = self.data_engine.get_data('camera_path', device=self._device)
        render_engine = Render_Engine(camera_params, FLAME_MODEL_PATH, with_texture=with_texture, device=self._device)
        vis_images = []
        mini_batchs = build_minibatch(self.data_engine.frames()[:1600], 64)
        for batch_frames in tqdm(mini_batchs, ncols=120, colour='#95bb72'):
            batch_data = self.data_engine.get_frames(batch_frames, keys=[anno_key], device=self._device)
            if with_texture:
                batch_data['texture_code'] = self.data_engine.get_data('texture_path', query_name='texture_params', device=self._device)
            batch_data['shape_code'] = self.data_engine.get_data('emoca_path', query_name='shape_code', device=self._device)
            vis_images += render_engine(batch_data, anno_key)
        vis_images = [i.to(torch.uint8).cpu() for i in vis_images]
        vis_images = torch.stack(vis_images, dim=0).permute(0, 2, 3, 1)
        print('Done.')
        return vis_images

    def run_smoothing(self, anno_key='synthesis', type='kalman'):
        if type == 'kalman':
            from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
            def smooth_params(params):
                from pykalman import KalmanFilter
                kf = KalmanFilter(initial_state_mean=params[0], n_dim_obs=params.shape[-1])
                smoothed_params = kf.em(params).smooth(params)[0]
                return smoothed_params
            print('Running Kalman Smoother...')
            smoothed_results = {}
            bboxes, quaternions, translations = [], [], []
            for frame_name in self.data_engine.frames():
                smoothed_results[frame_name] = self.data_engine.get_data(anno_key+'_path', query_name=frame_name)
                transform_matrix = smoothed_results[frame_name]['transform_matrix'].detach()
                bboxes.append(smoothed_results[frame_name]['bbox'].numpy())
                quaternions.append(matrix_to_rotation_6d(transform_matrix[:3, :3]).numpy())
                translations.append(transform_matrix[:3, 3].numpy())
            bboxes = smooth_params(np.array(bboxes))
            quaternions = smooth_params(np.array(quaternions))
            quaternions = smooth_params(np.array(quaternions))
            translations = smooth_params(np.array(translations))
            translations = smooth_params(np.array(translations))
            for idx, frame_name in enumerate(self.data_engine.frames()):
                smoothed_results[frame_name]['bbox'] = torch.tensor(bboxes[idx])
                rotation = rotation_6d_to_matrix(torch.tensor(quaternions[idx]))
                affine_matrix = torch.cat([rotation, torch.tensor(translations[idx])[:, None]], dim=-1).half().cpu()
                smoothed_results[frame_name]['transform_matrix'] = affine_matrix
            smoothed_results['meta_info'] = self.data_engine.get_data(anno_key+'_path', query_name=frame_name)
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
