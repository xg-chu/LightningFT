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


from .data_engine import DataEngine
from .lmks_engine import LandmarksEngine
from .emoca_engine import Emoca_V2_Engine
from .calibration import optimize_camera
from utils.renderer import Mesh_Renderer, Point_Renderer
from utils.utils import read_json, save_json
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
        

    def run_tracking_based_on(self, tracking_target):
        target = read_json(tracking_target)
        # landmarks
        if not os.path.exists(self.landmarks_path):
            landmarks = self.run_landmarks(self._lmdb_txn)
            torch.save(landmarks, self.landmarks_path)
        else:
            landmarks = torch.load(self.landmarks_path)
            print('Load buffered landmarks.')
        # emoca
        if not os.path.exists(self.emoca_results_path):
            emoca_results = self.run_emoca(landmarks, self._lmdb_txn)
            save_json(emoca_results, self.emoca_results_path)
        else:
            emoca_results = read_json(self.emoca_results_path)
            print('Load buffered EMOCA results.')
        # optimize landmarks
        camera_fov = target['meta_info']['fov']
        abs_results = self.run_abs(landmarks, emoca_results, camera_fov, meta_info=target['meta_info'])
        save_json(abs_results, self.abs_results_path)

        abs_results = self.kalman_smoother(abs_results)
        save_json(abs_results, self.smoothed_abs_results_path)

    def run(self, save_video=True):
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
            frame_shape = self.data_engine.get_frame(cali_frames[0]).shape
            frames, emoca_params, gt_landmarks = [], [], []
            for f in cali_frames:
                frame = self.data_engine.get_frame(f)
                emo = self.data_engine.get_emoca_params(f)
                lmk = self.data_engine.get_landmarks(f)
                if emo is not None:
                    frames.append(frame)
                    emoca_params.append(emo)
                    gt_landmarks.append(lmk)
            frames = torch.utils.data.default_collate(frames)
            emoca_params = torch.utils.data.default_collate(emoca_params)
            gt_landmarks = torch.utils.data.default_collate(gt_landmarks)
            camera_params, calibration_image = optimize_camera(emoca_params, gt_landmarks, frames)
            torchvision.utils.save_image(calibration_image/255.0, './debug.jpg')
            # self.data_engine.save(camera_params, 'camera_path')
        # optimize landmarks
        if not os.path.exists(self.abs_results_path):
            camera_fov, calibration_result = self.calibration(landmarks, emoca_results, self._lmdb_txn)
            torchvision.utils.save_image(calibration_result, self.calibration_result_path, nrow=4)
            abs_results = self.run_abs(landmarks, emoca_results, camera_fov)
            save_json(abs_results, self.abs_results_path)
        else:
            abs_results = read_json(self.abs_results_path)
            print('Load buffered landmark optimization results.')
        print('FOV: {}'.format(abs_results['meta_info']['fov']))
        # kalman smooth
        if not os.path.exists(self.smoothed_abs_results_path):
            abs_results = self.kalman_smoother(abs_results)
            save_json(abs_results, self.smoothed_abs_results_path)
        else:
            abs_results = read_json(self.smoothed_abs_results_path)
            print('Load buffered smoothed results.')
        # save video
        if save_video:
            self.render_video(abs_results, self._lmdb_txn, self.track_video_path)

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
        emoca_results = {}
        print('EMOCA encoding...')
        # processing
        for frame_name in tqdm(self.data_engine.frames()):
            frame = self.data_engine.get_frame(frame_name)
            landmarks = self.data_engine.get_landmarks(frame_name)['lmks_dense']
            emoca_res = self.emoca_engine.process_face(frame, landmarks) # please input unnorm image
            emoca_results[frame_name] = emoca_res
        print('Done.')
        return emoca_results

    @staticmethod
    def calibration(landmarks, emoca_results, lmdb_txn, device='cuda', cali_size=512):
        def optimize_landmarks(pred_lmk, gt_lmk, init_fov, length=1000):
            optimizer = torch.optim.Adam([
                    {'params': [FOV], 'lr': 1.0e-2},
                    {'params': [TRANS_XY], 'lr': 1.0e-3},
                    # {'params': [TRANS_Z], 'lr': 2.0e-1},
                ], lr=1.0e-5, amsgrad=False
            )
            best_dist, best_fov = 10000, init_fov
            all_dists, all_fovs = [], []
            for idx in tqdm(range(length), ncols=80, colour='#95bb72'):
                TRANS_Z = (1 / torch.tan(FOV[None]/360*math.pi)).expand(batch_size, -1)
                TRANS = torch.cat([TRANS_XY, TRANS_Z], dim=-1)
                optimizer.zero_grad()
                trans_matrix = camera.get_full_projection_transform(R=R, T=TRANS, fov=FOV)
                points = trans_matrix.transform_points(pred_lmk*flame_scale) * 0.5 + 0.5
                loss = torch.nn.functional.mse_loss(points[..., :2], gt_lmk[..., :2]) * 1.0e5
                loss.backward(retain_graph=True)
                optimizer.step()
                if loss < best_dist:
                    best_dist = loss.item()
                    best_fov = FOV.item()
                    all_dists.append(best_dist)
                    all_fovs.append(best_fov)
            result_fov = best_fov
            target = np.percentile(np.array(all_dists), 0.8)
            for i in range(len(all_dists)):
                if all_dists[i] < target:
                    result_fov = all_fovs[i]
                    break
            return result_fov, points.detach()
        
        flame = FLAME_MP(
            './assets/FLAME/generic_model.pkl', 
            './assets/FLAME/landmark_embedding.npy', 
            './assets/FLAME/mediapipe_landmark_embedding.npz', 
            './assets/FLAME/mediapipe_index.npy', 100, 50
        ).to(device)
        batch_size = 32
        cali_frames = random.choices(list(landmarks.keys()), k=batch_size)
        # build codes
        shape_codes = torch.tensor([emoca_results[f]['shape'] for f in cali_frames], device=device)
        exp_codes = torch.tensor([emoca_results[f]['exp'] for f in cali_frames], device=device)
        pose_codes = torch.tensor([emoca_results[f]['pose'] for f in cali_frames], device=device)
        cams = torch.tensor([emoca_results[f]['cam'] for f in cali_frames], device=device)
        pose_codes = torch.tensor([emoca_results[f]['pose'] for f in cali_frames], device=device)
        crop_box = torch.tensor([emoca_results[f]['crop_box'] for f in cali_frames], device=device)
        flame_scale = torch.tensor([cams[i][0]*(crop_box[i][2]/cali_size) for i in range(cams.shape[0])]).mean()
        # pred landmarks
        _, _, _, pred_key_lmk, pred_all_lmk = flame(
            shape_params=shape_codes, expression_params=exp_codes, pose_params=pose_codes
        )
        # gt landmarks
        gt_all_lmk = torch.stack([landmarks[f] for f in cali_frames], dim=0).to(device).float()
        gt_key_lmk = gt_all_lmk[:, get_mediapipe_indices()]
        # optimization
        # build params
        init_fov = 20.0
        FOV = torch.nn.Parameter(torch.tensor([init_fov], device=device), requires_grad=True)
        R = torch.tensor([[[1.,  0.,  0.], [ 0.,  -1.,  0.], [ 0.,  0., 1.]]], device=device).repeat(batch_size, 1, 1)
        normed_lmk = torch.einsum('nk,bmk->bmn', [R[0], pred_all_lmk * flame_scale * 0.5]) + 0.5
        shifts = gt_all_lmk.mean(dim=1) - normed_lmk.mean(dim=1)
        TRANS = shifts * 2 # normed_lmk += TRANS[:, None] * 0.5
        TRANS[..., 2] = 1 / math.tan(init_fov/360*math.pi)
        TRANS_XY = torch.nn.Parameter(TRANS[:, :2], requires_grad=True)
        # TRANS_Z = torch.nn.Parameter(TRANS[:, 2:], requires_grad=True)
        camera = FoVPerspectiveCameras(device=device)
        # optimize
        best_fov, points = optimize_landmarks(pred_key_lmk, gt_key_lmk, init_fov, length=300)
        best_fov, points = optimize_landmarks(pred_all_lmk, gt_all_lmk, best_fov, length=50)
        # visulization
        vis_calibration = []
        for idx, frame_name in enumerate(cali_frames):
            frame = load_img(frame_name, lmdb_txn=lmdb_txn)
            vis_i = torchvision.utils.draw_keypoints(frame.to(torch.uint8), gt_all_lmk[idx:idx+1]*512, colors="white", radius=1.5)
            vis_i = torchvision.utils.draw_keypoints(vis_i, gt_key_lmk[idx:idx+1]*512, colors="red", radius=1.5)
            vis_i = torchvision.utils.draw_keypoints(vis_i, points[idx:idx+1]*512, colors="green", radius=1.5)
            vis_calibration.append(vis_i)
        vis_calibration = torch.stack(vis_calibration[:8], dim=0)
        return best_fov, vis_calibration/255.0

    @staticmethod
    def run_abs(landmarks, emoca_results, camera_fov, meta_info=None, device='cuda'):
        def flame_to_camera(pred_lmks, gt_lmks, flame_pose):
            # rotation
            flame_pose[:, 1] += math.pi
            flame_pose[:, 0] *= -1
            rotation_matrix = euler_angles_to_matrix(flame_pose, ['X', 'Y', 'Z']).permute(0, 2, 1)
            # translation
            translation = rotation_matrix.new_zeros(rotation_matrix.shape[0], 3)
            translation[..., 2] = 1/np.tan(camera_fov/360*math.pi)
            camera.get_camera_center(R=rotation_matrix, T=translation)
            pred_lmks = camera.transform_points_screen(pred_lmks, image_size=(1, 1))
            translation[..., :2] = (pred_lmks.mean(dim=1) - gt_lmks.mean(dim=1))[..., :2] * 2
            return rotation_matrix, translation
        
        def build_minibatch(batch_size, num_frames, emoca_results, landmarks):
            all_mini_batch = []
            mini_batch = {'name': [], 'exp': [], 'pose': [], 'landmarks': []}
            for idx in range(num_frames):
                frame_name = 'f_{:07d}.jpg'.format(idx)
                mini_batch['name'].append(frame_name)
                mini_batch['exp'].append(torch.tensor(emoca_results[frame_name]['exp'], device=device))
                mini_batch['pose'].append(torch.tensor(emoca_results[frame_name]['pose'], device=device))
                mini_batch['landmarks'].append(landmarks[frame_name].to(device).float())
                if len(mini_batch['name']) % batch_size == 0:
                    mini_batch['exp'] = torch.stack(mini_batch['exp'])
                    mini_batch['pose'] = torch.stack(mini_batch['pose'])
                    mini_batch['landmarks'] = torch.stack(mini_batch['landmarks'])
                    all_mini_batch.append(mini_batch)
                    mini_batch = {'name': [], 'exp': [], 'pose': [], 'landmarks': []}
            if len(mini_batch['name']):
                mini_batch['exp'] = torch.stack(mini_batch['exp'])
                mini_batch['pose'] = torch.stack(mini_batch['pose'])
                mini_batch['landmarks'] = torch.stack(mini_batch['landmarks'])
                all_mini_batch.append(mini_batch)
            return all_mini_batch

        def optimize_landmarks(pred_lmks, gt_lmks, rotation_matrix, translation, length=500):
            TRANS_XY = torch.nn.Parameter(translation[..., :2], requires_grad=True)
            TRANS_Z = torch.nn.Parameter(translation[..., 2:], requires_grad=True)
            rotation = matrix_to_quaternion(rotation_matrix)
            ROTATS = torch.nn.Parameter(rotation, requires_grad=True)
            optimizer = torch.optim.Adam([
                    {'params': [TRANS_XY], 'lr': 5.0e-3},
                    {'params': [TRANS_Z], 'lr': 5.0e-4},
                    # {'params': [ROTATS], 'lr': 1.0e-3},
                ], lr=5e-3, amsgrad=False
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.3, total_iters=length
            )
            best_dist, best_r, best_t = 10000, None, None
            for idx in range(length):
                TRANS = torch.cat([TRANS_XY, TRANS_Z], dim=-1)
                optimizer.zero_grad()
                rotation_matrix = quaternion_to_matrix(ROTATS)
                camera.get_camera_center(R=rotation_matrix, T=TRANS)
                # render
                points = camera.transform_points_screen(pred_lmks, image_size=(1, 1))
                loss = torch.nn.functional.mse_loss(points[..., :2], gt_lmks[..., :2]) * 1.0e5
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                if loss < best_dist:
                    best_dist = loss.item()
                    best_r = rotation_matrix.detach()
                    best_t = TRANS.detach()
            return best_r, best_t

        print('Optimizing landmarks...')
        flame = FLAME_MP(
            './assets/FLAME/generic_model.pkl', 
            './assets/FLAME/landmark_embedding.npy', 
            './assets/FLAME/mediapipe_landmark_embedding.npz', 
            './assets/FLAME/mediapipe_index.npy', 100, 50
        ).to(device)
        mini_batch_size = 128
        num_frames = len(emoca_results.keys())
        if meta_info is None:
            shape_code = torch.tensor([emoca_results[k]['shape'] for k in emoca_results.keys()]).mean(dim=0)
            shape_code = shape_code[None].to(device).repeat(mini_batch_size, 1)
            flame_scale = torch.tensor([emoca_results[k]['cam'][0]*(emoca_results[k]['crop_box'][2]/512) for k in emoca_results.keys()]).mean().item()
            _, _, _, _, canonical_mp_landmarks = flame(
                shape_params=shape_code[:1], 
                expression_params=shape_code.new_zeros(1, 50), pose_params=shape_code.new_zeros(1, 6)
            )
            canonical_mp_landmarks = canonical_mp_landmarks * flame_scale
        else:
            shape_code = torch.tensor(meta_info['shape_code'])
            shape_code = shape_code[None].to(device).repeat(mini_batch_size, 1)
            flame_scale = meta_info['flame_scale']
            canonical_mp_landmarks = torch.tensor(meta_info['canonical_mp_landmarks'])[None]
        all_mini_batch = build_minibatch(mini_batch_size, num_frames, emoca_results, landmarks)
        camera = FoVPerspectiveCameras(device=device, fov=camera_fov)
        lmks_abs_results = {}
        for mini_batch in tqdm(all_mini_batch, ncols=80, colour='#95bb72'):
            flame_pose = mini_batch['pose'].clone()
            flame_pose[..., :3] *= 0
            this_batch_size = flame_pose.shape[0]
            # pred landmarks
            flame_verts, _, _, pred_key_lmk, pred_all_lmk = flame(
                shape_params=shape_code[:this_batch_size], 
                expression_params=mini_batch['exp'], pose_params=flame_pose
            )
            pred_key_lmk, pred_all_lmk, flame_verts = pred_key_lmk * flame_scale, \
                                                      pred_all_lmk * flame_scale, \
                                                      flame_verts * flame_scale
            # gt landmarks
            gt_all_lmk = mini_batch['landmarks']
            gt_key_lmk = gt_all_lmk[:, get_mediapipe_indices()]
            annot_mideapipe_landmarks = pred_all_lmk.clone()
            # transform params
            rotation_matrix, translation = flame_to_camera(pred_all_lmk, gt_all_lmk, mini_batch['pose'][..., :3])
            rotation_matrix, translation = optimize_landmarks(
                pred_all_lmk, gt_all_lmk, rotation_matrix, translation, length=200
            )
            rotation_matrix, translation = optimize_landmarks(
                pred_key_lmk, gt_key_lmk, rotation_matrix, translation, length=20
            )
            # set camera
            affine_matrix = torch.cat([rotation_matrix, translation[:, :, None]], dim=-1)
            ### DEBUG
            # if True:
            #     affine_matrix = torch.cat([rotation_matrix, translation[:, :, None]], dim=-1)
            #     render = Mesh_Renderer(
            #         512, obj_filename='./assets/FLAME/head_template.obj', device=device
            #     )
            #     camera.get_camera_center(R=affine_matrix[:, :3, :3], T=affine_matrix[:, :3, 3])
            #     images, _ = render(flame_verts, camera)
            #     images = images.cpu()
            #     torchvision.utils.save_image(images.cpu()[:8]/255, './outputs/debug.jpg', nrow=4)
            #     raise Exception
            affine_matrix = affine_matrix.cpu().numpy().tolist()
            expression = mini_batch['exp'].cpu().numpy().tolist()
            flame_pose = mini_batch['pose'].cpu().numpy().tolist()
            annot_mideapipe_landmarks = annot_mideapipe_landmarks.cpu().numpy().tolist()
            for idx, frame_name in enumerate(mini_batch['name']):
                lmks_list = [
                    [round(c, 3) for c in line] for line in annot_mideapipe_landmarks[idx] 
                ]
                frame_dict = {'file_path':frame_name}
                frame_dict['transform_matrix'] = affine_matrix[idx]
                frame_dict['expression'] = expression[idx]
                frame_dict['flame_pose'] = flame_pose[idx]
                frame_dict['mp_landmarks'] = lmks_list
                crop_box = emoca_results[frame_name]['crop_box']
                frame_dict['bbox'] = [
                    crop_box[0]-crop_box[2]/2, crop_box[1]-crop_box[2]//2,
                    crop_box[0]+crop_box[2]/2, crop_box[1]+crop_box[2]//2,
                ]
                lmks_abs_results[frame_name] = frame_dict
        lmks_abs_results['meta_info'] = {
            'fov': round(camera_fov, 3), 
            'flame_scale': round(flame_scale, 3),
            'canonical_mp_landmarks': canonical_mp_landmarks[0].cpu().numpy().tolist(),
            'shape_code': shape_code[0].cpu().numpy().tolist(),
        }
        print('Landmark optimization done.')
        return lmks_abs_results

    @staticmethod
    def render_video(abs_results, lmdb_txn, video_path, device='cuda'):
        from pytorch3d.renderer import look_at_view_transform
        def build_minibatch(batch_size, num_frames, abs_results):
            all_mini_batch = []
            mini_batch = {'name': [], 'expression': [], 'flame_pose': [], 'transform_matrix': [], 'bbox': [], 'mp_landmarks': []}
            for idx in range(num_frames):
                frame_name = 'f_{:07d}.jpg'.format(idx)
                mini_batch['name'].append(frame_name)
                mini_batch['bbox'].append(torch.tensor(abs_results[frame_name]['bbox'], device=device))
                mini_batch['expression'].append(torch.tensor(abs_results[frame_name]['expression'], device=device))
                mini_batch['flame_pose'].append(torch.tensor(abs_results[frame_name]['flame_pose'], device=device))
                mini_batch['transform_matrix'].append(torch.tensor(abs_results[frame_name]['transform_matrix'], device=device))
                mini_batch['mp_landmarks'].append(torch.tensor(abs_results[frame_name]['mp_landmarks'], device=device))
                if len(mini_batch['name']) % batch_size == 0:
                    mini_batch['bbox'] = torch.stack(mini_batch['bbox'])
                    mini_batch['expression'] = torch.stack(mini_batch['expression'])
                    mini_batch['flame_pose'] = torch.stack(mini_batch['flame_pose'])
                    mini_batch['transform_matrix'] = torch.stack(mini_batch['transform_matrix'])
                    mini_batch['mp_landmarks'] = torch.stack(mini_batch['mp_landmarks'])
                    all_mini_batch.append(mini_batch)
                    mini_batch = {'name': [], 'expression': [], 'flame_pose': [], 'transform_matrix': [], 'bbox': [], 'mp_landmarks': []}
            if len(mini_batch['name']):
                mini_batch['bbox'] = torch.stack(mini_batch['bbox'])
                mini_batch['expression'] = torch.stack(mini_batch['expression'])
                mini_batch['flame_pose'] = torch.stack(mini_batch['flame_pose'])
                mini_batch['transform_matrix'] = torch.stack(mini_batch['transform_matrix'])
                mini_batch['mp_landmarks'] = torch.stack(mini_batch['mp_landmarks'])
                all_mini_batch.append(mini_batch)
            return all_mini_batch

        print('Writing video...')
        flame = FLAME_MP(
            './assets/FLAME/generic_model.pkl', 
            './assets/FLAME/landmark_embedding.npy', 
            './assets/FLAME/mediapipe_landmark_embedding.npz', 
            './assets/FLAME/mediapipe_index.npy', 100, 50
        ).to(device)
        render = Mesh_Renderer(
            512, obj_filename='./assets/FLAME/head_template.obj', device=device
        )
        mediapipe_render = Point_Renderer(512, device=device)
        mini_batch_size = 64
        num_frames = min(len(abs_results.keys()) - 1, 1000)
        shape_code = torch.tensor(abs_results['meta_info']['shape_code'])[None].to(device).repeat(mini_batch_size, 1)
        all_mini_batch = build_minibatch(mini_batch_size, num_frames, abs_results)
        camera = FoVPerspectiveCameras(device=device, fov=abs_results['meta_info']['fov'])
        mp_camera = FoVPerspectiveCameras(device=device, fov=abs_results['meta_info']['fov'])
        R, T = look_at_view_transform(4, 0, 0, device=device) # D, E, A
        mp_camera.get_camera_center(R=R, T=T)
        # render
        vis_images = []
        for mini_batch in tqdm(all_mini_batch, ncols=80, colour='#95bb72'):
            mini_batch['flame_pose'][..., :3] *= 0
            this_batch_size = len(mini_batch['name'])
            # pred landmarks
            flame_verts, _, _, pred_key_lmk, pred_all_lmk = flame(
                shape_params=shape_code[:this_batch_size], 
                expression_params=mini_batch['expression'], pose_params=mini_batch['flame_pose']
            )
            pred_key_lmk, pred_all_lmk, flame_verts = pred_key_lmk * abs_results['meta_info']['flame_scale'], \
                                                      pred_all_lmk * abs_results['meta_info']['flame_scale'], \
                                                      flame_verts * abs_results['meta_info']['flame_scale']
            # set camera
            camera.get_camera_center(
                R=mini_batch['transform_matrix'][:, :3, :3], T=mini_batch['transform_matrix'][:, :3, 3]
            )
            # render
            images, alpha_images = render(flame_verts, camera)
            mp_images = mediapipe_render.render(mini_batch['mp_landmarks'])
            images, alpha_images, mp_images = images.cpu(), alpha_images.cpu(), mp_images.cpu()
            pred_key_lmk = camera.transform_points_screen(pred_key_lmk, image_size=(512, 512))
            pred_all_lmk = camera.transform_points_screen(pred_all_lmk, image_size=(512, 512))
            alpha_images = alpha_images.repeat(1, 3, 1, 1)
            for idx, frame_name in enumerate(mini_batch['name']):
                frame = load_img(frame_name, lmdb_txn=lmdb_txn).float()
                vis_i = torchvision.utils.draw_keypoints(frame.to(torch.uint8), pred_all_lmk[idx:idx+1], colors="white", radius=1.5)
                vis_i = torchvision.utils.draw_keypoints(vis_i, pred_key_lmk[idx:idx+1], colors="red", radius=1.5)
                vis_i = torchvision.utils.draw_bounding_boxes(vis_i, mini_batch['bbox'][idx:idx+1])
                frame[alpha_images[idx]>0.5] *= 0.3
                frame[alpha_images[idx]>0.5] += (images[idx, alpha_images[idx]>0.5] * 0.7)
                vis_image = torchvision.utils.make_grid([vis_i, mp_images[idx], frame], nrow=3)
                vis_images.append(vis_image)
        vis_images = torch.stack(vis_images, dim=0).permute(0, 2, 3, 1).to(torch.uint8).cpu()
        torchvision.io.write_video(video_path, vis_images, fps=30)
        print('Writing video done.')

    @staticmethod
    def kalman_smoother(abs_results):
        def smooth_params(params):
            kf = KalmanFilter(initial_state_mean=params[0], n_dim_obs=params.shape[-1])
            smoothed_params = kf.em(params).smooth(params)[0]
            return smoothed_params

        from pykalman import KalmanFilter
        print('Running Kalman Smoother...')
        num_frames = len(abs_results.keys()) - 1
        bboxes, quaternions, translations = [], [], []
        for idx in range(num_frames):
            frame_name = 'f_{:07d}.jpg'.format(idx)
            bboxes.append(abs_results[frame_name]['bbox'])
            quaternions.append(
                matrix_to_quaternion(
                    torch.tensor(abs_results[frame_name]['transform_matrix'])[:3, :3], 
                ).numpy()
            )
            translations.append(
                np.array(abs_results[frame_name]['transform_matrix'])[:3, 3]
            )
        
        bboxes = smooth_params(np.array(bboxes))
        quaternions = smooth_params(np.array(quaternions))
        translations = smooth_params(np.array(translations))
        for idx in range(num_frames):
            frame_name = 'f_{:07d}.jpg'.format(idx)
            abs_results[frame_name]['bbox'] = bboxes[idx].tolist()
            rotation = quaternion_to_matrix(torch.tensor(quaternions[idx]))
            affine_matrix = torch.cat([rotation, torch.tensor(translations[idx])[:, None]], dim=-1)
            abs_results[frame_name]['transform_matrix'] = affine_matrix.numpy().tolist()
        print('Done')
        return abs_results

