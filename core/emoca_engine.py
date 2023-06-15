import torch
import mediapipe
import numpy as np
import torchvision
import face_alignment

from model.EMOCA import EMOCA

class Emoca_Engine:
    def __init__(self, emoca_ckpt_path, device='cuda', lazy_init=True):
        self._emoca_ckpt_path = emoca_ckpt_path
        self._device = device
        if not lazy_init:
            self._init_model()

    def _init_model(self, ):
        print('Initializing trimmed EMOCA V2 models...')
        # emocav2
        ckpt = torch.load(self._emoca_ckpt_path, map_location='cpu')['state_dict']
        trimmed_ckpt = {}
        for key in list(ckpt.keys()):
            if 'E_flame' in key or 'E_expression' in key:
                trimmed_ckpt[key.replace('deca.', '')] = ckpt[key]

        emoca_model = EMOCA().to(self._device)
        emoca_model.eval()
        emoca_model.load_state_dict(trimmed_ckpt, strict=True)
        self.emoca_model = emoca_model
        # landmarks
        self.lmks_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self._device)
        self.lmks_dense_model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        print('Done.')

    @staticmethod
    def _crop_frame(frame, landmark):
        min_xy = landmark.min(dim=0)[0]
        max_xy = landmark.max(dim=0)[0]
        box = torch.tensor([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])
        size = int((box[2]+box[3]-box[0]-box[1])/2*1.375)
        center = torch.tensor([(box[0]+box[2])/2.0, (box[1]+box[3])/2.0, size])
        frame = torchvision.transforms.functional.crop(
            frame.float(), 
            top=int(center[1]-size/2), left=int(center[0]-size/2),
            height=size, width=size,
        )
        frame = torchvision.transforms.functional.resize(frame, size=224, antialias=True)
        return frame, center

    def process_face(self, image):
        if not hasattr(self, 'emoca_model'):
            self._init_model()
        # face alignment
        lmk_image = image.permute(1, 2, 0)
        lmks, scores, detected_faces = self.lmks_model.get_landmarks_from_image(
            lmk_image, return_landmark_score=True, return_bboxes=True
        )
        if lmks is None:
            lmks = self.last_lmks
        else:
            lmks = torch.tensor(lmks[0]).half()
            self.last_lmks = lmks
        # mediapipe
        lmk_image = lmk_image.to(torch.uint8).cpu().numpy()
        lmks_dense = self.lmks_dense_model.process(lmk_image)
        if lmks_dense.multi_face_landmarks is None:
            lmks_dense = self.last_lmks_dense
        else:
            lmks_dense = lmks_dense.multi_face_landmarks[0].landmark
            lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
            lmks_dense[:, 0] = lmks_dense[:, 0] * lmk_image.shape[1]
            lmks_dense[:, 1] = lmks_dense[:, 1] * lmk_image.shape[0]
            lmks_dense = torch.tensor(lmks_dense).half()
            self.last_lmks_dense = lmks_dense
            
        croped_frame, crop_center = self._crop_frame(image, lmks_dense)
        torchvision.utils.save_image(croped_frame/255.0, 'debug.jpg')
        # please input normed image
        croped_frame = croped_frame.to(self._device)[None]/255.0
        emoca_result = self.emoca_model.encode(croped_frame)
        bbox = torch.stack([
            crop_center[0] - crop_center[2]/2, crop_center[1] - crop_center[2]/2,
            crop_center[0] + crop_center[2]/2, crop_center[1] + crop_center[2]/2,
        ])
        bbox[[0, 2]] /= image.shape[-1]
        bbox[[1, 3]] /= image.shape[-2]
        results = {
            'shape': emoca_result['shape'], 
            'exp': emoca_result['exp'], 
            'pose': emoca_result['pose'],
            'lmks': lmks, 'lmks_dense': lmks_dense,
            'face_box': bbox.cpu().half(), 
        }
        return results
