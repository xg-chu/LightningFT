import torch
import numpy as np
import torchvision

from model.EMOCA import EMOCA

class Emoca_V2_Engine:
    def __init__(self, emoca_ckpt_path, device='cuda', lazy_init=True):
        self._emoca_ckpt_path = emoca_ckpt_path
        self._device = device
        if not lazy_init:
            self._init_model()

    def _init_model(self, ):
        print('Initializing trimmed EMOCA V2 models...')
        ckpt = torch.load(self._emoca_ckpt_path, map_location='cpu')['state_dict']
        trimmed_ckpt = {}
        for key in list(ckpt.keys()):
            if 'E_flame' in key or 'E_expression' in key:
                trimmed_ckpt[key.replace('deca.', '')] = ckpt[key]

        emoca_model = EMOCA().to(self._device)
        emoca_model.eval()
        emoca_model.load_state_dict(trimmed_ckpt, strict=True)
        self.emoca_model = emoca_model
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

    def process_face(self, image, landmarks):
        if not hasattr(self, 'emoca_model'):
            self._init_model()
        if landmarks is None:
            return None
        else:
            croped_frame, crop_center = self._crop_frame(image, landmarks)
            # please input normed image
            croped_frame = croped_frame.to(self._device)[None]/255.0
            emoca_result = self.emoca_model.encode(croped_frame)
            bbox = torch.stack([
                crop_center[0] - crop_center[2]/2, crop_center[1] - crop_center[2]/2,
                crop_center[0] + crop_center[2]/2, crop_center[1] + crop_center[2]/2,
            ])
            emoca_result['crop_box'] = bbox.cpu().half()
            return emoca_result
