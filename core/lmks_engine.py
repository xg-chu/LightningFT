import torch
import mediapipe
import numpy as np
import face_alignment

class LandmarksEngine:
    def __init__(self, device='cuda', lazy_init=True):
        self._device = device
        if not lazy_init:
            self._init_model()

    def _init_model(self, ):
        print('Initializing landmark models...')
        self.lmks_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self._device)
        self.lmks_dense_model = mediapipe.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        print('Done.')

    def process_face(self, image):
        if not hasattr(self, 'lmks_model'):
            self._init_model()
        # face alignment
        image = image.permute(1, 2, 0)
        lmks, scores, detected_faces = self.lmks_model.get_landmarks_from_image(
            image, return_landmark_score=True, return_bboxes=True
        )
        lmks = None if detected_faces is None else torch.tensor(lmks[0]).half()
        # mediapipe
        lmks_dense = self.lmks_dense_model.process(image.numpy())
        if lmks_dense.multi_face_landmarks is None:
            lmks_dense = None
        else:
            lmks_dense = lmks_dense.multi_face_landmarks[0].landmark
            lmks_dense = np.array(list(map(lambda l: np.array([l.x, l.y]), lmks_dense)))
            lmks_dense[:, 0] = lmks_dense[:, 0] * image.shape[1]
            lmks_dense[:, 1] = lmks_dense[:, 1] * image.shape[0]
            lmks_dense = torch.tensor(lmks_dense).half()
        return {'lmks': lmks, 'lmks_dense': lmks_dense}
