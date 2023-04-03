# LightningFT
Tracking faces as fast as lightning!

## Preparation
### 1. Prepare for EMOCA v2
```
mkdir -p assets/EMOCA
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip -O assets/EMOCA/EMOCA_v2_lr_mse_20.zip
unzip assets/EMOCA/EMOCA_v2_lr_mse_20.zip -d assets/EMOCA
rm assets/EMOCA/EMOCA_v2_lr_mse_20.zip
```
### 2. Prepare for FLAME
```
TODO
```

## Requirement
* python >= 3.8
* pytorch==1.13.0
* pytorch3d==0.7.2 (Following [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md))
* [mediapipe](https://google.github.io/mediapipe/)
* [face_alignment](https://github.com/1adrianb/face-alignment)
* tqdm, rich, lmdb, colored, pykalman

## Usage
Our tracker firstly tracks the expression and pose of the target head based on EMOCA v2, then optimize the camera pose based on landmarks (face_alignment and mediapipe).
Please refer to the following command to run the tracker.
```
python track_lightning.py --data video_path -d gpu_id -v
```


## Tips
* The camera is calibrated based on 32 randomly sampled frames.
* The running speed is greatly affected by the iteration number of landmark optimization.
* Tracking performance is greatly affected by face_alignment and landmarks detected by mediapipe.
