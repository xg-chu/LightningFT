# LightningFT
Tracking faces as fast as lightning!

## Preparation
### 1. Prepare for EMOCA v2
```
mkdir -p assets/EMOCA
wget https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip -O assets/EMOCA_v2_lr_mse_20.zip
unzip assets/EMOCA_v2_lr_mse_20.zip -d assets/EMOCA
rm assets/EMOCA_v2_lr_mse_20.zip
```
### 2. Prepare for FLAME
```
wget --post-data "username=[username]&password=[password]" https://download.is.tue.mpg.de/download.php\?domain\=flame\&sfile\=FLAME2020.zip\&resume\=1 -O 'assets/FLAME2020.zip' --no-check-certificate
wget --post-data "username=[username]&password=[password]" https://download.is.tue.mpg.de/download.php\?domain\=flame\&resume\=1\&sfile\=TextureSpace.zip -O 'assets/TextureSpace.zip' --no-check-certificate
mkdir -p assets/FLAME
unzip assets/FLAME2020.zip -d assets/FLAME
rm assets/FLAME2020.zip
mv assets/FLAME/Readme.pdf assets/FLAME/Readme_FLAME.pdf
unzip assets/TextureSpace.zip -d assets/FLAME
rm assets/TextureSpace.zip
mv assets/FLAME/Readme.pdf assets/FLAME/Readme_Texture.pdf
unzip assets/FLAME_embedding.zip -d assets/FLAME
```

## Requirement
* python >= 3.8
* pytorch ==2.0
* pytorch3d == 0.7.3 (Following [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md))
* [mediapipe](https://google.github.io/mediapipe/)
* [face_alignment](https://github.com/1adrianb/face-alignment)
* ffmpeg >= 5.1 (conda install -c conda-forge ffmpeg)
* pip install tqdm rich lmdb colored pykalman av pytorch-lightning chumpy

## Usage
Our tracker firstly tracks the expression and pose of the target head based on EMOCA v2, then optimize the camera pose based on landmarks (face_alignment and mediapipe).

Please refer to the following command to run the tracker.
```
python track_lightning.py -d 7 --data ./assets/demo.mp4 -v --synthesis
```


## Tips
* The camera is calibrated based on 32 randomly sampled frames.
* The running speed is greatly affected by the iteration number of landmark optimization.
* Tracking performance is greatly affected by face_alignment and landmarks detected by mediapipe.
