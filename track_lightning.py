import os
import sys
import argparse
import warnings
sys.path.append('./')

def set_devices(target_device: str):
    if target_device == 'cpu' or target_device == 'mps':
        return target_device 
    os.environ['CUDA_VISIBLE_DEVICES'] = target_device
    import torch
    if torch.cuda.device_count() != 1:
        raise Exception(
            'Please assign *one* GPU: now we have {}!'.format(
                torch.cuda.device_count()
            )
        )
    return 'cuda'


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ### CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument("--device", '-d', default='cpu')
    parser.add_argument('--synthesis', action='store_true')
    parser.add_argument('--no_smooth', action='store_true')
    parser.add_argument('--smooth_type', default='kalman')
    parser.add_argument('--visualization', '-v', action='store_true')
    parser.add_argument('--visualization_fps', default=24, type=int)
    parser.add_argument('--remove_buffer', '-r', action='store_true')
    args = parser.parse_args()
    ### SET DEVICE
    target_device = set_devices(args.device)
    ### TRACK
    from core.core_engine import TrackEngine
    track_engine = TrackEngine(args, device=target_device)
    if args.remove_buffer:
        track_engine.clear_buffer()
    track_engine.run()

