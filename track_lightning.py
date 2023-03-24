import os
import sys
import argparse
sys.path.append('./')

def set_devices(target_device: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = target_device
    import torch
    if torch.cuda.device_count() != 1:
        raise Exception(
            'Please assign *one* GPU: now we have {}!'.format(
                torch.cuda.device_count()
            )
        )


if __name__ == "__main__":
    ### CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument("--device", '-d', default='0')
    parser.add_argument('--visualization', '-v', action='store_true')
    parser.add_argument('--remove_buffer', '-r', action='store_true')
    args = parser.parse_args()
    ### SET DEVICE
    set_devices(args.device)
    ### TRACK
    from core.core_engine import TrackEngine
    track_engine = TrackEngine(args.data)
    if args.remove_buffer:
        track_engine.clear_buffer()
    track_engine.run_tracking(save_video=args.save_video)

