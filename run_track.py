import os
import sys
import argparse
sys.path.append('./')
import torch
from utils.utils import set_devices
from core.core_engine import TrackEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--tracker', default='abs')
    parser.add_argument('--save_video', '-v', action='store_true')
    parser.add_argument('--recap', '-r', action='store_true')
    parser.add_argument("--device", '-d', default='0')
    parser.add_argument("--base", default='')
    args = parser.parse_args()
    set_devices(args.device)
    track_engine = TrackEngine(args.data)
    if args.recap:
        data_name = os.path.splitext(os.path.basename(args.data))[0]
        abs_results_path = os.path.join('outputs', data_name, 'abs_results.json')
        smoothed_abs_results_path = os.path.join('outputs', data_name, 'smoothed_results.json')
        if os.path.exists(abs_results_path):
            os.remove(abs_results_path)
        if os.path.exists(smoothed_abs_results_path):
            os.remove(smoothed_abs_results_path)
    if len(args.base):
        track_engine.run_tracking_based_on(tracking_target=args.base)
    else:
        track_engine.run(save_video=args.save_video)
