import os
import sys
import torch
import random
import shutil
import argparse
sys.path.append('./')

def build_dataset_pth(data_path, val_length=5, test_length=1000):
    # dir_path = os.path.dirname(data_path)
    original_dict = torch.load(data_path, map_location='cpu')
    all_keys = list(original_dict.keys())
    total_length, train_frames, val_frames, test_frames = get_length(all_keys, val_length, test_length)
    print(f'Found dataset length: {total_length}.')
    print(f'Train / Val / Test length: {len(train_frames)} / {len(val_frames)} / {len(test_frames)}.')
    dataset_dict = {
        'train': build_records(original_dict, train_frames), 
        'val': build_records(original_dict, val_frames), 
        'test': build_records(original_dict, test_frames)
    }
    return dataset_dict


def build_records(original_dict, target_frames):
    records = {'records': [], 'meta_info':original_dict['meta_info']}
    for frame_key in target_frames:
        this_frame_dict = {'file_path': frame_key}
        for key in original_dict[frame_key].keys():
            this_frame_dict[key] = original_dict[frame_key][key].detach().half()
        records['records'].append(this_frame_dict)
    return records


def get_length(key_list, val_length, test_length):
    # find total length
    max_frame_idx = 0
    key_list = [key for key in key_list if key[0]=='f']
    for frame_key in key_list:
        max_frame_idx = max(max_frame_idx, int(frame_key[2:].split('.')[0]))
    total_length = max_frame_idx + 1
    # split dataset
    random.seed(42)
    train_keys, test_keys = [], []
    for frame_key in key_list:
        frame_idx = int(frame_key[2:].split('.')[0])
        if frame_idx < total_length - test_length:
            train_keys.append(frame_key)
        else:
            test_keys.append(frame_key)
    val_keys = random.choices(key_list, k=val_length)
    return total_length, train_keys, val_keys, test_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='')
    parser.add_argument('--add_bg', default='')
    args = parser.parse_args()
    dataset_dict = build_dataset_pth(args.data)
    if len(args.add_bg):
        ext_name = os.path.basename(args.add_bg).split(".")[-1]
        shutil.copyfile(args.add_bg, os.path.join(os.path.dirname(args.data), f'background.{ext_name}'))
        dataset_dict['train']['meta_info']['background'] = f'background.{ext_name}'
        dataset_dict['val']['meta_info']['background'] = f'background.{ext_name}'
        dataset_dict['test']['meta_info']['background'] = f'background.{ext_name}'
    torch.save(dataset_dict, os.path.join(os.path.dirname(args.data), 'dataset.pth'))

