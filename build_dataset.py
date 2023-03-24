import os
import sys
import lmdb
import random
import argparse
sys.path.append('./')
from tqdm import tqdm

from utils.utils import read_json, save_json, list_all_files

### json
def build_json(target_path, train_json, test_json):
    train_dict = read_json(train_json)
    train_records = {'records': [], 'meta_info':train_dict['meta_info']}
    for f_idx in range(len(train_dict.keys())-1):
        ori_frame_name = 'f_{:05d}.jpg'.format(f_idx)
        train_frame_name = 'train/f_{:05d}.jpg'.format(f_idx)
        train_dict[ori_frame_name]['file_path'] = train_frame_name
        train_records['records'].append(train_dict[ori_frame_name])

    test_dict = read_json(test_json)
    test_records = {'records': [], 'meta_info':test_dict['meta_info']}
    test_records = {'records': [], 'meta_info':test_dict['meta_info']}
    for f_idx in range(len(test_dict.keys())-1):
        ori_frame_name = 'f_{:05d}.jpg'.format(f_idx)
        test_frame_name = 'test/f_{:05d}.jpg'.format(f_idx)
        test_dict[ori_frame_name]['file_path'] = test_frame_name
        test_records['records'].append(test_dict[ori_frame_name])

    val_records = {
        'records': random.choices(test_records['records'], k=5), 
        'meta_info':test_records['meta_info']
    }

    results = {
        'train':train_records, 'val':val_records, 'test':test_records
    }
    save_json(results, target_path)

### lmdb
def build_lmdb(target_path, txn_train, num_train, txn_test, num_test):
    os.makedirs(target_path, exist_ok=False)
    env = lmdb.open(target_path, map_size=1099511627776) # Maximum 1T
    txn = env.begin(write=True)
    counter = 0
    for f_idx in tqdm(range(num_train), ncols=80, colour='#95bb72'):
        ori_img_name = 'f_{:05d}.jpg'.format(f_idx)
        train_image_name = 'train/f_{:05d}.jpg'.format(f_idx)
        image_buf = txn_train.get(ori_img_name.encode())
        txn.put(train_image_name.encode(), image_buf)
        counter += 1
        if counter % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    for f_idx in tqdm(range(num_test), ncols=80, colour='#95bb72'):
        ori_img_name = 'f_{:05d}.jpg'.format(f_idx)
        test_image_name = 'test/f_{:05d}.jpg'.format(f_idx)
        image_buf = txn_test.get(ori_img_name.encode())
        txn.put(test_image_name.encode(), image_buf)
        counter += 1
        if counter % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()

def read_lmdb(path):
    _lmdb_env = lmdb.open(
        path, readonly=True, lock=False, readahead=False, meminit=True
    ) 
    _lmdb_txn = _lmdb_env.begin(write=False)
    _num_frames = len(list(_lmdb_txn.cursor().iternext(values=False)))
    return _lmdb_txn, _num_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_bg', default='')
    parser.add_argument('--auto', default='')
    parser.add_argument('--train', default='')
    parser.add_argument('--test', default='')
    args = parser.parse_args()
    if len(args.add_bg):
        import torch
        import torchvision
        env = lmdb.open(os.path.join('./outputs/auto_dataset', 'lmdb'), map_size=1099511627776) # Maximum 1T
        txn = env.begin(write=True)
        img_tensor = torchvision.io.read_image(args.add_bg, mode=torchvision.io.ImageReadMode.RGB)
        img_encoded = torchvision.io.encode_jpeg(img_tensor.to(torch.uint8))
        img_encoded = b''.join(map(lambda x:int.to_bytes(x,1,'little'), img_encoded.numpy().tolist()))
        txn.put('background.jpg'.encode(), img_encoded)
        txn.commit()
        env.close()

        data_json = read_json(os.path.join('./outputs/auto_dataset', 'dataset.json'))
        data_json['train']['meta_info']['background'] = 'background.jpg'
        data_json['val']['meta_info']['background'] = 'background.jpg'
        data_json['test']['meta_info']['background'] = 'background.jpg'
        save_json(data_json,'./outputs/auto_dataset/dataset.json')
        sys.exit()


    if len(args.auto):
        pass
    else:
        assert len(args.train) and len(args.test)
        # merge lmdb
        txn_train, num_train = read_lmdb(os.path.join(args.train, 'lmdb'))
        txn_test, num_test = read_lmdb(os.path.join(args.test, 'lmdb'))
        build_lmdb('./outputs/auto_dataset/lmdb', txn_train, num_train, txn_test, num_test)
        # merge json
        build_json(
            './outputs/auto_dataset/dataset.json', 
            os.path.join(args.train, 'smoothed_results.json'),
            os.path.join(args.test, 'smoothed_results.json')
        )


