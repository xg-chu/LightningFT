import os
import json
# import yaml
import colored
from colored import stylize

def set_devices(target_device: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = target_device
    import torch
    if torch.cuda.device_count() != 1:
        raise Exception(
            'Please assign *one* GPU: now we have {}!'.format(
                torch.cuda.device_count()
            )
        )


def read_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def read_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} was not found.")
    with open(path) as f:
        json_dict = json.load(f)
    return json_dict


def save_json(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f)


def pretty_dict(input_dict, indent=0, highlight_keys=[]):
    out_line = ""
    tab = "    "
    for key, value in input_dict.items():
        if key in highlight_keys:
            out_line += tab * indent + stylize(str(key), colored.fg(1))
        else:
            out_line += tab * indent + stylize(str(key), colored.fg(2))
        if isinstance(value, dict):
            out_line += ':\n'
            out_line += pretty_dict(value, indent+1, highlight_keys)
        else:
            if key in highlight_keys:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(1)) + '\n'
            else:
                out_line += ":" + "\t" + stylize(str(value), colored.fg(2)) + '\n'
    if indent == 0:
        max_length = 0
        for line in out_line.split('\n'):
            max_length = max(max_length, len(line.split('\t')[0]))
        max_length += 4
        aligned_line = ""
        for line in out_line.split('\n'):
            if '\t' in line:
                aligned_number = max_length - len(line.split('\t')[0])
                line = line.replace('\t',  aligned_number * ' ')
            aligned_line += line+'\n'
        return aligned_line[:-2]
    return out_line


def list_all_files(dir_path):
    pair = os.walk(dir_path)
    result = []
    for path, dirs, files in pair:
        if len(files):
            for file_name in files:
                result.append(os.path.join(path, file_name))
    return result
