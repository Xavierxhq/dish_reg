import pickle
import os
import torch


# 类转dict
def object2dict(obj):
    dict = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            dict[name] = value
    return dict


def pickle_read(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        print('Pickle read error: not exits {}'.format(file_path))
        return None


def pickle_write(file_path, what_to_write):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    try:
        with open(file_path, 'wb+') as f:
            pickle.dump(what_to_write, f)
    except:
        print('Pickle write error: {}'.format(file_path))


def dist(y1, y2):
    return torch.sqrt(torch.sum(torch.pow(y1.cpu() - y2.cpu(), 2))).item()
