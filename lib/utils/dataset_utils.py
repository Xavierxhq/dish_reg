import os
import random
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms as T
import imgaug as ia
from imgaug import augmenters as iaa
from lib.utils.common_util import pickle_read, pickle_write, dist
import time
import shutil


def get_classid_by_filename(filename):
    """
    Return the class id of the sample according to the given file name, 
        based on the rule we set to name a sample.
    :param filename: the filename of the sample (or the full path of the sample).
    :return: the class id of the sample named after filename.
    """
    filename = os.path.basename(filename)
    return filename.split('_')[-1].split('.')[0]


def load_image_using_pil(path):
    """
    (Desparated.)
    Read an image by the given path.
    :param path: the full path of the picture to be read
    :return: the loaded image in np.array form
    """
    img = None
    # Keep reading image until succeed, this can avoid IOError incurred by heavy IO process.
    got_img = False
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will try again.".format(path))
            exit(-1)
    return np.array(img)


def load_image_using_opencv(path):
    """
    Read an image by the given path.
    :param path: the full path of the picture to be read
    :return: the loaded image in np.array form
    """
    img = None
    # Keep reading image until succeed, this can avoid IOError incurred by heavy IO process.
    got_img = False
    while not got_img:
        try:
            img = cv2.imread(path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will try again.".format(path))
            exit(-1)
    return img


def bgr2rgb_using_opencv(image):
    """
    Convert the image (from bgr to rgb)
    :param image: the image to be converted
    :return: the converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def augment_image_using_imgaug(image, config):
    """
    Augment an image using imgaug lib
    :param image: the image to be augmented, in np.array form
    :return: the augmented image
    """
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-config.DATA.ROTATE_FACTOR, config.DATA.ROTATE_FACTOR),
            scale={
                "x": (1.0 - config.DATA.SCALE_X_FACTOR, 1.0 + config.DATA.SCALE_X_FACTOR),
                "y": (1.0 - config.DATA.SCALE_Y_FACTOR, 1.0 + config.DATA.SCALE_Y_FACTOR)
            },
            translate_percent={
                "x": (-config.DATA.TRANSLATE_X_FACTOR, config.DATA.TRANSLATE_X_FACTOR),
                "y": (-config.DATA.TRANSLATE_Y_FACTOR, config.DATA.TRANSLATE_Y_FACTOR)
            },
            shear=(-config.DATA.SHEAR_FACTOR, config.DATA.SHEAR_FACTOR),
            mode=ia.ALL,
        ),
        iaa.Fliplr(config.DATA.FLIP_L_R_FACTOR),
        iaa.Flipud(config.DATA.FLIP_U_D_FACTOR),
        iaa.Multiply((1.0 - config.DATA.MULTIPLY_FACTOR,
                      1.0 + config.DATA.MULTIPLY_FACTOR)),
    ], random_order=True)
    image = seq.augment_image(image)
    return image
    pass


def augment_images_using_imgaug(images: list, config):
    """
    Augment images using imgaug lib
    :param images: the images to be augmented, in np.array form
    :return: the augmented images
    """
    seq = iaa.Sequential([
        iaa.Affine(
            rotate=(-config.DATA.ROTATE_FACTOR, config.DATA.ROTATE_FACTOR),
            scale={
                "x": (1.0 - config.DATA.SCALE_X_FACTOR, 1.0 + config.DATA.SCALE_X_FACTOR),
                "y": (1.0 - config.DATA.SCALE_Y_FACTOR, 1.0 + config.DATA.SCALE_Y_FACTOR)
            },
            translate_percent={
                "x": (-config.DATA.TRANSLATE_X_FACTOR, config.DATA.TRANSLATE_X_FACTOR),
                "y": (-config.DATA.TRANSLATE_Y_FACTOR, config.DATA.TRANSLATE_Y_FACTOR)
            },
            shear=(-config.DATA.SHEAR_FACTOR, config.DATA.SHEAR_FACTOR),
            mode=ia.ALL,
        ),
        iaa.Fliplr(config.DATA.FLIP_L_R_FACTOR),
        iaa.Flipud(config.DATA.FLIP_U_D_FACTOR),
        iaa.Multiply((1.0 - -config.DATA.MULTIPLY_FACTOR,
                      1.0 + config.DATA.MULTIPLY_FACTOR)),
    ], random_order=True)
    images = seq.augment_images(images)
    return images
    pass


def read_sample_by_path(path):
    """
    Read sample (picture form) by the given path.
    :param path: the full path of the sample to be read.
    :return: the loaded and transformed sample (in Tensor form).
    """
    sample = load_image_using_opencv(path)
    return sample


def get_random_triplet(anchor_path, dataset):
    """
    generate a triplet randomly, select positive and negative from the whole 
        data set randomly (class id is still considered).
    :param anchor_path: the path of the anchor.
    :param dataset: the whole dataset, elements are full paths of the samples.
    :return: the generated random triplet.
    """
    anchor_data = read_sample_by_path(anchor_path)

    anchor_id = get_classid_by_filename(anchor_path)
    positive_candidate_paths = [
        x for x in dataset if get_classid_by_filename(x) == anchor_id]
    random.shuffle(positive_candidate_paths)
    positive_path = positive_candidate_paths[0]
    positive_data = read_sample_by_path(positive_path)

    negative_candidate_paths = [
        x for x in dataset if get_classid_by_filename(x) != anchor_id]
    random.shuffle(negative_candidate_paths)
    negative_path = negative_candidate_paths[0]
    negative_data = read_sample_by_path(negative_path)
    return anchor_data, positive_data, negative_data


def get_hard_triplet(anchor_path, dataset, sample_sorted_by_distance_to_center: dict, class_to_nearest_class: dict):
    """
    generate a hard triplet.
    :param anchor_path: the path of the anchor.
    :param dataset: the whole dataset, elements are full paths of the samples.
    :param sample_sorted_by_distance_to_center: stores file names of samples that are sorted 
        by the distances to the corresponding center in ascending order.
    :param class_to_nearest_class: stores key-value pairs, key and value are both class ids, 
        value is the class that is closest ot class key.
    :return: the generated hard triplet.
    """
    anchor_data = read_sample_by_path(anchor_path)

    anchor_id = get_classid_by_filename(anchor_path)

    if anchor_id in sample_sorted_by_distance_to_center:
        random_int = random.randint(int(len(sample_sorted_by_distance_to_center[anchor_id]) / 2),
                                    len(sample_sorted_by_distance_to_center[anchor_id]) - 1)
        positive_path = sample_sorted_by_distance_to_center[anchor_id][random_int][0]
    else:
        positive_candidate_paths = [
            x for x in dataset if get_classid_by_filename(x) == anchor_id]
        random.shuffle(positive_candidate_paths)
        positive_path = positive_candidate_paths[0]
    positive_data = read_sample_by_path(positive_path)

    if anchor_id in class_to_nearest_class:
        negative_candidate_paths = [x for x in dataset if get_classid_by_filename(
            x) == class_to_nearest_class[anchor_id]]
    else:
        negative_candidate_paths = [
            x for x in dataset if get_classid_by_filename(x) != anchor_id]
    random.shuffle(negative_candidate_paths)
    negative_path = negative_candidate_paths[0]
    negative_data = read_sample_by_path(negative_path)
    return anchor_data, positive_data, negative_data


class Transform(object):

    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w

    def __call__(self, x):
        x = cv2.resize(x, (self.h, self.w))
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x


# analyze and get information for hard triplet samples
def analyze_and_get_hard_samples(model, analysis_set, test_pictures = None):
    if test_pictures == None:
        test_pictures = os.listdir(analysis_set)

    # _get_avg_feature_for_all(model, test_pictures = test_pictures)

    all_test_pkls_dir = 'tmp/training/%s_all_test_pkls' % 'tableware'.split('/')[0]
    feature_map = _calc_true_avg_feature(all_test_pkls_dir)
    sample_sorted_by_distance_to_center = _calc_inter_distance(all_test_pkls_dir)
    _, class_to_nearest_class = _calc_exter_class_distance(feature_map)

    for classid, d in sample_sorted_by_distance_to_center.items():
        _d = sorted(d.items(), key=lambda x: x[1])
        sample_sorted_by_distance_to_center[classid] = _d[-40:]
    return sample_sorted_by_distance_to_center, class_to_nearest_class


def _calc_true_avg_feature(feature_pkls_dir):
    t1 = time.time
    feature_pkls = [x for x in os.listdir(
        feature_pkls_dir) if 'features.pkl' in x]
    feature_map = {}
    for pkl in feature_pkls:
        features = pickle_read(os.path.join(feature_pkls_dir, pkl))
        features = list(features.values())
        # _avg_feature = np.zeros(shape=features[0].shape)
        _avg_feature = None
        for _feature in features:
            _feature = _feature.cpu().detach().numpy()
            # print("shape 0", _feature.shape)
            if _avg_feature is None:
                _avg_feature = _feature
                # print("shape 0 avg", _avg_feature.shape)
                continue
            # print("shape", _avg_feature.shape, _feature.shape)
            _avg_feature += _feature
        _avg_feature /= len(features)
        classid = pkl.split('_')[0]
        feature_map[classid] = torch.Tensor(_avg_feature)

    pickle_write('tmp/training/%s_true_avg_feature_for_each_class.pkl' %
                 'tableware'.split('/')[0], feature_map)
    print('Time for _calc_true_avg_feature: %.1f s' % (time.time() - t1))
    return feature_map

def _calc_exter_class_distance(feature_map):
    t1 = time.time()
    id_feature_ls = [(_id, _feature)
                     for _id, _feature in feature_map.items()]
    exter_class_distance_dict, class_to_nearest_class = {}, {}
    for _i in range(len(id_feature_ls)):
        classid, feature = id_feature_ls[_i]
        nearest_id, neareast_d = None, 1e6
        for _second_classid, _second_feature in id_feature_ls[_i+1:]:
            _d = dist(feature, _second_feature)
            _key = classid + '-' + _second_classid
            exter_class_distance_dict[_key] = _d
            if neareast_d > _d:
                neareast_d = _d
                nearest_id = _second_classid
        class_to_nearest_class[classid] = nearest_id

    pickle_write('tmp/training/%s_exter_class_distances.pkl' %
                 'tableware'.split('/')[0], exter_class_distance_dict)
    print('Time for _calc_exter_class_distance: %.1f s' %
          (time.time() - t1))
    return exter_class_distance_dict, class_to_nearest_class

def _calc_inter_distance(feature_map_dir, feature_map=None):
    t1 = time.time()
    distance_dict = {}

    if feature_map is None:
        feature_map = pickle_read(
            'tmp/training/%s_true_avg_feature_for_each_class.pkl' % 'tableware'.split('/')[0])
    for pkl in [x for x in os.listdir(feature_map_dir) if 'features.pkl' in x]:
        classid = pkl.split('_')[0]
        distance_dict[classid] = {}
        for _filename, _feature in pickle_read(os.path.join(feature_map_dir, pkl)).items():
            distance_dict[classid][_filename] = dist(
                feature_map[classid], _feature)
    pickle_write('tmp/training/%s_inter_class_distances.pkl' %
                 'tableware'.split('/')[0], distance_dict)
    print('Time for _calc_inter_distance: %.1f s' % (time.time() - t1))
    return distance_dict


def get_hard_pictures(all_pictures, limit=100):
    hard_pictures = []
    for i in range(78):
        pre = "_%d." % i
        _ls = [x for x in all_pictures if pre in os.path.basename(x)]
        hard_pictures.extend(_ls[:limit])
    return hard_pictures