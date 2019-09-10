import os
import random
from PIL import Image
import numpy as np
import torch
import cv2
from torchvision import transforms as T
import imgaug as ia
from imgaug import augmenters as iaa


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


# TODO: analyze and get information for hard triplet samples
