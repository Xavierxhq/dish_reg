import unittest
from lib.core.config_parser import Configuration
from lib.utils.dataset_utils import *
import numpy as np


class TestDataUtils(unittest.TestCase):

    # def test_get_classid_by_filename(self):
    #     # TODO: specify your path referring to the test picture, or just name one randomly for not I/O
    #     classid = get_classid_by_filename("./lib/a/jflajfl_3.png")
    #     self.assertEqual("3", classid)

    # def test_load_image_using_opencv(self):
    #     # TODO: specify your path referring to the test picture
    #     s = load_image_using_opencv(
    #         "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png")
    #     print("image:", s)

    # def test_read_sample_by_path(self):
    #     # TODO: specify your path referring to the test picture
    #     s = read_sample_by_path(
    #         "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png")
    #     print("sample(train)", s)

    # def test_get_random_triplet(self):
    #     # TODO: specify your path referring to the test picture
    #     a = "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png"
    #     # TODO: specify your root referring to the test pictures
    #     test_root = "/home/xhq/datasets/cifar10/test"
    #     dataset = []
    #     for folder in os.listdir(test_root):
    #         for i in os.listdir(os.path.join(test_root, folder)):
    #             dataset.append(os.path.join(
    #                 test_root, folder, i))
    #     anc, pos, neg = get_random_triplet(a, dataset)
    #     print("random triplet anchor:", anc)
    #     print("random triplet positive:", pos)
    #     print("random triplet negative:", neg)

    # def test_get_hard_triplet(self):
    #     # TODO: specify your path referring to the test picture
    #     a = "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png"
    #     # TODO: specify your root referring to the test pictures
    #     test_root = "/home/xhq/datasets/cifar10/test"
    #     dataset = []
    #     dict_sample, dict_nearest = {}, {}
    #     for folder in os.listdir(test_root):
    #         dict_sample[folder] = []
    #         for i in os.listdir(os.path.join(test_root, folder)):
    #             dataset.append(os.path.join(
    #                 test_root, folder, i))
    #             dict_sample[folder].append(os.path.join(
    #                 test_root, folder, i))
    #     anc, pos, neg = get_hard_triplet(
    #         a, dataset, dict_sample, dict_nearest)
    #     print("hard triplet anchor:", anc)
    #     print("hard triplet positive:", pos)
    #     print("hard triplet negative:", neg)

    # def test_augment_image_using_imgaug(self):
    #     config = Configuration(
    #         '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
    #     # TODO: specify your path referring to the test picture
    #     s = load_image_using_opencv(
    #         "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png")
    #     s = augment_image_using_imgaug(s, config)
    #     print(s)

    # def test_augment_images_using_imgaug(self):
    #     config = Configuration(
    #         '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
    #     # TODO: specify your path referring to the test pictures
    #     s1 = load_image_using_opencv(
    #         "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png")
    #     # TODO: specify your path referring to the test pictures
    #     s2 = load_image_using_opencv(
    #         "/home/xhq/datasets/cifar10/validation/0/cifar10_0_100293.png")
    #     s = augment_images_using_imgaug([s1, s2], config)
    #     print(s)


if __name__ == '__main__':
    unittest.main()
