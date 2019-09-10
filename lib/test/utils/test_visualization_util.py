import unittest
from lib.utils.visualization_util import *
import datetime
from lib.utils.dataset_utils import *
from lib.core.config_parser import Configuration
# from lib.utils.common_util import object2dict
from matplotlib.pylab import plt, savefig


class TestVisualizationUtils(unittest.TestCase):

    def test_augmentation_visualize_and_save(self):
        # init config
        config = Configuration(
            '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
        # config = object2dict(config)

        # get test images
        images = []
        images_names = []
        # TODO: specify your root for storing test pictures
        root = "/home/xhq/datasets/test/"
        for filename in os.listdir(root):
            if os.path.isdir(os.path.join(root, filename)):
                continue
            _image = load_image_using_opencv(os.path.join(root, filename))
            images.append(_image)
            images_names.append(filename)

        # augment images and visualization
        # TODO: specify your root for storing test pictures
        augmentation_visualize_and_save(
            config, images, images_names, "/home/xhq/datasets/test/augmented/", 2)


if __name__ == '__main__':
    unittest.main()
