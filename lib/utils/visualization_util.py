from matplotlib.pylab import plt, savefig
from lib.utils.dataset_utils import *
import datetime
import os


def augmentation_visualize_and_save(config, images, images_names, path, times: int = 2):
    """
    Visualization of image enhancements.
    :param config: configuration from yaml file.
    :param images: images to be augmented.
    :param images_names: corresponding names of the images.
    :param path: the root where the augmented pictures will be saved.
    :param times: how many times each image getting augmented.
    :return:
    """

    rows = len(images)
    cols = times + 1
    for (index, image), name in zip(enumerate(images), images_names):
        plt.subplot(rows, cols, index * cols + 1)
        plt.axis('off')
        plt.title(name)
        _image = bgr2rgb_using_opencv(image)
        plt.imshow(_image)
        for col in range(1, cols):
            plt.subplot(rows, cols, index * cols + col + 1)
            plt.axis('off')
            plt.title("Augmented NO. " + str(col))
            # augment image
            augmented_image = augment_image_using_imgaug(_image, config)
            plt.imshow(augmented_image)

    # Save the full figure
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    savefig(os.path.join(path, "%s_Comp.png" % now_time), dpi=600)
    # Clear the current figure
    plt.clf()
    plt.cla()
    plt.close()
