from torch.utils.data import Dataset
from lib.utils.dataset_utils import *


class Dishes(Dataset):
    """
    Customized data loader to load dishes data sets.
    """

    # def __init__(self):
    #     raise NotImplementedError()

    def __init__(self, config, transform=None, sample_sorted_by_distance_to_center: dict, class_to_nearest_class: dict, training=True):
        """
        Initial function of DishesDataset
        :param config: the class that contains configuration info
        :param transform: transform applied to input
        :param sample_sorted_by_distance_to_center: stores file names of samples that are sorted 
            by the distances to the corresponding center in ascending order.
        :param class_to_nearest_class: stores key-value pairs, key and value are both class ids, 
            value is the class that is closest to class key.
        """
        self.augmentation = config
        self.transform = transform
        self.sample_sorted_by_distance_to_center = sample_sorted_by_distance_to_center
        self.class_to_nearest_class = class_to_nearest_class
        self.training = training

        dataset_path = config.DATA.ROOT
        try:
            self.dataset = [os.path.join(dataset_path, x)
                            for x in os.listdir(dataset_path)]
            self.anchor_set = [
                x for x in self.dataset if "anchor" in os.path.basename(x)]
            if len(self.anchor_set) == 0:
                print(
                    "No anchor set specified, all data set will be used as anchor set.")
                self.anchor_set = self.dataset
        except Exception:
            print(dataset_path, "not found!")
        pass

    def __getitem__(self, item):
        anchor_path = self.anchor_set[item]
        anchor_id = get_classid_by_filename(anchor_path)
        if "extra" in os.path.basename(anchor_path):
            anchor_path = os.path.join(os.path.dirname(anchor_path),
                                       os.path.basename(anchor_path).replace("extra", ""))
        if "reweight" in os.path.basename(anchor_path):
            anchor_path = os.path.join(os.path.dirname(anchor_path),
                                       os.path.basename(anchor_path).replace("reweight", ""))
        if not self.training:
            anchor = read_sample_by_path(anchor_path)
            if self.transform is not None:
                anchor = self.transform(anchor)
            return anchor, anchor_id, anchor_path

        if random.randint(1, 2) % 2 == 0:
            anchor, positive, negative = get_hard_triplet(
                anchor_path, self.dataset,
                self.sample_sorted_by_distance_to_center, self.class_to_nearest_class)
        else:
            anchor, positive, negative = get_random_triplet(
                anchor_path, self.dataset)
        images = augment_images_using_imgaug(
            [anchor, positive, negative], self.augmentation)

        if self.transform is not None:
            anchor = self.transform(images[0])
            positive = self.transform(images[1])
            negative = self.transform(images[2])
        return anchor, positive, negative, anchor_id

    def __len__(self):
        return len(self.anchor_set)

    def add_extra_data(self, extra_data: list):
        """
        Add extra data to trainin set, the added data will not be used as anchor
        :param extra_data: extra data to be added
        :return:
        """
        self.dataset = [
            x for x in self.dataset if "extra" not in os.path.basename(x)]
        for i in extra_data:
            self.dataset.append(i)
            ii = os.path.basename(i)
            ii = "extra" + ii
            self.dataset.append(os.path.join(os.path.dirname(i), ii))
            pass

    def add_reweight_data(self, reweight_data: list):
        """
        Add reweight data to trainin set, the added data will be used as anchor
        :param reweight_data: reweight data to be added
        :return:
        """
        self.dataset = [
            x for x in self.dataset if "reweight" not in os.path.basename(x)]
        self.anchor_set = [
            x for x in self.anchor_set if "reweight" not in os.path.basename(x)]
        for i in reweight_data:
            ii = os.path.basename(i)
            ii = "reweight" + ii
            self.dataset.append(os.path.join(os.path.dirname(i), ii))
            self.anchor_set.append(os.path.join(os.path.dirname(i), ii))
            pass
