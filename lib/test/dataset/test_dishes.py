import os
import unittest
from torch.utils.data import DataLoader
from lib.core.config_parser import Configuration
from lib.utils.dataset_utils import Transform
from lib.dataset.dishes import Dishes


class TestDishes(unittest.TestCase):
    def test_init(self):
        config = Configuration(
            '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
        trans = Transform(32, 32)

        # training
        dataset = Dishes(config, transform=trans, training=True)
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)
        print("length:", len(loader))
        for als, pls, nls, idls in loader:
            print(type(als), len(als))

        # test
        dataset = Dishes(config, transform=trans, training=False)
        loader = DataLoader(
            dataset, batch_size=32, num_workers=32, pin_memory=True)
        print("length:", len(loader))
        for als, idls, pathls in loader:
            print(type(als), len(als))

    def test_add_extra_data(self):
        config = Configuration(
            '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
        trans = Transform(32, 32)
        extra_data = [os.path.join(config.DATA.ROOT, x)
                      for x in os.listdir(config.DATA.ROOT)[:50]]
        dataset = Dishes(config, transform=trans)
        print("Length:", len(dataset.dataset))
        print("Length:", len(dataset.anchor_set))
        dataset.add_extra_data(extra_data)
        print("Length:", len(dataset.dataset))
        print("Length:", len(dataset.anchor_set))
        print("Test adding extra data.")

    def test_add_reweight_data(self):
        config = Configuration(
            '../../../configuration/inception_v4/224x244_adam_lr1e-3.yaml')
        trans = Transform(32, 32)
        reweight_data = [os.path.join(config.DATA.ROOT, x)
                         for x in os.listdir(config.DATA.ROOT)[:50]]
        dataset = Dishes(config, transform=trans)
        print("Length:", len(dataset.dataset))
        print("Length:", len(dataset.anchor_set))
        dataset.add_reweight_data(reweight_data)
        print("Length:", len(dataset.dataset))
        print("Length:", len(dataset.anchor_set))
        print("Test adding reweight data")

    # TODO: visualization


if __name__ == '__main__':
    unittest.main()
