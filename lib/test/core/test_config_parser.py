import unittest
from lib.core.config_parser import Configuration, DataConfig, ModelConfig, TrainConfig
#  ref: https://docs.python.org/zh-cn/3/library/unittest.html


class TestConfiguration(unittest.TestCase):
    def test_init(self):
        config = Configuration('../../../resources/test.yaml')


class TestDataConfig(unittest.TestCase):
    def test_init(self):
        data_dict = {'name': 'Dishes',
                     'batch_size': 64, 'input_size': [224, 224]}
        data_config = DataConfig(data_dict)
        self.assertEqual(data_config.BATCH_SIZE, data_dict['batch_size'])


if __name__ == '__main__':
    unittest.main()
