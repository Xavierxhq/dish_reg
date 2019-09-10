import unittest
from lib.model.inception_v4 import InceptionV4


class TestInceptionV4(unittest.TestCase):
    def test_init(self):
        model = InceptionV4("test_model_state.pth")
        model = model.cuda()
        model.save(0, 0)


if __name__ == '__main__':
    unittest.main()
