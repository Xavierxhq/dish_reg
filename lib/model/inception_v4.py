import os
import torch
import torch.nn as nn
from lib.model.layer import *


class InceptionV4(nn.Module):

    def __init__(self, save_path, num_classes=1001, ):
        super(InceptionV4, self).__init__()
        self.save_path = save_path
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2),
            Conv2d(32, 32, kernel_size=3, stride=1),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        # self.last_linear = nn.Linear(1536, num_classes)
        # raise NotImplementedError()

    def forward(self, X):
        x = self.features(input)
        x = x.view(x.size(0), -1)
        return x
        # raise NotImplementedError()

    def load(self, path, is_pretrained=False):
        """
        Load parameter from path
        :param path: file path of trained model.
        :return:
        """
        state = torch.load(path)
        if is_pretrained:
            state = {k: v for k, v in state.items() if "last_linear" not in k}
        else:
            state = state["state"]
        self.load_state_dict(state)
        print("Model", path, "loaded.")
        # raise NotImplementedError()

    def save(self, epoch: int, day: int):
        """
        Save current model to path. Mind the format of the saved model tar file
        :param epoch:
        :param day:
        :return:
        """
        state = self.state_dict()
        model_state = {
            "epoch": epoch,
            "day": day,
            "state": state,
        }
        _save_name = "Day(%d)-Epoch(%d)-%s" % (
            day, epoch, os.path.basename(self.save_path))
        _save_path = os.path.join(os.path.dirname(self.save_path), _save_name)
        torch.save(model_state, _save_path)
        print("Model_state is saved to path", _save_path)
        # raise NotImplementedError()
