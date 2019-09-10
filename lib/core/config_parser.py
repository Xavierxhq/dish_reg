import yaml  # pip install pyyaml
import os


class Configuration(object):
    def __init__(self, yaml_path):
        """
        Parse yaml configuration file.
        :param yaml_path: Please be noticed that yaml name is meaningful, new a name carefully.
        """
        self.CONFIG_NAME = os.path.basename(yaml_path).split('.')[0]
        with open(yaml_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        self.DATA = DataConfig(data_loaded['DATA'])
        self.TRAIN = TrainConfig(data_loaded['TRAIN'])
        self.MODEL = ModelConfig(data_loaded['MODEL'])


class DataConfig(object):
    def __init__(self, data_dict):
        self.NAME = data_dict['NAME']
        self.ROOT = data_dict['ROOT']
        self.BATCH_SIZE = data_dict['BATCH_SIZE']
        self.INPUT_SIZE = data_dict['INPUT_SIZE']
        self.ROTATE_FACTOR = data_dict["ROTATE_FACTOR"]  # added by xhq
        self.SCALE_X_FACTOR = data_dict["SCALE_X_FACTOR"]  # added by xhq
        self.SCALE_Y_FACTOR = data_dict["SCALE_Y_FACTOR"]  # added by xhq
        # added by xhq
        self.TRANSLATE_X_FACTOR = data_dict["TRANSLATE_X_FACTOR"]
        # added by xhq
        self.TRANSLATE_Y_FACTOR = data_dict["TRANSLATE_Y_FACTOR"]
        self.SHEAR_FACTOR = data_dict["SHEAR_FACTOR"]  # added by xhq
        self.FLIP_L_R_FACTOR = data_dict["FLIP_L_R_FACTOR"]  # added by xhq
        self.FLIP_U_D_FACTOR = data_dict["FLIP_U_D_FACTOR"]  # added by xhq
        self.MULTIPLY_FACTOR = data_dict["MULTIPLY_FACTOR"]  # added by xhq


class TrainConfig(object):
    def __init__(self, data_dict):
        self.N_DAYS = data_dict['num_days']
        self.OPTIM_NAME = data_dict['optim_name']
        self.SCHEDULER_NAME = data_dict['scheduler_name']
        self.LOSS_NAME = data_dict['loss_name']
        self.LR = data_dict['lr']
        self.EPOCHS = data_dict['epochs']
        self.MARGIN = data_dict['MARGIN']

        self.MODEL_SAVE_PATH = data_dict['model_save_path']
        self.LOG_SAVE_PATH = data_dict['log_save_path']

        self.CONFIDENCE = data_dict['confidence']


class ModelConfig(object):
    def __init__(self, data_dict):
        self.NAME = data_dict['NAME']
        self.N_CLASSES = data_dict['NUM_CLASSES']
        self.PRETRAINED = data_dict['PRETRAINED']  # added by xhq
        self.SAVE_PATH = data_dict['SAVE_PATH']  # added by xhq


class LoggerConfig(object):
    def __init__(self, data_dict):
        self.DIR = data_dict['dir']


if __name__ == "__main__":
    pass
