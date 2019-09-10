import logging
import pathlib
import time
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import *
from lib.model.loss import *
from lib.model.inception_v4 import InceptionV4
from lib.dataset.dishes import Dishes


def create_data_loader(config, dataset=None, extra_data=None, training=True):
    if dataset is not None:
        if extra_data is not None:
            dataset.add_extra_data(extra_data)
        else:
            raise ValueError('dataset is not None but extra_data is None!')
    else:
        # TODO: do analysis, get hard positives and hard negatives

        # TODO: pass the correct parameters to generate dataset, mind the mode: training
        dataset = eval(config.DATA.NAME)(config)  # e.g. eval('Dished')()
    dataloader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE,
                            shuffle=training, pin_memory=True, num_workers=int(config.DATA.BATCH_SIZE / 2))
    return dataloader, dataset


def create_model(config):
    model = eval(config.MODEL.NAME)(
        config.MODEL.SAVE_PATH, config.MODEL.N_CLASSES)  # e.g. eval('InceptionV4')()
    return model


def create_loss_fn(config):
    loss_fn = eval(config.TRAIN.LOSS_NAME)(
        config.TRAIN.MARGIN)  # e.g. eval('TripletLoss')()
    return loss_fn


def create_optim_and_scheduler(config, model):
    # Use if ... else ... statement if necessary
    optimizer = eval(config.TRAIN.OPTIM_NAME)(
        model.parameters(), lr=config.TRAIN.LR)
    scheduler = eval(config.TRAIN.SCHEDULER_NAME)(optimizer, )
    return optimizer, scheduler


def create_logger(config):
    root_output_dir = pathlib.Path(config.TRAIN.LOG_SAVE_PATH)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(config.CONFIG_NAME, time_str)
    final_log_file = root_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


def train_epoch(train_loader, model, loss_fn, optimizer, logger):
    """
    One epoch
    :param train_loader:
    :param model: Should be in train mode.
    :param loss_fn:
    :param optimizer:
    :param logger:

    :return: epoch_train_loss
    """
    epoch_train_loss = []
    for anchors, positives, negatives, y in train_loader:
        anchors = model(anchors)
        positives = model(positives)
        negatives = model(negatives)
        loss = loss_fn(anchors, positives, negatives)
        optimizer.zero_grad()
        loss.backward()
        epoch_train_loss.append(loss.numpy())
        optimizer.step()

    # TODO: add logging. Better use logger instead of direct print.
    # e.g.  logger.info('training loss {} on iter {}'.format(...))
    return np.mean(epoch_train_loss)


def test_epoch(test_loader, model, loss_fn):
    """
    One epoch
    :param test_loader:
    :param model:
    :param loss_fn:
    :return: epoch_test_loss
    """
    model.eval()
    epoch_test_loss = []
    for X, y, paths in test_loader:
        pred = model(X)
        loss = loss_fn(pred, y)  # TODO: add accuracy function
        epoch_test_loss.append(loss.numpy())
    model.train()

    # TODO: add logging. Better use logger instead of direct print.
    # e.g.  logger.info('training loss {} on iter {}'.format(...))

    return np.mean(epoch_test_loss)


def train(config, train_loader, test_loader, model, loss_fn, optimizer, scheduler, logger, day, min_improvement=0.95):
    best_loss = None
    logger.info('train on day {}'.format(day))
    for epoch in range(config.TRAIN.EPOCHS):
        train_loss = train_epoch(
            train_loader, model, loss_fn, optimizer, logger)
        test_loss = test_epoch(test_loader, model, loss_fn)
        if best_loss is not None:
            if best_loss * min_improvement > test_loss:
                model.save(epoch, day)
        else:
            best_loss = test_loss
        scheduler.step()


def online_data_selection(model, images, thresh):
    """
    Select new data that are going to be used for online-learning
    :param model: the model used for feature extraction
    :param images: unseen images, where we will select some for online-learning
    :param thresh: a threshold that enforces the selected unseen samples are over a certain confidence
    :return: the selected unseen samples in list form
    """
    model.eval()

    # TODO: to select new data for online-learning, we will label the unseen samples and then select some samples
    #   that are closer to its corresponding class center then the furthest sample (with same class id) in the base set
    conf = model(images)
    keep_flag = torch.gt(conf, thresh)
    images_keep = images[keep_flag]
