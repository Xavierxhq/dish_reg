import argparse
from lib.core.function import create_logger, create_model, create_data_loader, create_loss_fn, \
    create_optim_and_scheduler, train, online_data_selection
from lib.core.config_parser import Configuration


def main():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()

    # init config
    config = Configuration(args.cfg)
    logger = create_logger(config)
    train_loader, train_set = create_data_loader(config, training=True)
    test_loader = create_data_loader(config, training=False)
    model = create_model(config)
    # TODO: load pre-trained model provided by PyTorch
    model.load(config.MODEL.PRETRAINED, is_pretrained=True)

    loss_fn = create_loss_fn(config)
    optimizer, scheduler = create_optim_and_scheduler(config, model)

    for i in range(config.TRAIN.N_DAYS):
        if i == 0:
            train(config, train_loader, test_loader, model,
                  loss_fn, optimizer, scheduler, logger, day=i)
        else:
            new_data = None  # TODO: set up new data by yourself.
            new_data_selected = online_data_selection(
                model, new_data, config.TRAIN.CONFIDENCE)
            train_loader, _ = create_data_loader(
                config, dataset=train_set, extra_data=new_data_selected, training=True)
            train(config, train_loader, test_loader, model,
                  loss_fn, optimizer, scheduler, logger, day=i)


if __name__ == '__main__':
    main()
