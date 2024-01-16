import os
import logging
import uuid

logger = logging.getLogger(__name__)


def get_ckpt_path_train(config):
    path = 'train_log'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, config.model.type + "_" + config.data.type)
    if not os.path.isdir(path):
        os.mkdir(path)
    pathname = "calibquant" + '_W' + \
        str(config.quant.w_qconfig.bit) + 'A' + str(config.quant.a_qconfig.bit)
    num = int(uuid.uuid4().hex[0:4], 16)
    pathname += '_' + str(num)
    path = os.path.join(path, pathname)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def get_ckpt_path_test(config):
    path = 'test_log'
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, config.model.type + "_" + config.data.type)
    if not os.path.isdir(path):
        os.mkdir(path)
    pathname = "calibquant" + '_W' + \
        str(config.quant.w_qconfig.bit) + 'A' + str(config.quant.a_qconfig.bit)
    num = int(uuid.uuid4().hex[0:4], 16)
    pathname += '_' + str(num)
    path = os.path.join(path, pathname)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def set_util_logging(filename):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename),
            logging.StreamHandler()
        ]
    )
