import numpy as np
import copy
import time
import torch
import torch.nn as nn
import argparse
import imagenet_utils
from recon import reconstruction
from fold_bn import search_fold_and_remove_bn, StraightThrough
from calibquant.model import load_model, specials
from calibquant.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from calibquant.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from calibquant.quantization.fake_quant import QuantizeBase
from calibquant.quantization.observer import ObserverBase
from calibquant.quantization.util_log import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# python ./calibquant/solver/main_imagenet.py --config ./exp/w4a4/rs101/config.yaml

parser = argparse.ArgumentParser(description='Calibquant configuration',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', default='config.yaml', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = imagenet_utils.parse_config(args.config)


# logging setting
output_path = get_ckpt_path_train(config)
set_util_logging(output_path + "/calibquant.log")
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(output_path + "/calibquant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(output_path)
logger.info(args)
logger.info(config)


def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while (st < ed):
            tmp_qoutput = qoutput if st == ed - 1 else True
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](
                    child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module,
                        None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    replace_module(model, config_quant.w_qconfig,
                   config_quant.a_qconfig, qoutput=False)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and 'weight' in name:
            w_list.append(module)
        if isinstance(module, QuantizeBase) and 'act' in name:
            a_list.append(module)
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)
    a_list[0].set_bit(8)
    'the image input has already been in 256, set the last layer\'s input to 8-bit'
    a_list[-1].set_bit(8)
    logger.info('finish quantize model:\n{}'.format(str(type(model))))
    return model


def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


# Load validation data
imagenet_utils.set_seed(config.process.seed)
logger.info('==> Preparing data..')
'cali data'
train_loader, val_loader = imagenet_utils.load_data(**config.data)
cali_data = get_cali_data(train_loader, config.quant.calibrate)

'model'
# Model
logger.info('==> Building Model..')
model = load_model(config.model)
logger.info('==> Folding BN..')
search_fold_and_remove_bn(model)

# Set Quantizer
logger.info('==> Setting Quantizer..')
if hasattr(config, 'quant'):
    model = quantize_model(model, config.quant)
model.cuda()
model.eval()
fp_model = copy.deepcopy(model)
disable_all(fp_model)
for name, module in model.named_modules():
    if isinstance(module, ObserverBase):
        module.set_name(name)
# Calibrate first
logger.info('==> Calibrate Quantization Params..')
with torch.no_grad():
    st = time.time()
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    model(cali_data[: 256].cuda())
    enable_calibration_woquantization(
        model, quantizer_type='weight_fake_quant')
    model(cali_data[: 2].cuda())
    ed = time.time()
    logger.info('the calibration time is {}'.format(ed - st))
logger.info("Calibrate has Done!")

# Recon
logger.info('==> Recon Quantization Params..')
if hasattr(config.quant, 'recon'):
    enable_quantization(model)

    def recon_model(module: nn.Module, fp_module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                logger.info('begin reconstruction for module:\n{}'.format(
                    str(child_module)))
                reconstruction(model, fp_model, child_module, getattr(
                    fp_module, name), cali_data, config.quant.recon)
            else:
                recon_model(child_module, getattr(fp_module, name))
    # Start reconstruction
    recon_model(model, fp_model)
logger.info("Recon has Done!")

enable_quantization(model)

"""
Save the quantized model...
"""
# Save the quantized model...
num = int(uuid.uuid4().hex[0:4], 16)
path = "model_quantized"
pathname = "{}_{}_w{}_a{}".format(
    config.model.type, config.data.type, config.quant.w_qconfig.bit, config.quant.a_qconfig.bit)
path = os.path.join(path, pathname)
if not os.path.isdir(path):
    os.mkdir(path)
quantized_path = '{}/{}_w{}_a{}_{}.pth'.format(
    path, config.model.type, config.quant.w_qconfig.bit, config.quant.a_qconfig.bit, num)
torch.save(model, quantized_path)
logger.info("Pretranied model has been save Done!")
logger.info("Pretranied model: {}".format(quantized_path))

logger.info('==> Validate Accuracy..')
imagenet_utils.validate_model(val_loader, model, device)
