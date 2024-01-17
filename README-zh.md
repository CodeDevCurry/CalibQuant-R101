# CalibQuant-R101
## 介绍
CalibQuant-R101: 校准导向的ResNet101量化算法（Calibration-Guided Quantization Algorithm for ResNet101）

针对ResNet系列网络模型独特的结构，我们提出了一种新颖的量化算法。该算法通过逐层或逐块重构策略，高效地处理模型的每个模块，并且选择合适的校准数据集，我们的方法能够更准确地调整量化重构过程中的参数。最终，我们成功地将ResNet101网络模型量化到4比特，同时保持了模型的核心性能，仅造成了1.04%的精度损失，接近1%。此外，这种方法在保持模型精度的同时，显著降低了模型的存储和计算需求，为在资源受限的设备上部署高效的深度学习模型提供了可能。

## 文件结构

```
CalibQuant-R101/

├── exp/                                [Quantization Configuration]
│   ├── w4a4                            [Quantize weights and activations to 4 bits]
│   ├── ├──config.yaml                  [Specific Quantization Settings, this includes specific quantization settings such as the type of quantization for the model, methods for quantizing weights and activations, and parameters for the reconstruction process, among others]
│   ├── w2a4                            [Quantize weights to 2 bits, activations to 4 bits]     

├── imagenet/                           [ImageNet dataset]
│   ├── val                             [Used for testing accuracy]   
│   ├── sel                             [sel_1024_1000classes, randomly select 1024 images from the ImageNet training dataset, ensuring at least one image from each class, for calibration in the quantization process]

├── model_quantized/                    [Saving of quantized models]
│   ├── resnet101_imagenet_w4_a4        [Saving the quantized model after quantizing both weights and activations of the ResNet101 model to 4 bits]

├── calibquant/  
│   ├── model                           [Definition of quantized models]

│   ├── quantization/                   [Quantization tools]
│   │   ├── fake_quant.py               [Implement quantize and dequantize functions]   
│   │   ├── observer.py                 [Collect the information of distribution and calculate quantization clipping range]     
│   │   ├── state.py                    [Set quantization states]

│   ├── solver/ 
│   |   ├── main_imagenet.py            [Run quantization on imagenet dataset]
│   |   ├── test_imagenet.py            [Run accuracy testing with the quantized model]
│   |   ├── test.py                     [Run testing to evaluate the accuracy of the original, unquantized model on the ImageNet dataset]
│   |   ├── imagenet_util.py            [Load imagenet dataset]
│   |   ├── recon.py                    [Reconstruct models]

├── train_log/                          [Save logs during the quantization process]

├── test_log/                           [Save logs during the testing process of the quantized model]
```
## 环境配置
```text
GPU: 1 Card（NVIDIA A100 Tensor Core GPU）
CPU: 4 Core
Memory: 20 GB
Block Storage: 50 GB
Operating System: Linux
Programming Language: Python（3.8）
```
### 创建虚拟环境
```shell
conda create -n calibquant_py38 python=3.8
```
### 激活虚拟环境
```shell
conda activate calibquant_py38
```
### 安装其它配置
```shell
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install PyYAML

pip install easydict

pip install scipy
```

## 量化
进入 `exp/w4a4` 目录，可以找到每种架构的 `config.yaml` 文件，执行 `main_imagenet.py` 来运行量化模型。对于其它比特设置，只需在 yaml 文件中更改相应的比特数即可。 

要在 ImageNet 数据集上将 ResNet101 模型量化为较低的比特精度，可以轻松地使用 Shell 命令来实现。例如，要将  ImageNet 数据集上的 ResNet101 模型量化为 W4A4，只需执行以下命令：

```shell
cd CalibQuant-R101

# Quantize the ResNet101 model to W4A4
python ./calibquant/solver/main_imagenet.py --config ./exp/w4a4/rs101/config.yaml
```
此命令启动量化过程，允许直接在 ImageNet  数据集上将 ResNet101 模型转换为指定的较低比特精度。

## 保存量化模型的权重文件
在模型量化的过程中，量化后的模型结构及其参数会被完整保存。例如，在将 ResNet101 模型量化为 W4A4 的情况下，量化后的模型和参数存储在名为 `model_quantized/resnet101_imagenet_w4_a4` 的文件夹中的 `.pth` 文件里。

文件名遵循格式 `resnet101_w4_a4_[random_number].pth`，其中 `[random_number]` 是一串随机数字。这个随机数字的目的是为了区分每个保存的量化文件，确保文件名的唯一性。

例如，一个特定的文件名可能是 `resnet101_w4_a4_57283.pth`，其中 `57283` 是随机生成的数字。这种方法允许用户轻松识别和管理不同批次量化产生的模型文件。

## 测试量化模型

可以使用 Shell 命令轻松评估先前保存的量化模型文件的性能。例如，要评估在 ImageNet 数据集上量化为 W4A4 的 ResNet101 模型的准确性，只需运行以下命令：

```shell
python ./calibquant/solver/test_imagenet.py --config ./exp/w4a4/rs101/config.yaml --quantized_model_path resnet101_w4_a4_57283.pth
```
It's important to note that the parameter following "--quantized_model_path" should be the path to the saved quantized model file. This allows you to quickly test the performance of the saved quantized model on the specified dataset without the need to re-quantize it. Just provide the path to the saved quantized model file, and the testing script will assess its effectiveness on the specified dataset.

需要注意的是，"--quantized_model_path" 后面的参数应该是已保存的量化模型文件的路径。这样，可以快速测试已保存的量化模型在指定数据集上的性能，而无需重新进行量化。只需提供已保存的量化模型文件的路径，测试脚本就会评估其在指定数据集上的有效性。

## 实验结果

将 ResNet101 模型量化到低比特的实验结果如下：

|  Methods   | Bits (W/A) | ResNet101 |
| :--------: | :--------: | :-------: |
| Full Prec. |   32/32    |  77.368   |
| CalibQuant |    4/4     |  73.448   |
| CalibQuant |    2/4     |   71.18   |

此外，发现选择合适的校准数据集对量化重构过程至关重要。因此，探讨了校准数据集对量化重构过程的影响。通过随机选择三个校准数据集，发现合适的校准数据集可以显著提高量化模型的准确性。结果如下：

| Calibration  Dataset | Bits (W/A) | ResNet101 |
| :------------------: | :--------: | :-------: |
|      Full Prec.      |   32/32    |  77.368   |
|        sel_1         |    4/4     |   75.31   |
|        sel_2         |    4/4     |  73.448   |
|        sel_3         |    4/4     |  76.328   |
|        sel_1         |    2/4     |  73.276   |
|        sel_2         |    2/4     |   71.18   |
|        sel_3         |    2/4     |  74.264   |

根据表格可知，选择合适的校准数据集可以在 w4a4 上提高量化模型的性能 2.88%，在 w2a4 上提高 3.084%。最终，将 ResNet101 量化为 w4a4，损失精度得以控制在 1.04% 以内，实现了令人满意的量化效果。
