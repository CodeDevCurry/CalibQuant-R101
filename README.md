# Quant_ResNet101
## Introduction
Our algorithm, tailored to the unique structure of ResNet, reconstructs  the model layer-by-layer or block-by-block, and by selecting an  appropriate calibration dataset, ultimately achieves a precision loss of only 1.04% when quantizing the ResNet101 network to 4 bits, closely  approaching 1%.

## File Organization

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
## Environment Configuration
```text
GPU: 1 Card（NVIDIA A100 Tensor Core GPU）
CPU: 4 Core
Memory: 20 GB
Block Storage: 50 GB
Operating System: Linux
Programming Language: Python（3.8）
```
### Create a virtual environment
```shell
conda create -n calibquant_py38 python=3.8
```
### Activate a virtual environment
```shell
conda activate calibquant_py38
```
### Install other configurations
```shell
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install PyYAML

pip install easydict

pip install scipy
```

## Quantize
Go into the exp/w4a4 directory. You can find config.yaml for each architecture. Execute `main_imagenet.py` for quantized model. Other bit settings only need to change the corresponding bit number in yaml file.
To quantize the ResNet101 model on the ImageNet dataset to a lower bit precision, you can easily achieve this using a Shell command. For example, to quantize the ResNet101 model to W4A4 on the ImageNet dataset, simply execute the following command:
```shell
cd CalibQuant-R101

# Quantize the ResNet101 model to W4A4
python ./calibquant/solver/main_imagenet.py --config ./exp/w4a4/rs101/config.yaml
```
This command initiates the quantization process, allowing you to transform the ResNet101 model to the specified lower bit precision directly on the ImageNet dataset.

## Save the Quantized Model Weight Files
During the process of model quantization, the quantized model structure and its parameters are fully saved. For example, in the case of quantizing the ResNet101 model to W4A4, the quantized model and parameters are stored in a `.pth` file within a folder named `model_quantized/resnet101_imagenet_w4_a4`.
The file name follows the format `resnet101_w4_a4_[random_number].pth`, where `[random_number]` is a sequence of random digits. The purpose of this random number is to distinguish each quantized file saved, ensuring the uniqueness of the file names. 
For instance, a specific file name could be `resnet101_w4_a4_57283.pth`, where `57283` is a randomly generated number.
This method allows users to easily identify and manage model files produced from different batches of quantization.

## Testing Quantized Models
You can easily assess the performance of previously saved quantized model files using Shell commands. For example, to evaluate the accuracy of the ResNet101 model quantized to W4A4 on the ImageNet dataset, simply run the following command:
```shell
python ./calibquant/solver/test_imagenet.py --config ./exp/w4a4/rs101/config.yaml --quantized_model_path resnet101_w4_a4_57283.pth
```
It's important to note that the parameter following "--quantized_model_path" should be the path to the saved quantized model file. This allows you to quickly test the performance of the saved quantized model on the specified dataset without the need to re-quantize it. Just provide the path to the saved quantized model file, and the testing script will assess its effectiveness on the specified dataset.

## Results
Optimizing ResNet101 for low-bit representation and then displaying the results.
|  Methods   | Bits (W/A) | ResNet101 |
| :--------: | :--------: | :-------: |
| Full Prec. |   32/32    |  77.368   |
| CalibQuant |    4/4     |  73.448   |
| CalibQuant |    2/4     |   71.18   |

Furthermore, we found that selecting an appropriate calibration dataset  is crucial for the quantization reconstruction process. Therefore, we explored the impact of the calibration dataset on the quantization  reconstruction process. By randomly selecting three calibration datasets, an appropriate calibration dataset can significantly improve the accuracy of the quantized model. The results are as follows:

| Calibration  Dataset | Bits (W/A) | ResNet101 |
| :------------------: | :--------: | :-------: |
|      Full Prec.      |   32/32    |  77.368   |
|        sel_1         |    4/4     |   75.31   |
|        sel_2         |    4/4     |  73.448   |
|        sel_3         |    4/4     |  76.328   |
|        sel_1         |    2/4     |  73.276   |
|        sel_2         |    2/4     |   71.18   |
|        sel_3         |    2/4     |  74.264   |

We discovered that selecting the appropriate calibration dataset can  improve the performance of the quantized model by 2.88% on w4a4 and by  3.084% on w2a4. Ultimately, quantizing ResNet101 to w4a4, the error was  controlled within 1.04%, achieving a satisfactory quantization effect.
