# 基于Paddle实现图像超分，降噪

## 目录
* [1. 模型](#1-模型) 
* [2. 数据准备](#2-数据准备)
* [3. 训练](#3-训练)
* * [3.1 单机单卡启动](#31-单机单卡启动)
* * [3.2 单机多卡启动](#32-单机多卡启动)

## 1. 模型
* [ ] [Unet]()

## 2. 数据准备

## 3. 训练
此模型支持单机单卡和单机多卡训练，以下使用`UNet`跑`denoise`举例。
### 3.1 单机单卡启动
```
python tools/train.py -c ./configs/denoise/unet_watermark.yaml
```

### 3.2 单机多卡启动
```
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ./configs/denoise/unet_watermark.yaml
```

所有的训练日志都默认保存在 `./output/UNet/train.log`

