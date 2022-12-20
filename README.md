# 基于Paddle实现图像超分，降噪

## 目录
* [1. 模型](#1-模型) 
* [2. 数据准备](#2-数据准备)
* [3. 训练](#3-训练)
* * [3.1 单机单卡](#31-单机单卡启动)
* * [3.2 单机多卡](#32-单机多卡启动)
* [5. 评估](#4-评估)
* * [5.1 单机单卡](#41-单机单卡)
* * [5.2 单机多卡](#42-单机多卡)

## 1. 模型
* [x] [Unet]()

## 2. 数据准备

## 3. 训练
此模型支持单机单卡和单机多卡训练，以下使用`UNet`跑`denoise`举例。
### 3.1 单机单卡
```
python tools/train.py -c ./configs/denoise/unet_watermark.yaml
```

### 3.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ./configs/denoise/unet_watermark.yaml
```

所有的训练日志都默认保存在 `./output/UNet/train.log`

## 实验结果
<table align="center">
    <tr>
        <td>模型</td>
        <td>数据集</td>
        <td>batch_size</td>
        <td>Iter</td>
        <td>PSNR</td>
        <td>MS_SSIM</td>
        <td>训练时长</td>
        <td>log</td>
        <td>param</td>
    </tr>
    <tr>
        <td>UNet</td>
        <td>watermark</td>
        <td>6</td>
        <td>2000</td>
        <td>24.45</td>
        <td>0.968</td>
        <td>34min</td>
        <td></td>
        <td></td>
    </tr>
    
</table>

## 5. 评估
此模型支持单机单卡和单机多卡评估，以下使用`UNet`跑`denoise`举例，生成的模型位置在`/output/UNet/best_model.pdparams`
### 5.1 单机单卡
```
python tools/eval.py -c ./configs/denoise/unet_watermark_4gpu.yaml -o Global.pretrained_model=./output/UNet/best_model.pdparams
```

### 5.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/eval.py -c ./configs/denoise/unet_watermark_4gpu.yaml -o Global.pretrained_model=./output/UNet/best_model.pdparams
```

所有的评估日志都默认保存在 `./output/UNet/eval.log`

