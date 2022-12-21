# 基于Paddle实现图像超分，降噪

## 目录
* [1. 模型](#1-模型) 
* [2. 数据准备](#2-数据准备)
* [3. 训练](#3-训练)
* * [3.1 单机单卡](#31-单机单卡启动)
* * [3.2 单机多卡](#32-单机多卡启动)
* [4. 实验结果](#4-实验结果)
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

## 4. 实验结果
<table align="center">
    <tr>
        <td>模型</td>
        <td>数据集</td>
        <td>batch_size</td>
        <td>Iter</td>
        <td>训练时长</td>
        <td>PSNR</td>
        <td>MS_SSIM</td>
        <td>log</td>
        <td>param</td>
    </tr>
    <tr>
        <td>UNet</td>
        <td>watermark</td>
        <td>6</td>
        <td>2000</td>
        <td>34min</td>
        <td>24.45</td>
        <td>0.968</td>
        <td></td>
        <td></td>
    </tr>
    
</table>

## 5. 评估
此模型支持单机单卡和单机多卡评估，以下使用`UNet`跑`denoise`举例，生成的模型位置在`/output/UNet/best_model.pdparams`
### 5.1 单机单卡
```
python tools/eval.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/UNet/best_model.pdparams
```

### 5.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1,2,3 tools/eval.py -c ./configs/denoise/unet_watermark_4gpu.yaml -o Global.pretrained_model=./output/UNet/best_model.pdparams
```

所有的评估日志都默认保存在 `./output/UNet/eval.log`

## 6. 推理
### 6.1 模型导出
首先需要导出推理模型，例如训练好的模型参数在`./output/UNet/best_model.pdparams`，命令为
```
python tools/export_model.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/UNet/best_model.pdparams
```
模型将自动导出到`Global.output_dir/Arch.name/inference`，其中`Global.output_dir`和`Arch.name`均在配置文件`.yaml`中。

### 6.2 模型推理
模型导出后将使用测试数据集对模型进行推理，例如所有的测试文件都在`./test_data`中，运行命令
```
python tools/inference.py -c ./configs/denoise/unet_watermark.yaml
```