# 基于Paddle实现图像超分，降噪

## 目录
* [1. 模型](#1-模型) 
* [2. 数据准备](#2-数据准备)
* [3. 训练](#3-训练)
* * [3.1 单机单卡](#31-单机单卡启动)
* * [3.2 单机多卡](#32-单机多卡启动)
* [4. 实验结果](#4-实验结果)
* [5. 评估](#5-评估)
* * [5.1 单机单卡](#51-单机单卡)
* * [5.2 单机多卡](#52-单机多卡)
* [6 推理](#6-推理)
* * [6.1 模型导出](#61-模型导出)
* * [6.2 模型推理](#62-模型推理)


## 1. 模型
* [x] [Unet](https://arxiv.org/abs/1505.04597)

## 2. 数据准备
数据集采用的是百度去水印竞赛中提供的数据集，详情见[这里](https://aistudio.baidu.com/aistudio/competition/detail/706/0/datasets)。也可以使用自己的数据集，其中需要修改`.yaml`文件中的两个参数 `data_root` 和 `index_file`，如下。
```
Data:
  Train:
    Dataset:
      name: WaterMark
      data_root: ../dataset/watermark/data/train
      index_file: ../dataset/watermark/data/train/train_index.txt
```
需要修改训练集和测试集中的`index_file`，此文件每一行有两列，列与列之间用一个空格分割，第一列为噪声图像，第二列为干净图像，举例如下
```
watermark_scripts/generate_testdata/1331_0.jpg bg_pics/1331.jpg
watermark_scripts/generate_testdata/653_0.jpg bg_pics/653.jpg
watermark_scripts/generate_testdata/730_0.jpg bg_pics/730.jpg
watermark_scripts/generate_testdata/2718_0.jpg bg_pics/2718.jpg
watermark_scripts/generate_testdata/1623_0.jpg bg_pics/1623.jpg
watermark_scripts/generate_testdata/787_0.jpg bg_pics/787.jpg
watermark_scripts/generate_testdata/3068_0.jpg bg_pics/3068.jpg
watermark_scripts/generate_testdata/1810_0.jpg bg_pics/1810.jpg
```

## 3. 训练
此模型支持单机单卡和单机多卡训练，以下使用`UNet`跑`denoise`举例。（此次实验默认在单机双卡 2x3090 上运行的结果）
### 3.1 单机单卡
```
python tools/train.py -c ./configs/denoise/unet_watermark.yaml
```

### 3.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ./configs/denoise/unet_watermark.yaml
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
        <td>5h 12min</td>
        <td>32.97</td>
        <td>99.51</td>
        <td> <a href="https://drive.google.com/file/d/15VHAefjy6z11lGcVJc5wb0HNHZ0s-u6u/view?usp=share_link"> 链接</a></td>
        <td><a href="https://drive.google.com/file/d/15UuVmIIpLGqb0kzyVTP7cvERlsfyFaWi/view?usp=share_link">链接</a> </td>
    </tr>
    
</table>

## 5. 评估
此模型支持单机单卡和单机多卡评估，以下使用`UNet`跑`denoise`举例，生成的模型位置在`/output/denoise/UNet/best_model.pdparams`
### 5.1 单机单卡
```
python tools/eval.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model
```

### 5.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model
```

所有的评估日志都默认保存在 `./output/denoise/UNet/eval.log`

## 6. 推理
### 6.1 模型导出
首先需要导出推理模型，例如训练好的模型参数在`./output/denoise/UNet/best_model.pdparams`，命令为
```
python tools/export_model.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model
```
模型将自动导出到`./output/denoise/UNet/inference`。

### 6.2 模型推理
模型导出后将使用测试数据集对模型进行推理，例如所有的测试文件都在`./test_data`中，运行命令
```
python tools/inference.py -c ./configs/denoise/unet_watermark.yaml -o Data.Test.path=./test_data
```
模型会将推理的结果放入`./output/denoise/UNet/Img`中。