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
* [x] [EDSR](https://arxiv.org/abs/1707.02921)

## 2. 数据准备
数据集采用的是DIV2K数据集，详情见[这里](https://data.vision.ee.ethz.ch/cvl/DIV2K/)。也可以使用自己的数据集，其中需要修改`.yaml`文件中的两个参数 `data_root` 和 `index_file`，这里用`DIV2K`数据集举例如下。
```
Data:
  Train:
    Dataset:
      name: DIV2K
      data_root: ../dataset/DIV2K/data/train
      index_file: ../dataset/DIV2K/data/train/train_index.txt
```
需要修改训练集和测试集中的`index_file`，此文件每一行有两列，列与列之间用一个空格分割，第一列为噪声图像，第二列为干净图像，举例如下
```
DIV2K_train_LR_bicubic/X2/0103x2.png DIV2K_train_HR/0103.png
DIV2K_train_LR_bicubic/X2/0413x2.png DIV2K_train_HR/0413.png
DIV2K_train_LR_bicubic/X2/0031x2.png DIV2K_train_HR/0031.png
DIV2K_train_LR_bicubic/X2/0660x2.png DIV2K_train_HR/0660.png
DIV2K_train_LR_bicubic/X2/0126x2.png DIV2K_train_HR/0126.png
DIV2K_train_LR_bicubic/X2/0793x2.png DIV2K_train_HR/0793.png
DIV2K_train_LR_bicubic/X2/0764x2.png DIV2K_train_HR/0764.png
DIV2K_train_LR_bicubic/X2/0550x2.png DIV2K_train_HR/0550.png
```

## 3. 训练
此模型支持单机单卡和单机多卡训练，以下使用`EDSR`跑`DIV2K`，放大倍数为`X2`举例。（此次实验默认在单机双卡 2x3090 上运行的结果）
### 3.1 单机单卡
```
python tools/train.py -c ./configs/sr/edsr_div2k_x2.yaml
```

### 3.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ./configs/sr/edsr_div2k_x2.yaml
```

训练过程日志如下
```
2023-01-16 03:12:22,146 - ppsr - Train: Epoch: [187/500]  Step: [25/125]  lr: 0.00020  batch_cost: 0.7735s  read_cost: 0.3207s  L1Loss: 3.3663 loss: 3.3663  ips: 82/s  eta: 0d 8:24:5
2023-01-16 03:12:31,236 - ppsr - Train: Epoch: [187/500]  Step: [50/125]  lr: 0.00020  batch_cost: 0.3637s  read_cost: 0.0003s  L1Loss: 3.4125 loss: 3.4125  ips: 175/s  eta: 0d 3:56:49
2023-01-16 03:12:40,349 - ppsr - Train: Epoch: [187/500]  Step: [75/125]  lr: 0.00020  batch_cost: 0.3644s  read_cost: 0.0003s  L1Loss: 3.5957 loss: 3.5957  ips: 175/s  eta: 0d 3:57:10
2023-01-16 03:12:49,477 - ppsr - Train: Epoch: [187/500]  Step: [100/125]  lr: 0.00020  batch_cost: 0.3651s  read_cost: 0.0004s  L1Loss: 3.4811 loss: 3.4811  ips: 175/s  eta: 0d 3:57:29
2023-01-16 03:12:58,571 - ppsr - Train: Epoch: [187/500]  Step: [125/125]  lr: 0.00020  batch_cost: 0.3636s  read_cost: 0.0001s  L1Loss: 3.5846 loss: 3.5846  ips: 176/s  eta: 0d 3:56:20
2023-01-16 03:13:49,312 - ppsr - Eval: batch_cost: 0.5060s   read_cost: 0.0000s  L1Loss: 3.3315 loss: 3.3315  best_metric: 34.3915  PSNR: 34.3369 SSIM: 0.9357 
```

所有的训练日志都默认保存在 `./output/sr/X2/`

## 4. 实验结果
<table align="center">
    <tr>
        <td>模型</td>
        <td>数据集</td>
        <td>放大倍数</td>
        <td>PSNR</td>
        <td>log</td>
        <td>param</td>
    </tr>
    <tr>
        <td>torch_EDSR_baseline</td>
        <td>DIV2K</td>
        <td>X2</td>
        <td>34.61dB</td>
        <td> </td>
        <td> </td>
    </tr>
    <tr>
        <td>paddle_EDSR_baseline</td>
        <td>DIV2K</td>
        <td>X2</td>
        <td>34.51dB</td>
        <td> <a href="https://drive.google.com/file/d/1P7RSvywZ7-LsygYTX0bipvqiljjZqvK5/view?usp=share_link">链接</a></td>
        <td> <a href="https://drive.google.com/file/d/1pAnDAZBVkF062SvJS8p8RdsmDUaL44HY/view?usp=sharing">链接</a></td>
    </tr>
    
</table>

## 5. 评估
此模型支持单机单卡和单机多卡评估，以下使用`EDSR`跑`sr`举例，生成的模型位置在`./output/sr/X2/EDSR/best_model.pdparams`
### 5.1 单机单卡
```
python tools/eval.py -c ./configs/sr/edsr_div2k_x2.yaml -o Global.pretrained_model=./output/sr/X2/EDSR/best_model
```

### 5.2 单机多卡
```
python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/sr/edsr_div2k_x2.yaml -o Global.pretrained_model=./output/sr/X2/EDSR/best_model
```

所有的评估日志都默认保存在 `./output/sr/X2/EDSR/eval.log`，且设置预训练模型时无需后缀`.pdparams`

## 6. 推理
### 6.1 模型导出
首先需要导出推理模型，例如训练好的模型参数在`./output/sr/X2/EDSR/best_model.pdparams`，命令为
```
python tools/export_model.py -c ./configs/sr/edsr_div2k_x2.yaml -o Global.pretrained_model=./output/sr/X2/EDSR/best_model
```
模型将自动导出到`./output/sr/X2/EDSR/inference`。

### 6.2 模型推理
模型导出后将使用测试数据集对模型进行推理，例如所有的测试文件都在`./test_data`中，运行命令
```
python tools/inference.py -c ./configs/sr/edsr_div2k_x2.yaml -o Data.Test.path=./test_data
```
模型会将推理的结果放入`./output/sr/X2/EDSR/Img`中。