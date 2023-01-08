# train unet for denoise

# 单卡训练
# python tools/train.py -c ./configs/denoise/unet_watermark.yaml

# 多卡训练，卡数制定gpus即可
python -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ./configs/denoise/unet_watermark.yaml


# eval
# 单卡评估
# python tools/eval.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model.pdparams

# 多卡评估
# python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model.pdparams


# 模型导出
# python tools/export_model.py -c ./configs/denoise/unet_watermark.yaml -o Global.pretrained_model=./output/denoise/UNet/best_model.pdparams

# 模型推理
# python tools/inference.py -c ./configs/denoise/unet_watermark.yaml -o Data.Test.path=./test_data