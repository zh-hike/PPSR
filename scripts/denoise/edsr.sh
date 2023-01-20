# 多卡训练，卡数制定gpus即可
python -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ./configs/denoise/edsr_watermark.yaml


# eval
# 单卡评估
# python tools/eval.py -c ./configs/denoise/edsr_watermark.yaml -o Global.pretrained_model=./output/denoise/EDSR/best_model

# 多卡评估
# python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/denoise/edsr_watermark.yaml -o Global.pretrained_model=./output/denoise/EDSR/best_model

# 多卡评估所有水印测试样本
python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/denoise/edsr_watermark.yaml -o Global.pretrained_model=./output/denoise/EDSR/best_model -o Data.Eval.Dataset.index_file=../dataset/watermark/data/train/all_val_index.txt -o Global.eval_bar_disable=true

# 模型导出
# python tools/export_model.py -c ./configs/denoise/edsr_watermark.yaml -o Global.pretrained_model=./output/denoise/EDSR/best_model


# 模型推理
# python tools/inference.py -c ./configs/denoise/edsr_watermark.yaml
