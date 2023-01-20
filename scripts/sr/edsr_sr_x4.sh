# train edsr

############    超分训练      ##########

# 超分x4
# 单卡训练
# python tools/train.py -c ./configs/sr/edsr_div2k_x4.yaml

# 多卡训练，卡数制定gpus即可
python -m paddle.distributed.launch --gpus=0,1 tools/train.py -c ./configs/sr/edsr_div2k_x4.yaml

# eval
# 单卡评估
# python tools/eval.py -c ./configs/sr/edsr_div2k_x4.yaml -o Global.pretrained_model=./output/sr/X4/EDSR/best_model

# 多卡评估
# python -m paddle.distributed.launch --gpus=0,1 tools/eval.py -c ./configs/sr/edsr_div2k_x4.yaml -o Global.pretrained_model=./output/sr/X4/EDSR/best_model


# 模型导出
# python tools/export_model.py -c ./configs/sr/edsr_div2k_x4.yaml -o Global.pretrained_model=./output/sr/X4/EDSR/best_model

# 模型推理
# python tools/inference.py -c ./configs/sr/edsr_div2k_x4.yaml -o Data.Test.path=./docs/imgs/lr/X4