# yaml文件参数说明
----
* **Global**: (必选)
* * **checkoutpoints** （str， 可选）：加载提供的checkoutpoints路径（如./output/ck.pd.tar)， 默认为 `null`。 [!目前未实现此功能]
* * **pretrained_model** （str，可选）：提供的预训练模型（包括.pdparams和.pdopt）的路径，无需设置后缀，如(./output/best_model)，默认为 `null`。
* * **output_dir**（str，可选）：模型的输出（包括日志，模型保存等）路径，若无，则默认为 `./output`。
* * **save_interval**（int，可选）：模型每 *save_interval* 个epoch保存一次此epoch的模型参数，-1表示不保存模型epoch参数记录（注意，模型会默认保存在训练过程中的best_model以及least_model，此参数不影响这两个模型的保存），默认为 `-1`。
* * **dist**（bool，可选）：是否开启分布式训练，如果在命令行使用多卡训练时，则此参数必须为 True，默认为 `False`。
* * **eval_during_train**（bool，可选）：是否在训练中对验证集进行评估，默认为 `True`。
* * **eval_interval**（int，可选）：若 *eval_during_train=True*，则每隔 *eval_interval* 个epoch进行评估一次，默认为 `1`。
* * **seed**（int or False，可选）：设置随机数种子，默认为 `False`。
* * **epochs**（int，必选）：总共训练的 epoch 数。
* * **step_per_epoch**（int，可选）：每个 epoch 训练多少个 iter。当 step_per_epoch * batch_size 大于训练样本数时默认失效。
* * **print_batch_step**（int，必选）：每隔多少个 iter 打印一次日志。
* * **bar_disable**（bool，可选）：是否在训练时不展示进度条，默认为 `True`。
* * **img_size**（list[channel: int, height: int, weight: int]，必选）：在评估以及推理时，需要先将图片分割成 patch，此参数即为 patch的大小，建议根据验证集以及测试集进行适当调整。
* * **schedule_update_by**（str，可选）：schedule的更新频率，可选择 `epoch` 或者 `step`，默认为 `step`。
* * **rgb_range**（float，可选）：在训练过程中像素的最大值，可选择 `1.0` 或者 `255.0`，默认为 `1.0`。
* * **trainer**（str，必选）：选择 trainer，此参数需要与[这里](../engine/trainer/)的文件名保持一致，通常使用 `common` 即可，若使用自定义 trainer，则需要进行修改。
* * **scale**（int，可选）：进行超分辨率训练时的图像扩大倍数，当为 `1` 时则为图像降噪训练，默认为 `1`。

----
* **AMP**（可选）
* * **level**（str，可选）：选择混合精度限制级别，选择 `O1` 或者 `O2`，默认为 `O1`。
* * **init_loss_scaling**（float，可选）：设置损失放大倍数，默认为 `32768.0`。
* * **use_dynamic_loss_scaling**（bool，可选）：是否使用动态调整损失扩大倍数，默认为 `True`。

----
* **Arch**（必选）
* * **name**（str，必选）：选择 Arch 的名字，具体可选的名字参考[此文件](../arch/backbone/__init__.py)导入的 Arch 名。
* * **use_sync_bn**（bool，可选）：是否使用 `sync_bn`，建议在多卡训练时设置为 `True`，默认为 `False`。
* * **kwargs：此处需要根据不同的模型所需的参数进行配置。

----
* **Data**（必选）
* * **Train**（必选）
* * * **Dataset**（必选）
* * * * **name**（str，必选）：选择数据集的名字，选择参考列表为[此文件](../dataloader/data/__init__.py)所导入的数据集名字。
* * * * **data_root**（str，必选）：数据集所在目录。
* * * * **index_file**：（str，必选）：数据的训练样本和label样本所对应的文件路径。
* * * * **data_expane**（int or None，可选）：训练样本扩展倍数，当为 `null` 时则不进行样本扩展。默认为 `null`。
* * * * **ops**（list，必选）：使用不同的数据增强方法，具体可参考[这里](../configs/sr/edsr_div2k_x2.yaml)。
* * * **DataLoader**（必选，具体使用方法可参考[这里](../configs/sr/edsr_div2k_x2.yaml)）。

* * **Eval**（必选）
* * * **Dataset**（必选）
* * * * **name**（str，必选）：必须设置成 `ValDataset`。
* * * * **data_root**（str，必选）：数据集所在目录。
* * * * **index_file**：（str，必选）：数据的验证样本和label样本所对应的文件路径。
* * * **DataLoader**（必选)，具体使用方法可参考[这里](../configs/sr/edsr_div2k_x2.yaml）。

* * **Test**（必选）
* * * **path**（str，必选）：设置测试图片所在路径。

----
* **Loss**（必选）
* * **Train**（list，必选）具体使用方法可参考[这里](../configs/sr/edsr_div2k_x2.yaml）。
* * **Eval**（list，可选）具体使用方法可参考[这里](../configs/sr/edsr_div2k_x2.yaml）。

----
* **Optimizer**（必选）
* * **name**（str，必选）：设置优化器的名字。
* * **keargs**：其他参数需要根据不同的优化器设置。

----
* **Metric**（必选）
* * **save_rely_metric**（str，必选）：保存模型时需要保存 `best_model`，而 `best_model` 的选择需要有个依据，此参数设置的评估名即为保存 `best_model` 的依据。此参数需为 `Metric.Eval` 中的一个。
* * **Eval**（list，必选）：设置多个评估方式，具体使用方法可参考[这里](../configs/sr/edsr_div2k_x2.yaml）。
