CUDA_VISIBLE_DEVICES=1 python3 train.py \
--cfg config/bankcard_v1.1.0.yaml \
--epochs 800 \
--project /data/weixianwei/models/bankcard/text_det/ \
--name v1.1.0 \
--resume