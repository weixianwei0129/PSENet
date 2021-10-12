# source ~/venv/py36torch/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 train.py \
--cfg config/bankcard_v1.0.0.yaml \
--epochs 800 \
--project /data/weixianwei/models/bankcard/text_det/ \
--name v1.0.0 \
--resume
# --pretrain \
# --weights /data/weixianwei/models/psenet/uniform/v1.4.0/ckpt/last.pt
