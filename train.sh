CUDA_VISIBLE_DEVICES=1 python3 train.py \
--cfg config/pse_v1.4.0.yaml \
--epochs 800 \
--project /data/weixianwei/models/psenet/uniform \
--name v1.4.0 \
--resume
#--pretrain \
#--weights /data/weixianwei/models/psenet/uniform/v1.3.0/ckpt/last.pt
