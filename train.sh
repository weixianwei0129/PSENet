CUDA_VISIBLE_DEVICES=1 python3 train.py \
--cfg config/pse_v1.2.0.yaml \
--epoch 600 \
--project /data/weixianwei/models/psenet/uniform \
--name v1.2.0 \
# --resume
--pretrain \
--weights /data/weixianwei/psenet/models/psenet_r50_custom_736/checkpoint_600ep.pth.tar