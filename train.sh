CUDA_VISIBLE_DEVICES=1 python3 train.py \
--cfg config/pse_v1.1.0.yaml \
--batch_size 2 \
--epoch 600 \
--project /data/weixianwei/models/psenet/uniform \
--name v1.1.0
