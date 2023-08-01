#!/usr/bin/env bash


#---in PyramidNet--------------------------------------------
python train_imagenet.py \
--arch=M2_D_C_I_imagenet \
--dataset=imagenet \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.05 \
--alpha=1.0

