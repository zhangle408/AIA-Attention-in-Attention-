#!/usr/bin/env bash


#---in PyramidNet--------------------------------------------
python train_cinic.py \
--arch=MX_D_C_I \
--dataset=CINIC10 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=2 \
--num_frequency=8 \
--alpha=1.0

