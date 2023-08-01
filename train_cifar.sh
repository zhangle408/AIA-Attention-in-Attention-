#!/usr/bin/env bash


#---in PyramidNet--------------------------------------------
python train_cifar.py \
--arch=M2_D_C_I \
--dataset=cifar100 \
--epochs=300 \
--start_epoch=0 \
--batch_size=128 \
--lr=0.1 \
--lr_schedule=1 \
--alpha=0.5
