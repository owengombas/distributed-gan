#!/bin/bash
cd src

batch_size=10
generator_lr=0.0002
discriminator_lr=0.0002
log_interval=50
local_epochs=1
epochs=200
model=cifar
dataset=$model
device=mps
seed=1

python standalone_gan.py --local_epochs $local_epochs \
    --epochs $epochs \
    --model $model \
    --dataset $dataset \
    --generator_lr $generator_lr \
    --discriminator_lr $discriminator_lr \
    --device $device \
    --batch_size $batch_size \
    --seed $seed \
    --log_interval $log_interval
