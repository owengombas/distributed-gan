#!/bin/bash

batch_size=10
discriminator_lr=0.0002
generator_lr=0.0002
dataset=cifar
model=$dataset
epochs=1000
local_epochs=1
iid=1
n_samples_fid=10000
device=mps
log_interval=50
