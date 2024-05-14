#!/bin/bash
cd src

world_size=5
batch_size=10
discriminator_lr=0.0002
generator_lr=0.0002
seed=3
backend=gloo
port=1234
dataset=cifar
model=$dataset
epochs=200
local_epochs=1
swap_interval=1
iid=1
n_samples_fid=10000
device=mps
master_addr=localhost
master_port=1234
log_interval=50

client_start=1
client_end=4
for i in $(seq ${client_start} ${client_end}); do
    echo "Starting client $i"
    port=$(($2 + i))
    python bootstrap.py \
        --name "Client $i" \
        --backend $backend \
        --port $port \
        --world_size $world_size \
        --dataset $dataset \
        --rank $i \
        --epochs $epochs \
        --local_epochs $local_epochs \
        --swap_interval $swap_interval \
        --discriminator_lr $discriminator_lr \
        --n_samples_fid $n_samples_fid \
        --generator_lr $generator_lr \
        --model $model \
        --device $device \
        --batch_size $batch_size \
        --iid $iid \
        --seed $seed \
        --master_addr $master_addr \
        --master_port $master_port \
        --log_interval $log_interval &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
