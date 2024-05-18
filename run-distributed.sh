#!/bin/bash
. ./shared-args.sh
cd src

seed=3
world_size=5
backend=gloo
port=1234
swap_interval=100
master_addr=localhost
master_port=1234
n_workers=$(($world_size-1))

if [ $(($n_workers%2)) -ne 0 ]; then
    echo "Number of workers must be even, change the world_size to have consider an even number of workers (3, 5, 7, ...)"
    exit 1
fi

client_start=${1:-0}
client_end=${2:-4}
if [ $client_start -le 0 ]; then
    python bootstrap.py \
        --name "Server" \
        --backend $backend \
        --port $port \
        --world_size $world_size \
        --dataset $dataset \
        --rank 0 \
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
fi

if [ $client_start -le 0 ]; then
    client_start=1
fi

for i in $(seq ${client_start} ${client_end}); do
    port=$((${port} + i))
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
