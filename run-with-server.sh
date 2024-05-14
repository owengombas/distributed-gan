#!/bin/bash
. ./set-args.sh
cd src

seed=3
world_size=5
backend=gloo
port=1234
swap_interval=1
master_addr=localhost
master_port=1234
n_clients=4

echo "Starting server on $master_addr:$master_port"
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

for i in $(seq 1 $((${n_clients}))); do
    echo "Starting client $i"
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
