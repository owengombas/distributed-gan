#!/bin/bash
cd src

echo "Starting server on $1:$2"
python bootstrap.py \
    --name master \
    --backend gloo \
    --port $2 \
    --world_size $3 \
    --rank 0 \
    --dataset $4 \
    --epochs $6 \
    --local_epochs $8 \
    --swap_interval ${10} \
    --n_samples_fid ${14} \
    --discriminator_lr ${12} \
    --generator_lr ${13} \
    --model $5 \
    --device $9 \
    --batch_size ${11} \
    --iid ${16} \
    --seed ${17} \
    --master_addr $1 \
    --master_port $2 &

for i in $(seq 1 $((${15}))); do
    echo "Starting client $i"
    port=$(($2 + i))
    python bootstrap.py \
        --name "Client $i" \
        --backend gloo \
        --port $port \
        --world_size $3 \
        --dataset $4 \
        --rank $i \
        --epochs $6 \
        --local_epochs $8 \
        --swap_interval ${10} \
        --discriminator_lr ${12} \
        --n_samples_fid ${14} \
        --generator_lr ${13} \
        --model $7 \
        --device $9 \
        --batch_size ${11} \
        --iid ${16} \
        --seed ${17} \
        --master_addr $1 \
        --master_port $2 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
