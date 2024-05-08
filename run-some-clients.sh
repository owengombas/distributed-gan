#!/bin/bash
cd src

for i in $(seq ${15} ${16}); do
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
        --master_addr $1 \
        --master_port $2 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
