#!/bin/bash

# trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [ "$#" -ne 1 ]; then
    echo "You need to supply config file as argument! ($#)"
else

    CONFIG_FILE=$1
# python train.py --t_c=configs/train/base_train.yaml \
#                 --m_c=configs/model/dcgan_base.yaml 
                    # --m_name=$MODEL_NAME &

    # python train.py --config=configs/base_dcgan_config.yaml
    python train.py --config=$CONFIG_FILE

#     LOG=$1
#     if [ "$LOG" = "--log" ]; then
#         tensorboard --logdir="runs" &
#     fi

fi

# wait