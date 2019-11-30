#!/bin/bash

python train.py --epochs=200 --batch_size=256 --nz=100 --use_cpu \
                --model_dir=first_model --model_name=first_model \
                --save_model
