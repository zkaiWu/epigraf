#!/bin/bash

CIDX=$1
NUM_GPU=$2
BATCH_SIZE=$3

CUDA_VISIBLE_DEVICES=$CIDX python src/infra/launch.py hydra.run.dir=. \
    desc=eg3d_fake_gpc0.5-noaug-fixpose-epigraffake \
    dataset=ffhq_posed \
    dataset.path=/home/zhongkaiwu/data/dreamfusion_data/epigraf_fake/epigraf_generation_data_g4.0_noise500_long_prompt \
    dataset.resolution=256 \
    training.gamma=0.1 \
    training.resume=null \
    num_gpus=$NUM_GPU \
    training.batch_size=$BATCH_SIZE \
    model.discriminator.camera_cond=true \
    model.discriminator.camera_cond_raw=true \
    model.generator.camera_cond=true \
    training.augment.mode=noaug \
    training.kimg=10000 \
