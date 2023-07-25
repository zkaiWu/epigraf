#!/bin/bash

CIDX=$1
NUM_GPU=$2
BATCH_SIZE=$3

CUDA_VISIBLE_DEVICES=$CIDX python src/infra/launch.py hydra.run.dir=. \
    desc=epigraf_fake_noCond4D_Wcond4G_noaug \
    dataset=ffhq_posed \
    dataset.path=/home/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise500_long_prompt \
    dataset.resolution=256 \
    training.gamma=0.1 \
    training.resume=null \
    num_gpus=$NUM_GPU \
    training.batch_size=$BATCH_SIZE \
    model.discriminator.camera_cond=false \
    model.discriminator.camera_cond_raw=false \
    model.generator.camera_cond=true \
    model.loss_kwargs.density_loss=0 \
    training.augment.mode=noaug \
    training.kimg=10000 \
