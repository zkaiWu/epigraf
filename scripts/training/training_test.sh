#!/bin/bash



CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python src/infra/launch.py hydra.run.dir=. \
    desc=eg3d_fake_gpc0.5-noaug \
    dataset=ffhq_posed \
    dataset.path=/home/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise500_long_prompt \
    dataset.resolution=256 \
    training.gamma=0.1 \
    training.resume=null \
    num_gpus=6 \
    training.batch_size=24 \
    model.discriminator.camera_cond=true \
    model.discriminator.camera_cond_raw=true \
    model.generator.camera_cond=true \
    training.augment.mode=noaug \
    training.kimg=10000 \
