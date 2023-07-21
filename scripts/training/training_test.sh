#!/bin/bash



CUDA_VISIBLE_DEVICES=1 python src/infra/launch.py hydra.run.dir=. \
    desc=eg3d_fake_saveimage_debugging \
    dataset=ffhq_posed \
    dataset.path=/home2/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise500_long_prompt \
    dataset.resolution=256 \
    training.gamma=0.1 \
    training.resume=null \
    num_gpus=1 \
    training.batch_size=4 \
    model.discriminator.camera_cond=true \
    model.discriminator.camera_cond_raw=true \
    model.generator.camera_cond=true \
    training.augment.mode=noaug \
    training.kimg=10000 \
