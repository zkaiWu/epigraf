#!/bin/bash



CUDA_VISIBLE_DEVICES=1 python src/infra/launch.py hydra.run.dir=. \
    desc=eg3d_fake_test_launch_debug \
    dataset=ffhq_posed \
    dataset.path=/home2/zhongkaiwu/data/dreamfusion_data/eg3d_fake/eg3d_generation_data_g4.0_noise500_long_prompt \
    dataset.resolution=256 \
    training.gamma=0.1 \
    training.resume=null \
    num_gpus=1 \
    training.batch_size=16 \