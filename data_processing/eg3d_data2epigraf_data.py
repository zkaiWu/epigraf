import json
import argparse
import os
import torch 
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str) # the FFHQ dataset created by `dataset_preprocessing/ffhq/runme.py`
    args = parser.parse_args()
    return args


def transform_json(args):
    """
    This function can transform the eg3d camera pose to epigraf camera angle
    """
    input_dir = args.input_dir
    eg3d_json_path = os.path.join(input_dir, 'dataset_eg3d.json')

    epigraf_json_path = os.path.join(input_dir, 'dataset_epigraf.json')
    epigraf_camera_angle = {
        'camera_angles': {}
    }

    with open(eg3d_json_path, 'r') as jfp:
        camera_pose_data = json.load(jfp)

    camera_params = camera_pose_data['labels']
    for key in camera_params.keys():
        camera_p = camera_params[key]
        camera_p = torch.tensor(camera_p)[:16]
        camera_p = camera_p.resize(4, 4)
        transition = camera_p[:, -1]
        # angle_p = torch.arctan((transition[2] / transition[1])) # pitch

        radius = torch.sqrt(transition[0] ** 2 + transition[1] ** 2 + transition[2] ** 2)
        # angle_y = torch.arctan((transition[2] / transition[0]))
        angle_y = torch.arctan((transition[0] / transition[2]))
        angle_p = torch.arccos(transition[1] / radius)
        # if angle_y < 0:
        #     angle_y = angle_y + np.pi
        epigraf_camera_angle['camera_angles'][key] = [angle_y.item(), angle_p.item(), 0.0]
        
    with open(epigraf_json_path, 'w') as epigraf_jfp:
        json.dump(epigraf_camera_angle, epigraf_jfp, indent=4)

    print('loaded')

if __name__ == '__main__':
    args = parse_args()
    transform_json(args)

