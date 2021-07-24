import tensorflow as tf
import cv2
import time
import argparse
import os
import csv
import pprint
import numpy as np

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_csv_dir', type=str, default='./output_csv')
args = parser.parse_args()

def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_csv_dir:
            if not os.path.exists(args.output_csv_dir):
                os.makedirs(args.output_csv_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            with open(args.output_csv_dir + "/motion_model.csv",'a') as write_file:
                writer = csv.writer(write_file)            

                # clip
                keypoint_coords[0,:,0] = keypoint_coords[0,:,0] - min(keypoint_coords[0,:,0])
                keypoint_coords[0,:,1] = keypoint_coords[0,:,1] - min(keypoint_coords[0,:,1])
                
                key = max(keypoint_coords[0,:,1])
                
                keypoint_coords[0,:,0] = keypoint_coords[0,:,0]/key
                keypoint_coords[0,:,1] = keypoint_coords[0,:,1]/key

                # normalize
                x_l2_norm = np.linalg.norm(keypoint_coords[0,:,0],ord=2)
                #pose_coords_x = (keypoint_coords[0,:,0] / x_l2_norm).tolist()
                pose_coords_x = (keypoint_coords[0,:,0]).tolist()
                y_l2_norm = np.linalg.norm(keypoint_coords[0,:,1],ord=2)
                #pose_coords_y = (keypoint_coords[0,:,1] / y_l2_norm).tolist()
                pose_coords_y = (keypoint_coords[0,:,1]).tolist()

                tpm_row = [f.replace(args.image_dir,'')] + pose_coords_x + pose_coords_y + keypoint_scores[0,:].tolist() + [pose_scores[0]]
                writer.writerow(tpm_row)

        print('Average FPS:', len(filenames) / (time.time() - start))
        print('Complete making CSV File!!')


if __name__ == "__main__":
    main()
