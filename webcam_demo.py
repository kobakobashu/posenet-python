import tensorflow as tf
import cv2
import time
import argparse
import torch
from omegaconf import OmegaConf
from models.networks.LSTM import LSTM
import numpy as np

import posenet

#csvへの書き込み
import csv
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


cfg = OmegaConf.load('./configs/project/default.yaml')
print(cfg)
# cfg.model.initial_ckpt = "./model.pth"
# cfg.model.embedder.initial_ckpt = "./embedder.pth"
model = LSTM(cfg)


def main():
    with tf.Session() as sess:
        #make csv
        with open("./data/sample.csv", 'w') as f:
            writer = csv.writer(f)

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        leftShoulder_list = [0]*10
        rightShoulder_list = [0]*10
        leftElbow_list = [0]*10
        rightElbow_list = [0]*10
        leftWrist_list = [0]*10
        rightWrist_list = [0]*10
        leftHip_list = [0]*10
        rightHip_list = [0]*10

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        H = [0] * 510
        H_score = [0] * 10
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)

            # 相対化
            x0 = (keypoint_coords[0,5,:][1] + keypoint_coords[0,6,:][1]) / 2
            y0 = (keypoint_coords[0,5,:][0] + keypoint_coords[0,6,:][0]) / 2
            # 正規化
            y_min = keypoint_coords[0,0,:][0]
            y_max = (keypoint_coords[0,15,:][0] + keypoint_coords[0,16,:][0]) / 2
            y_diff = int(y_max - y_min)

            h = []
            for i in range(17):
                if y_diff != 0:
                    h.append((keypoint_scores[0, :][i]))
                    h.append((keypoint_coords[0,i,:][1] - x0) / y_diff)
                    h.append((- keypoint_coords[0,i,:][0] + y0) / y_diff)
                else:
                    h = [0] * 51
            if y_diff != 0:
                H_score.append(1)
            else:
                H_score.append(0)
            H.extend(h)
            H[0:51] = []
            H_score[0:1] = []
            # print(len(H))

            # for demo
            x = []
            x.append([H[i * 51 : (i+1) * 51] for i in range(10)])
            x = torch.from_numpy(np.array(x)).float()
            
            outputs = model.network(x)
            # print(outputs)
            if outputs[0][1] <= outputs[0][0] and outputs[0][2] <= outputs[0][0]:
                print("go forward")
            elif outputs[0][0] <= outputs[0][1] and outputs[0][2] <= outputs[0][1]:
                print("go forward a little")
            else:
                print("stop")

            #H_csv = H
            #H_csv.append(0)
            
            #vel = 0
            #print(H_score)
            #print(len(H))
            #print(H[31])
            #print(H_csv)
#            print("len(H)")
#            print(len(H))
            #print("len(H_csv)")
            #rint(len(H_csv))
            """
            #csvデータの作成
            H.append(2)
            with open("./data/sample.csv", 'a') as f:
                writer = csv.writer(f)
                writer.writerow(H)
            del H[-1]
            print(H)
            print(len(H))
            """
            
            """
            # プロトタイプ
            if sum(H_score) == 5:
                for i in range(4):
                    x_vec = H[i * 51 + 31] - H[(i + 1) * 51 + 31]
                    y_vec = H[i * 51 + 32] - H[(i + 1) * 51 + 32]
                    vel = (x_vec ** 2 + y_vec ** 2) ** 0.5
                #print(vel)
                if vel >= 0.08:
                    print("go forward")
                elif vel >= 0.02:
                    print("go forward a little")
                else:
                    print("stop")
                #print(vel)
            else:
                print("stop")
            #for time in range(4):
            """

            frame_count += 1
            #print('Average FPS: ', frame_count / (time.time() - start))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()