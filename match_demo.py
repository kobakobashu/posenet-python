import tensorflow as tf
import cv2
import csv
import time
import argparse
import csv
import posenet
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--match_model', type=str, default='./output_csv/motion_model.csv')
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def weightedDistanceMatching(poseVector1_x, poseVector1_y, vector1Confidences, vector1ConfidenceSum, poseVector2):
#   First summation
    summation1 = 1.0 / vector1ConfidenceSum
#   Second summation
    summation2 = 0
    for indent_num in range(len(poseVector1_x)):
        tempSum = vector1Confidences[indent_num] * ( abs(poseVector1_x[indent_num] -  poseVector2[indent_num]) + abs(poseVector1_y[indent_num] -  poseVector2[indent_num + len(poseVector1_x)]))
        summation2 = summation2 + tempSum
    return summation1 * summation2

def main():
    with tf.Session() as sess:
        with open(args.match_model) as f:
            reader = csv.reader(f)
            motion_model = [row for row in reader]
        for i in range(len(motion_model)):
            motion_model[i][1:] = list(map(lambda x:float(x), motion_model[i][1:]))

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
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

            if pose_scores[0] > 0.5:
                # clip
                pose_coords_x = keypoint_coords[0,:,0] - min(keypoint_coords[0,:,0])
                pose_coords_y = keypoint_coords[0,:,1] - min(keypoint_coords[0,:,1])

                # normalize
                x_l2_norm = np.linalg.norm(keypoint_coords[0,:,0],ord=2)
                pose_coords_x = (pose_coords_x / x_l2_norm).tolist()
                y_l2_norm = np.linalg.norm(keypoint_coords[0,:,1],ord=2)
                pose_coords_y = (pose_coords_y / y_l2_norm).tolist()

                distance_min = 1000000
                min_num = -1
#                motion_model[min_num][0]="Miss"
                for teach_num in range(len(motion_model)):
                    distance = weightedDistanceMatching(pose_coords_x, pose_coords_y, keypoint_scores[0,:], pose_scores[0], motion_model[teach_num][1:35])
                    # distance = cos_sim(pose_coords_x + pose_coords_y, motion_model[teach_num][1:35])
                    if distance < distance_min:
                        distance_min = distance
                        min_num = teach_num
#                    print(distance)
                cv2.putText(display_image, motion_model[min_num][0] + "aiueo", (int(keypoint_coords[0,:,0][0]),int(keypoint_coords[0,:,0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=2)
                print(motion_model[min_num][0])

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)
            # TODO this isn't particularly fast, use GL for drawing and display someday...
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()