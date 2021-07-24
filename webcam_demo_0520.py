import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
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
        H = [0] * 255
        H_score = [0] * 5
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
            vel = 0
            #print(H_score)
            #print(len(H))
            #print(H[31])
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
                

#             leftShoulder_list.append(keypoint_coords[0,5,1],keypoint_coords[0,5,0])
#             print(leftShoulder_list)
#             rightShoulder_list.append(keypoint_coords[0,6,:])
#             leftElbow_list.append(keypoint_coords[0,7,:])
#             rightElbow_list.append(keypoint_coords[0,8,:])
#             leftWrist_list.append(keypoint_coords[0,9,:])
#             rightWrist_list.append(keypoint_coords[0,10,:])
#             leftHip_list.append(keypoint_coords[0,11,:])
#             rightHip_list.append(keypoint_coords[0,12,:])

#             leftShoulder_list.pop(0)
#             rightShoulder_list.pop(0)
#             leftElbow_list.pop(0)
#             rightElbow_list.pop(0)
#             leftWrist_list.pop(0)
#             rightWrist_list.pop(0)
#             leftHip_list.pop(0)
#             rightHip_list.pop(0)
#             #print(keypoint_coords[0,5,:])
#             #print(keypoint_scores[0, :][0])
#             #print(len(keypoint_scores[0, :]))
#             #print(pose_scores[0])
#             #print(keypoint_coords[0,10,:])
#             print("-----")
            
            #print(rightHip_list)
            
#             shoulder = (keypoint_coords[0,5,:]+keypoint_coords[0,6,:])/2
#             hip = (keypoint_coords[0,11,:]+keypoint_coords[0,12,:])/2            
#             vec_1 = keypoint_coords[0,7,:]-shoulder
#             vec_2 = keypoint_coords[0,9,:]-keypoint_coords[0,7,:]
#             vec_3 = keypoint_coords[0,8,:]-shoulder
#             vec_4 = keypoint_coords[0,10,:]-keypoint_coords[0,8,:]
#             vec_5 = hip-shoulder
#             vec_6 = keypoint_coords[0,13,:]-hip
#             vec_7 = keypoint_coords[0,15,:]-keypoint_coords[0,13,:]
#             vec_8 = keypoint_coords[0,14,:]-hip
#             vec_9 = keypoint_coords[0,16,:]-keypoint_coords[0,14,:]
            
#             slope_1 = vec_1[0]/vec_1[1]
#             slope_2 = vec_2[0]/vec_2[1]
#             slope_3 = vec_3[0]/vec_3[1]
#             slope_4 = vec_4[0]/vec_4[1]
#             slope_5 = vec_5[0]/vec_5[1]
#             slope_6 = vec_6[0]/vec_6[1]
#             slope_7 = vec_7[0]/vec_7[1]
#             slope_8 = vec_8[0]/vec_8[1]
#             slope_9 = vec_9[0]/vec_9[1]

#             if -1<slope_1 and slope_1<1 and -1<slope_2 and slope_2<1 and -1<slope_3 and slope_3<1 and -1<slope_4 and slope_4<1 and vec_2[1]>0 and vec_4[1]<0:
#                 print("T")
#             elif -1<slope_1 and slope_1<1 and -1<slope_2 and slope_2<1 and -1<slope_3 and slope_3<1 and -1<slope_4 and slope_4<1 and vec_2[1]>0 and vec_4[1]>0:
#                 if slope_7>1 or slope_7<-1:
#                     if slope_9>1 or slope_9<-1:
#                         print("J")
#                 elif slope_7<1 and slope_7>-1 and slope_9<1 and slope_9>-1:
#                     print("S")
#             elif -1<slope_1 and slope_1<1 and -1<slope_2 and slope_2<1 and -1<slope_3 and slope_3<1 and -1<slope_4 and slope_4<1 and vec_2[1]<0 and vec_4[1]<0:
#                 if slope_7>1 or slope_7<-1:
#                     if slope_9>1 or slope_9<-1:
#                         print("L")
#                 elif slope_7<1 and slope_7>-1 and slope_9<1 and slope_9>-1:
#                     print("Z")
#             elif -1>slope_1 or slope_1>1:
#                 if -1>slope_2 or slope_2>1:
#                     if -1>slope_3 or slope_3>1:
#                         if -1>slope_4 or slope_4>1:
#                             if -1>slope_7 or slope_7>1:
#                                 if -1>slope_9 or slope_9>1:
#                                     print("I")
#             else:
#                 print("not I,T,J,S,L,Z")
            
            
            frame_count += 1
            #print('Average FPS: ', frame_count / (time.time() - start))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()