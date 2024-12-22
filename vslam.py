"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography (planar) + minimal monocular odometry demo.
"""

import cv2
import numpy as np
import torch

from time import time, sleep
import argparse, sys
import threading
from modules.xfeat import XFeat
import queue
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream. only 640x480 or 320x224 resolution')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream. only 640x480 or 320x224 resolution')
    parser.add_argument('--max_kpts', type=int, default=3000, help='Maximum number of keypoints.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat',
                        help='Local feature detection method to use.')
    parser.add_argument('--cam', type=int, default=0, help='Webcam device number.')
    parser.add_argument('--video_path', type=str, default="", help='video path.')
    parser.add_argument('--inference_type', type=str, default='hailo', help='"hailo" or "torch" or "onnx"')
    return parser.parse_args()

class FrameGrabber():
    def __init__(self, cap, width, height):
        super().__init__()
        self.cap = cap
        _, self.frame = self.cap.read()
        self.second_last_frame = self.frame
        self.running = False
        self.width = width
        self.height = height

    def run(self):
        # self.running = True
        # while self.running:
        ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream ended?).")
            self.stop()
            # break
        self.second_last_frame = self.frame
        self.frame = frame
        # sleep(0.1)

    def stop(self):
        self.running = False
        self.cap.release()

    def get_last_frame(self):
        self.run()
        # Safeguard: reshape or resize if needed
        self.frame = np.resize(self.frame, (self.height, self.width, 3))
        return self.frame
    
    def get_second_last_frame(self):
        return self.second_last_frame

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts, width, height, device):
    if method == "ORB":
        return Method(
            descriptor=cv2.ORB_create(max_kpts, fastThreshold=10),
            matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        )
    elif method == "SIFT":
        return Method(
            descriptor=cv2.SIFT_create(
                max_kpts, contrastThreshold=-1, edgeThreshold=1000
            ),
            matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        )
    elif method == "XFeat":
        # XFeat instance used both as 'descriptor' and 'matcher' depending on usage
        return Method(
            descriptor=CVWrapper(
                XFeat(top_k = max_kpts, width=width, height=height, device=device)
            ),
            matcher=XFeat(width=width, height=height, device=device)
        )
    else:
        raise RuntimeError("Invalid Method.")

class MatchingDemo:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.device = args.inference_type
        self.ref_frame = None
        self.ref_precomp = [[],[]]

        # Setup camera / video
        if args.video_path == "":
            self.cap = cv2.VideoCapture(args.cam)
            self.setup_camera()
        else:
            self.cap = cv2.VideoCapture(args.video_path)

        # Init FrameGrabber thread
        self.frame_grabber = FrameGrabber(self.cap, self.width, self.height)
        # self.frame_grabber.start()

        # Homography params (still used if needed)
        self.margin = 100
        self.min_inliers = self.margin
        self.ransac_thr = 4.0

        # Corners used for homography (if you still want the overlay)
        self.corners = [
            [self.margin, self.margin],
            [self.width-self.margin, self.margin],
            [self.width-self.margin, self.height-self.margin],
            [self.margin, self.height-self.margin]
        ]

        # FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30  # average FPS over this number of frames

        # Local feature method (XFeat, SIFT, ORB, etc.)
        self.method = init_method(
            args.method, max_kpts=args.max_kpts,
            width=self.width, height=self.height,
            device=self.device
        )
        
        # Font for debug text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 2

        # Global rotation/translation
        self.global_R = np.eye(3)
        self.global_t = np.zeros((3, 1))

        # For minimal VO trajectory visualization
        # ----------------------------------------
        self.trajectory = np.zeros((600, 600, 3), dtype=np.uint8)
        self.traj_center = (300, 300)  # Center in the middle
        self.scale = 1.0               # Adjust scale if needed

        # We store the last compute state from XFeat if needed
        self.prev_compute = None

        self.window_name = "Real-time matching - Press 'q' to exit."

    def setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def match_and_draw_visual_flow(self, current_frame):
        """
        1) Match features between current_frame and the second-last frame.
        2) Estimate camera motion with Essential Matrix.
        3) Update global pose.
        4) Draw the updated camera position on `self.trajectory`.
        """
        if self.prev_compute is None:
            # First run. Match with the last frame in memory.
            # match_xfeat_star returns (points_flow1, points_flow2, prev_compute, _null)
            points_flow1 ,points_flow2, self.prev_compute, _ = \
                self.method.descriptor.mtd.match_xfeat_star(
                    np.copy(current_frame),
                    self.frame_grabber.get_last_frame()
                ) 
        else:
            # Bootstrap version for faster subsequent matches
            # match_xfeat_star_bootstrap returns (points_flow1, points_flow2, updated_prev_compute)
            points_flow1, points_flow2, self.prev_compute = \
                self.method.descriptor.mtd.match_xfeat_star_bootstrap(
                    self.prev_compute,
                    np.copy(current_frame)
                )

        # Fictitious intrinsics for demonstration
        fx = 700
        fy = 700
        cx = self.width / 2
        cy = self.height / 2

        K = np.array([[fx,   0, cx],
                      [ 0,  fy, cy],
                      [ 0,   0,  1]], dtype=np.float64)

        # Estimate motion via Essential matrix
        E, mask = cv2.findEssentialMat(
            points_flow1, points_flow2, K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return  # Could not find a valid Essential matrix this frame

        _, R, t, mask = cv2.recoverPose(E, points_flow1, points_flow2, K)

        # Print some debug info (optional)
        print("Essential Matrix:\n", E)
        print("Rotation Matrix:\n", R)
        print("Translation Vector:\n", t)

        # Update global pose
        # global_t = global_t + global_R * t
        # global_R = R * global_R
        self.global_t += self.global_R @ t
        self.global_R = R @ self.global_R

        # Draw the current position on self.trajectory
        # We'll interpret X as global_t[0], Z as global_t[2].
        x = self.global_t[0, 0]
        z = self.global_t[2, 0]

        traj_x = int(self.traj_center[0] + self.scale * x)
        traj_y = int(self.traj_center[1] + self.scale * z)

        cv2.circle(self.trajectory, (traj_x, traj_y), 2, (0, 255, 0), -1)

        # Show the frame and the trajectory
        cv2.imshow("Frame", current_frame)
        cv2.imshow("Trajectory", self.trajectory)

    def main_loop(self):
        self.current_frame = self.frame_grabber.get_last_frame()
        self.ref_frame = self.current_frame.copy()  # not strictly needed now

        while True:
            if self.current_frame is None:
                break

            t0 = time()
            self.match_and_draw_visual_flow(self.current_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            # Get a new frame from the grabber
            self.current_frame = self.frame_grabber.get_last_frame()

            # Measure avg FPS
            self.time_list.append(time() - t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
            print("FPS: {:.2f}".format(self.FPS))

        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = MatchingDemo(args=argparser())
    demo.main_loop()
