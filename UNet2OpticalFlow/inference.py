import argparse
import importlib
import json
import os
from pickletools import float8
from termios import VMIN
import time
import re
import traceback
import subprocess
import glob
import sys
from turtle import width
import torchvision.transforms.functional as transform
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt


import cv2
from matplotlib.pyplot import figure, gray
import torch
from PIL import Image
import pickle
import numpy as np
from unet_module import UNet, compute_max_depth

np.set_printoptions(threshold=sys.maxsize)
from numpy.matlib import repmat
from tqdm import tqdm
from helper import Helper
sensor = importlib.import_module(
    "driving-benchmarks.version084.carla.sensor", package="driving-benchmarks")


class GroundTruthCreator():
    def __init__(self, data_dir, cam) -> None:


        self.ts = str(time.time())
        self.data_dir = data_dir
        self.cam = cam


        self._create_dirs()

        
        self.images_bgr = self._load_images_bgr(os.path.join(data_dir, "CameraRGB%s" % str(cam)))
        self.depth_images = self._load_depth_images(os.path.join(data_dir, "CameraDepth%s" % str(cam)))
        self.seg_images = self._load_seg_images(os.path.join(data_dir, "CameraSemSeg%s" % str(cam)))
        self.opt_flow = self._load_opt_flow(os.path.join(data_dir, "CameraOptFlow%s" % str(cam)))
        self.opt_flow_vis = self._load_opt_flow_vis(os.path.join(data_dir, "CameraOptFlowVis%s" % str(cam)))
        # self.K = self._load_calib(os.path.join(data_dir, 'camera_intrinsic.json'))
        self.model = self._load_model("checkpoints/NN-Output1654165269.0380652_Unnormalized-small-set/last.ckpt")

        self.helper = Helper()


    def _create_dirs(self):
        if not os.path.isdir("_out/images/distancesrgb/%s/" %
                 self.ts):
            os.makedirs("_out/images/distancesrgb/%s/" %
                 self.ts)
            os.makedirs("_out/images/distancesmasked/%s/" %
                 self.ts)

        if not os.path.isdir("_out/videos/distancesrgb/%s" %self.data_dir):
            os.makedirs("_out/videos/distancesrgb/%s/" % self.data_dir)
        if not os.path.isdir("_out/videos/distancesmasked/%s/" % self.data_dir):
            os.makedirs("_out/videos/distancesmasked/%s/" % self.data_dir)
    
        if not os.path.isdir("_out/pickles/"):
            os.makedirs("_out/pickles")

        if not os.path.isdir("matplot_images/%s" % self.ts):
            os.makedirs("matplot_images/%s" % self.ts)

    @staticmethod
    def _load_calib(filepath):
        f = open(filepath)
        data = np.array(json.load(f))
        P = np.concatenate((data, np.zeros((3, 1))), axis=1)
        K = P[0:3, 0:3]
        return K

    @staticmethod
    def _load_images_bgr(filepath):
        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path)
                  for path in image_paths]
        return images 

    @staticmethod
    def _load_depth_images(filepath):

        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path) for path in image_paths]
        depths = []

        for image in images:
            array = image
            array = array.astype(np.float32)
            # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            depths.append(normalized_depth)

        return depths 
    
    @staticmethod
    def _load_seg_images(filepath):

        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path)
                  for path in image_paths]

        return images 

    @staticmethod
    def _load_opt_flow(filepath):
        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [np.load(path)
                  for path in image_paths]

        return images 

    @staticmethod
    def _load_opt_flow_vis(filepath):
        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path)
                  for path in image_paths]

        return images 

    @staticmethod
    def _load_model(path):
        model_module_params = {"learning_rate":0.00001, "mean": 1, "std":1}

        model = UNet.load_from_checkpoint(path)
        print(model.learning_rate)

        model.eval()

        return model

    def depth_to_local_point_cloud(self, image, name, color=None, k = np.eye(3),max_depth=1.1):
        """
        Convert an image containing CARLA encoded depth-map to a 2D array containing
        the 3D position (relative to the camera) of each pixel and its corresponding
        RGB color of an array.
        "max_depth" is used to omit the p+oints that are far enough.
        Reference: 
        https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/carla/image_converter.py
        """
        far = 1000.0  # max depth in meters.
        normalized_depth = image# depth_to_array(image)
        height, width = image.shape


        # 2d pixel coordinates
        pixel_length = width * height
        u_coord = repmat(np.r_[width-1:-1:-1],
                        height, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[height-1:-1:-1],
                        1, width).reshape(pixel_length)
        if color is not None:
            color = color.reshape(pixel_length, 3)
        normalized_depth = np.reshape(normalized_depth, pixel_length)

        # Search for pixels where the depth is greater than max_depth to
        # delete them
        max_depth_indexes = np.where(normalized_depth > max_depth)
        normalized_depth = np.delete(normalized_depth, max_depth_indexes)
        u_coord = np.delete(u_coord, max_depth_indexes)
        v_coord = np.delete(v_coord, max_depth_indexes)
        if color is not None:
            color = np.delete(color, max_depth_indexes, axis=0)

        # pd2 = [u,v,1]
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
        # P = [X,Y,Z]
        p3d = np.dot(np.linalg.inv(k), p2d)
        p3d *= normalized_depth * far
        
        p3d = np.transpose(p3d, (1,0))

        if color is not None: 
            color = color / 255.0

        return p3d, p2d, color

    def dense_optical_flow(self, img1, img2):
        prvs = img1
        hsv = np.zeros_like(prvs)
        hsv[...,1] = 255
        prvs = cv2.cvtColor(prvs,cv2.COLOR_BGR2GRAY)
        
        frame2 = img2
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None,pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=10, poly_n=5, poly_sigma=1.2,
                                            flags=0)
                                                

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


        cv2.imshow('frame2',np.concatenate((frame2[...,::-1], rgb), axis=1))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            pass
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)

        cv2.destroyAllWindows()
        return rgb

    def build_data(self, gray1, gray2):
        gray1 = transform.to_tensor(gray1)
        gray2 = transform.to_tensor(gray2)
        image = torch.cat((gray1, gray2), 0)
        return image.unsqueeze(0)


    def generate_video(self):
        os.chdir("matplot_images/%s"% self.ts)
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'video_name.mp4'
        ])
        for file_name in glob.glob("*.png"):
            os.remove(file_name)

    def build_matplot_images(self, gray_prev, gray_next, ground_truth, predicted, diff, i):

        def format_axes(fig):
            for i, ax in enumerate(fig.axes):
                ax.figure.set_size_inches(15, 15)

        fig = plt.figure(constrained_layout=True)

        gs2 = GridSpec(4, 3, figure=fig)

        ax1 = fig.add_subplot(gs2[0, 0])
        ax1.imshow(gray_prev,cmap='gray')
        ax1.set_title("frame1")
        ax2 = fig.add_subplot(gs2[0, 2])
        ax2.imshow(gray_next,cmap='gray')
        ax2.set_title("frame2")

        ax3 = fig.add_subplot(gs2[1, 0])
        ax3.imshow(ground_truth[:,:,0],cmap='gray', vmin=0, vmax=10)
        ax3.set_title("GT-Y")
        ax4 = fig.add_subplot(gs2[1, 1])
        ax4.imshow(ground_truth[:,:,1],cmap='gray', vmin=0, vmax=10)
        ax4.set_title("GT-X")
        ax5 = fig.add_subplot(gs2[1, 2])
        ax5.imshow(ground_truth[:,:,2],cmap='gray', vmin=0, vmax=10)
        ax5.set_title("GT-Z")

        ax6 = fig.add_subplot(gs2[2, 0])
        ax6.imshow(predicted[:,:,0],cmap='gray', vmin=0, vmax=10)
        ax6.set_title("Predicted-Y")
        ax7 = fig.add_subplot(gs2[2, 1])
        ax7.imshow(predicted[:,:,1],cmap='gray', vmin=0, vmax=10)
        ax7.set_title("Predicted-X")
        ax8 = fig.add_subplot(gs2[2, 2])
        ax8.imshow(predicted[:,:,2],cmap='gray', vmin=0, vmax=10)
        ax8.set_title("Predicted-Z")


        ax9 = fig.add_subplot(gs2[3, 0])
        ax9.imshow(diff[:,:,0],cmap='gray', vmin=0, vmax=10)
        ax9.set_title("Diff-Y")
        ax10 = fig.add_subplot(gs2[3, 1])
        ax10.imshow(diff[:,:,1],cmap='gray', vmin=0, vmax=10)
        ax10.set_title("Diff-X")
        ax11 = fig.add_subplot(gs2[3, 2])
        ax11.imshow(diff[:,:,2],cmap='gray', vmin=0, vmax=10)
        ax11.set_title("Diff-Z")

        fig.suptitle("GridSpec")
        format_axes(fig)

        plt.savefig('matplot_images/%s' % self.ts + "/file%02d.png" % i)

    def build_video_image(self, gray_prev, gray_next, ground_truth, predicted, diff):
        ## Build rows 
        print(gray_next.max())
        print(ground_truth.max())
        print(predicted.max())
        print(diff.max())

        row1 = np.hstack((np.hstack((gray_prev, np.zeros((512,612)))), gray_next))
        row2 = np.hstack((np.hstack((np.hstack((np.hstack((ground_truth[:,:,0], np.zeros((512, 50)))), ground_truth[:,:,1])),np.zeros((512, 50)))), ground_truth[:,:,2]))
        row3 = np.hstack((np.hstack((np.hstack((np.hstack((predicted[:,:,0], np.zeros((512, 50)))), predicted[:,:,1])),np.zeros((512, 50)))), predicted[:,:,2]))
        row4 = np.hstack((np.hstack((np.hstack((np.hstack((diff[:,:,0], np.zeros((512, 50)))), diff[:,:,1])),np.zeros((512, 50)))), diff[:,:,2]))
        full_image = np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((np.vstack((row1,np.zeros((50,1636)))), row2)), np.zeros((50,1636)))), row3)), np.zeros((50,1636)))), row4))
        scale_percent = 50 # percent of original size
        width = int(full_image.shape[1] * 80 / 100)
        height = int(full_image.shape[0] * 40 / 100)
        dim = (width, height)
       
        resized = cv2.resize(full_image, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Output: ", resized)
        cv2.waitKey(100)
        

    def create_mask(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.where(gray > 0,1 ,0) 

    def create_segmentation_mask(self, index):
        seg = self.seg_images[index][:,:,2] 
        mask = np.where(seg == 4 ) and np.where(seg == 10)
        return mask
    
    def get_diff(self, depth1, depth2, gt_opt_flow, index):
        ptc1, p2d1, _ = self.depth_to_local_point_cloud(depth1, "depth1")
        ptc2, p2d2, _ = self.depth_to_local_point_cloud(depth2, "depth2")

        ptc1 = np.reshape(ptc1, (512,512,3))
        ptc2 = np.reshape(ptc2, (512,512,3))
        differences = np.zeros(ptc1.shape)
        for i in range(len(depth1[0])):
            for j in range(len(depth1[1])):
                motion = gt_opt_flow[i][j].astype(int)
                try:
                    diff = ptc2[i+motion[1]][j+motion[0]] - ptc1[i][j]
                except: 
                    diff = ptc2[i][j] - ptc1[i][j]
                differences[i][j] = diff 
        labels_sum = np.abs(differences)
        labels_diff = differences
        test_image_mask = np.zeros((512,512,3))
        test_image = np.zeros((512,512,3))
        mask = self.create_segmentation_mask(index)
        label_sum_image = np.where(labels_sum > 0) and np.where(labels_sum < 50)


        test_image[label_sum_image] = labels_sum[label_sum_image] *255 / labels_sum[label_sum_image].max()
        try: 
            test_image_mask[mask] = labels_sum[mask] *255 / labels_sum[mask].max()
        except:
            pass

        diff = np.zeros((512,512,3)) 
        diff[mask] = differences[mask]
        diff = np.where(np.abs(diff) > 5, 0 ,diff)
        
        return diff



def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
   
    argparser.add_argument(
        '-s', '--scenario',
        metavar='S',
        default = "stationary",
        help= "Is the ego camera stationary or in motion?",
    )

    args = argparser.parse_args()
    print(args)

    data_dir_path = '/storage/group/intellisys/datasets/carla/3d_motion_with_opticalFlow/stationary/'
    if args.scenario == "stationary":
        data_dir = data_dir_path
    else:
        data_dir = data_dir_path.replace("stationary", "moving")

    dir_paths = [os.path.join(data_dir, dir)
                       for dir in sorted(os.listdir(data_dir))]
    
    print(dir_paths)

    for j, dir_path in enumerate(dir_paths):
        if j < 15:
            continue
        if args.scenario == "stationary":
            print("Start to generate Data for:  %s" % dir_path)
            gtc = GroundTruthCreator(dir_path, 0)

            depth_prev = gtc.depth_images[0]
            gray_prev = cv2.cvtColor(gtc.images_bgr[0], cv2.COLOR_BGR2GRAY)
            rgb_prev = gtc.images_bgr[0][...,::-1]
            for i in range(len(gtc.images_bgr)):

                data_package = {}
                if i == 0: 
                    continue
                else: 
                    depth_next = gtc.depth_images[i]
                    gray_next = cv2.cvtColor(gtc.images_bgr[i], cv2.COLOR_BGR2GRAY)
                    rgb_next = gtc.images_bgr[0][...,::-1]

                    ground_truth = gtc.get_diff(depth_prev, depth_next, gtc.opt_flow[i-1], i)

                    package = gtc.build_data(gray_prev, gray_next)
                    predicted = gtc.model(package)[0].cpu().detach().numpy()
                    predicted = np.swapaxes(predicted,0,2 )
                    predicted = np.swapaxes(predicted, 0, 1)

                    diff = np.abs(ground_truth - predicted)

                    gtc.build_matplot_images(gray_prev, gray_next, np.abs(ground_truth), np.abs(predicted), diff, i)
                    

                depth_prev = depth_next
                gray_prev = gray_next
                
                # save data package as yummi pickle 
                head_tail = os.path.split(dir_path)
                path = "/storage/remote/atcremers1/s0099/pickles/%s-%s-%s.pkl" % (head_tail[1], str(i), str(i-1))
                with open(path, "wb") as tf:
                    pickle.dump(data_package, tf)

            cv2.destroyAllWindows()
        gtc.generate_video()
            
        if args.scenario == "moving":
            print("Start to generate Data for:  %s" % dir_path)
            gtc = GroundTruthCreator(dir_path, 0)

            depth_prev = gtc.depth_images[0]
            for i in range(len(gtc.images_bgr)):

                data_package = {}
                if i == 0: 
                    continue
                else: 
                    depth_next = gtc.depth_images[i]


                    diff = gtc.get_diff_moving(depth_prev, depth_next, gtc.opt_flow[i-1], i)
                    data_package['depth1'] = depth_prev
                    data_package['depth2'] = depth_next
                    data_package['gt_label'] = diff 
                depth_prev = depth_next
                
                # save data package as yummi pickle 
                head_tail = os.path.split(dir_path)
                path = "/storage/remote/atcremers1/s0099/pickles/%s-%s-%s.pkl" % (head_tail[1], str(i), str(i-1))
                with open(path, "wb") as tf:
                    pickle.dump(data_package, tf)



            # cv2.imshow('frame2',np.concatenate((gtc.images_bgr[i][...,::-1], gtc.opt_flow_vis[i]), axis=1))
            # cv2.waitKey(200)
        
        # path1 = "_out/images/distancesrgb/%s/" % gtc.ts
        # path2 = "_out/images/distancesmasked/%s/" % gtc.ts
        # gtc.helper.create_video(path1, "_out/videos/distancesrgb/%s/video.avi" %  os.path.split(dir_path)[1])
        # gtc.helper.create_video(path2, "_out/videos/distancesmasked/%s/video.avi" %  os.path.split(dir_path)[1])

            
    

    

if __name__ == "__main__":
    try:
        main()
    except:
        print(traceback.format_exc())