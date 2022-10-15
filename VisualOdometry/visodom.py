import argparse
import importlib
import json
import os
import time
import traceback

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from visodom_helper import *

image_converter = importlib.import_module(
    "driving-benchmarks.version084.carla.image_converter", package="driving-benchmarks")
sensor = importlib.import_module(
    "driving-benchmarks.version084.carla.sensor", package="driving-benchmarks")


class VisualOdometry():
    def __init__(self, data_dir, start, end, singleframe):
        self.ts = str(time.time())

        if not os.path.isdir("_out/images/%s" % self.ts):
            os.makedirs(("_out/images/%s/concatenated" % self.ts))
            os.makedirs(("_out/images/%s/epilines" % self.ts))

        if not os.path.isdir("_out/htmls/%s" % self.ts):
            os.makedirs("_out/htmls/%s" % self.ts)

        if not os.path.isdir("_out/pointclouds/%s" % self.ts):
            os.makedirs("_out/pointclouds/%s" % self.ts)

        if not os.path.isdir("_out/errors/%s" % self.ts):
            os.makedirs("_out/errors/%s" % self.ts)

        if not os.path.isdir("_out/videos/%s" % self.ts):
            os.makedirs("_out/videos/%s" % self.ts)

        self.K, self.P = self._load_calib(
            os.path.join(data_dir, 'camera_intrinsic.json'))
        self.gt_poses = self._load_poses(os.path.join(
            data_dir, "transforms.json"), start, end, singleframe)
        self.images = self._load_images(os.path.join(
            data_dir, "CameraRGB0"), start, end, singleframe)
        self.depth_images = self._load_depth_images(os.path.join(
            data_dir, "CameraDepth0"), start, end, singleframe)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file
        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        f = open(filepath)
        data = np.array(json.load(f))
        P = np.concatenate((data, np.zeros((3, 1))), axis=1)
        K = P[0:3, 0:3]

        return K, P

    @staticmethod
    def _load_poses(filepath, start, end, singleframe):
        """
        Loads the GT poses
        Parameters
        ----------
        filepath (str): The file path to the poses file
        Returns
        -------
        poses (ndarray): The GT poses
        """

        f = open(filepath)
        data = json.load(f)
        poses = []

        for t in data.values():
            poses.append(t[0])

        if not singleframe:
            return poses[start:end]
        if singleframe:
            return [poses[start], poses[end]]

    @staticmethod
    def _load_images(filepath, start, end, singleframe):
        """
        Loads the images
        Parameters
        ----------
        filepath (str): The file path to image dir
        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                  for path in image_paths]

        if not singleframe:
            return images[start:end]

        if singleframe:
            return [images[start], images[end]]

    @staticmethod
    def _load_depth_images(filepath, start, end, singleframe):
        """
        Loads the images
        Parameters
        ----------
        filepath (str): The file path to image dir
        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file)
                       for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path) for path in image_paths]
        normalized = []

        for image in images:
            array = image
            array = array.astype(np.float32)
            # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            normalized.append(normalized_depth)

        if not singleframe:
            return normalized[start:end]
        if singleframe:
            return [normalized[start], normalized[end]]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector
        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        Parameters
        ----------
        i (int): The current frame
        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor=-1,  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(
            self.images[i], kp1, self.images[i-1], kp2, good, None, **draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)
        img = Image.fromarray(np.uint8(img3.copy()))
        img.save("_out/images/%s/concatenated/concatenated%s.png" %
                 (self.ts, str(i)))

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2, i):
        """
        Calculates the transformation matrix
        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        try:
            pts_1, pts_2, F = calculate_fundamental(q1, q2)

            lines1 = cv2.computeCorrespondEpilines(
                pts_2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)

            img5, img6 = drawlines(
                self.images[i-1], self.images[i], lines1, pts_1, pts_2, self.ts)
            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines2 = cv2.computeCorrespondEpilines(
                pts_1.reshape(-1, 1, 2), 1, F)
            lines2 = lines2.reshape(-1, 3)
            img3, img4 = drawlines(
                self.images[i-1], self.images[i], lines2, pts_2, pts_1, self.ts)
            img5.save("_out/images/%s/epilines/epiline%s_1.png" %
                      (self.ts, str(i)))
            img3.save("_out/images/%s/epilines/epiline%s_2.png" %
                      (self.ts, str(i)))
        except:
            print(traceback.format_exc())
            pass

        # Essential matrix
        E, mask = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        _, R, t, mask = cv2.recoverPose(E, q1, q2, self.K, mask=mask)

        # Get transformation matrix
        T = self._form_transf(R, np.squeeze(t))
        P = np.matmul(np.concatenate(
            (self.K, np.zeros((3, 1))), axis=1), T)
        hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)

        # Un-homogenize
        depths = hom_Q1[:3, :] / hom_Q1[3, :]

        return T, R, t, depths


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-dd', '--data_dir',
        metavar="P",
        default='episode_000',
        help='path to the images')
    argparser.add_argument(
        '-s', '--start',
        metavar='S',
        default=0,
        help='frame to start from')
    argparser.add_argument(
        '-e', '--end',
        metavar='E',
        default=499,
        help='the ending frame')
    argparser.add_argument(
        '-sf', '--singleframe',
        metavar='F',
        default="False",
        help='Only use two images referenced in start and end')
    args = argparser.parse_args()

    print(args)

    if args.singleframe.lower() in ['true', '1', 't', 'y', 'yes']:
        args.singleframe = True
    else:
        args.singleframe = False

    vo = VisualOdometry(args.data_dir, int(args.start),
                        int(args.end), args.singleframe)

    gt_path = []
    estimated_path = []
    R_ests = []
    t_ests = []
    R_gts = []
    t_gts = []
    
    print(len(vo.gt_poses))

    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf, R, t, depths = vo.get_pose(q1, q2, i)
            gt_pose_change = np.array(gt_pose) - np.array(vo.gt_poses[i-1])
            R_gt = gt_pose_change[:3, :3]
            t_gt = gt_pose_change[:3, 3]
            R_gts.append(R_gt)
            t_gts.append(t_gt)
            R_ests.append(R)
            t_ests.append(np.squeeze(t))

            # save_pointclouds(vo.depth_images[i], depths, vo.ts)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((np.array(gt_pose)[0, 3], np.array(gt_pose)[2, 3]))
        estimated_path.append(
            (np.array(cur_pose)[0, 3], np.array(cur_pose)[2, 3]))
    visualize_paths(gt_path, estimated_path, "Visual Odometry",
                    file_out="_out/htmls/%s/poses.html" % vo.ts)
    estimate_errors(R_ests, t_ests, R_gts, t_gts, vo.ts)
    create_video(vo.ts)


if __name__ == "__main__":
    try:
        main()
    except:
        print(traceback.format_exc())
