import glob
import importlib
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_file, save, show
from bokeh.layouts import column, gridplot, layout
from bokeh.models import Div, WheelZoomTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import ColumnDataSource, figure
from numpy.matlib import repmat
from PIL import Image

sensor = importlib.import_module(
    "driving-benchmarks.version084.carla.sensor", package="driving-benchmarks")

class Helper():
    def __init__(self) -> None:
        pass


    def save_pointclouds(self, image, ts, name, color=None, max_depth=0.1):
        far = 1000.0  # max depth in meters.
        normalized_depth = image
        # (Intrinsic) K Matrix
        w, h = image.shape
        k = np.identity(3)
        k[0, 2] = w / 2.0
        k[1, 2] = h / 2.0
        k[0, 0] = k[1, 1] = w / \
            (2.0 * math.tan(90 * math.pi / 360.0))

        # 2d pixel coordinates
        pixel_length = w * h
        u_coord = repmat(np.r_[w-1:-1:-1],
                        h, 1).reshape(pixel_length)
        v_coord = repmat(np.c_[h-1:-1:-1],
                        1, w).reshape(pixel_length)
        if color is not None:
            color = color.reshape(pixel_length, 3)
            color = color.astype(np.float32) / 255
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

        # Formating the output to:
        # [[X1,Y1,Z1,R1,G1,B1],[X2,Y2,Z2,R2,G2,B2], ... [Xn,Yn,Zn,Rn,Gn,Bn]]
        if color is not None:
            # numpy.concatenate((numpy.transpose(p3d), color), axis=1)
            return sensor.PointCloud(
                5,
                np.transpose(p3d),
                color_array=color/255)
        # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
        pointcloud = sensor.PointCloud(5, np.transpose(p3d))
        pointcloud.save_to_disk("_out/pointclouds/%s/%s.ply" % (ts, name))

    def create_video(self, path, target):
        img_list = []
        base_path = os.getcwd()
        temp = glob.glob("%s*.png" % path)
        temp.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for filename in temp:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_list.append(img)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(target, fourcc, 15, (512,512))
        for i in range(len(img_list)):
            out.write(img_list[i])
        out.release()
