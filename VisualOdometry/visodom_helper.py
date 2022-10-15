import glob
import importlib
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from bokeh.io import output_file, save, show
from bokeh.layouts import column, gridplot, layout
from bokeh.models import Div, WheelZoomTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import ColumnDataSource, figure
from numpy.matlib import repmat
from PIL import Image

sensor = importlib.import_module(
    "driving-benchmarks.version084.carla.sensor", package="driving-benchmarks")


def visualize_paths(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
                                        px=pred_path[:, 0], py=pred_path[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                  x_axis_label="x", y_axis_label="y")
    fig1.circle("gtx", "gty", source=source, color="blue",
                hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green",
                hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source,
                    legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="above",
                  x_axis_label="frame", y_axis_label="error")
    fig2.circle("diffx", "diffy", source=source,
                hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    save(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))


def drawlines(img1, img2, lines, pts1, pts2, ts):

    r, c = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(np.array(img1), (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(np.array(img1), tuple(pt1), 5, color, -1)
        img2 = cv2.circle(np.array(img2), tuple(pt2), 5, color, -1)

        img1 = Image.fromarray(np.uint8(img1.copy()))
        img2 = Image.fromarray(np.uint8(img2.copy()))
    return img1, img2


def calculate_fundamental(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return pts1, pts2, F


def save_pointclouds(image, depths, ts, color=None, max_depth=0.1):
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
    pointcloud.save_to_disk("_out/pointclouds/%s/pointcloud_gt.ply" % ts)

    # Save depths as pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(depths).T)
    o3d.io.write_point_cloud(
        "_out/pointclouds/%s/pointcloud_depth.ply" % ts, pcd)


def estimate_errors(R_ests, t_ests, R_gts, t_gts, ts):
    R_ests = np.array(R_ests)
    t_ests = np.array(t_ests)
    R_gts = np.array(R_gts)
    t_gts = np.array(t_gts)

    rotation_err = []
    translation_err = []
    angles = []


    for i in range(len(R_ests)):
        delta_R = R_gts[i].T @ R_ests[i]
        delta_t = R_gts[i].T @ (t_ests[i] - t_gts[i])
        angle = math.acos((np.trace(delta_R) - 1) / 2)

        rotation_err.append(delta_R.sum())
        translation_err.append(delta_t.sum())
        angles.append(angle)

    timestep = list(range(len(R_ests)))
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('From top to bottom: Rotation, Translation, Angle')
    ax1.plot(timestep, rotation_err)
    ax2.plot(timestep, translation_err)
    ax3.plot(timestep, angles)

    plt.savefig("_out/errors/%s/error.png" % ts)


def create_video(ts):
    img_list = []
    base_path = os.getcwd()
    temp = glob.glob("_out/images/%s/concatenated/*.png" % ts)
    temp.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for filename in temp:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_list.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter("_out/videos/%s/video.avi" %
                          ts, fourcc, 15, size)
    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()
