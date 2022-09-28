import traceback
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from collections import Counter
from scipy import ndimage
import cv2
import imageio
import matplotlib.pyplot as plt
import time
import os 

# to enable interaction with plots (e.g. zoom, translation) 
import mpld3

#----------additional imports ------------------
from PIL import Image
import imutils

class FeatureDetector():

    def __init__(self, show:bool=True, save:bool=False) -> None:
        self.show = show
        self.save = save
        self.ts = str(time.time())
        self._create_dirs()

    def _create_dirs(self):
        try:
            os.makedirs("../_out/img")
            os.makedirs("../_out/img/%s" % self.ts)
        except:
            pass
    
    @staticmethod
    def convert_to_grayscale(path):
        return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2GRAY)

    def compute_descriptors(self, I, corners):
        # get the gradients of the image either with numpy or using a sobel filter
        ts2 = str(time.time())
        dy, dx = np.gradient(I)
        
        
        h, w = I.shape
        dx = np.pad(dx, 8, mode='edge')
        dy = np.pad(dy, 8, mode='edge')

        if self.save:
            img_xx = Image.fromarray(np.uint8(dx))
            img_xx.save("../img/%s/padd_%s.jpeg" % (self.ts, ts2))

        magnitude_matrices = list()
        orientation_matrices = list()
        for corner in corners: 
            
            dx_win = dx[int(corner[0]+8) - 8:int(corner[0]+8)+8 , int(corner[1]+8) - 8:int(corner[1]+8)+8]
            dy_win = dy[int(corner[0]+8) - 8:int(corner[0]+8)+8 , int(corner[1]+8) - 8:int(corner[1]+8)+8]
            magnitude_matrix = np.sqrt(dx_win**2 + dy_win**2)
            orientation_matrix = np.arctan2(dy_win ,dx_win)
            magnitude_matrices.append(magnitude_matrix)
            orientation_matrices.append(orientation_matrix)

        histograms = list()
        for i in range(0, len(magnitude_matrices)):
            corner_histo = list()
            for x in range(0,16,4):
                for y in range(0,16,4):
                    window_histo = np.zeros(8)
                    window_orientation = orientation_matrices[i][x:x+4, y:y+4]
                    window_magnitude = magnitude_matrices[i][x:x+4, y:y+4]
                    
                    for l in range(4):
                        for w in range(4):
                            if ( -np.pi/8 < window_orientation[l][w] < np.pi/8):
                                window_histo[0] += window_magnitude[l][w]
                            elif( np.pi/8 < window_orientation[l][w] < 3*np.pi/8):
                                window_histo[1] += window_magnitude[l][w]
                            elif( 3*np.pi/8 < window_orientation[l][w] < 5*np.pi/8):
                                window_histo[2] += window_magnitude[l][w]
                            elif( 5*np.pi/8 < window_orientation[l][w] < 7*np.pi/8):
                                window_histo[3] += window_magnitude[l][w]
                            elif( -3*np.pi/8 < window_orientation[l][w] < -np.pi/8):
                                window_histo[7] += window_magnitude[l][w]
                            elif( -5*np.pi/8 < window_orientation[l][w] < -3*np.pi/8):
                                window_histo[6] += window_magnitude[l][w]
                            elif( -7*np.pi/8 < window_orientation[l][w] < -5*np.pi/8):
                                window_histo[5] += window_magnitude[l][w]
                            else:
                                window_histo[4] += window_magnitude[l][w]
                    corner_histo = np.concatenate((corner_histo, window_histo))
            corner_histo = corner_histo / np.sqrt(np.sum(corner_histo**2))
            histograms.append(corner_histo)
        histograms = np.array(histograms)
            
        return histograms

    def compute_corners(self, I, thresh, filter='gauss', kernel_size = 3, distance=None):
        # get the gradients of the image either with numpy or using a sobel filter
        ts2 = str(time.time())
        
        dy, dx = np.gradient(I)
        
        # xx and yy of structure matrix
        i_xx = dx **2
        i_yy = dy **2 

        kernel = None
        if filter =='gauss':
            l = kernel_size
            sig = 1
            ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
            kernel = np.outer(gauss, gauss)
            kernel = kernel / np.sum(kernel)

        # convolve over a certain area W 
        s_xx = ndimage.convolve(i_xx, kernel, mode='constant', cval=0.0)
        s_yy = ndimage.convolve(i_yy, kernel, mode='constant', cval=0.0)
        d_xx = ndimage.convolve(dx, kernel, mode='constant', cval=0.0)
        d_yy = ndimage.convolve(dy, kernel, mode='constant', cval=0.0)

        # normalize for better intuition and visualization
        s_xx = s_xx * (255/np.amax(s_xx))
        s_yy = s_yy * (255/np.amax(s_yy))
        d_xx = d_xx * (255/np.amax(d_xx))
        d_yy = d_yy * (255/np.amax(d_yy))

        if self.save:
            img_xx = Image.fromarray(np.uint8(s_xx))
            img_xx.save("../_out/img/%s/s_xx_%s.jpeg" % (self.ts, ts2))
            img_yy = Image.fromarray(np.uint8(s_yy))
            img_yy.save("../_out/img/%s/s_yy_%s.jpeg" % (self.ts, ts2))
            img_both = Image.fromarray(np.uint8((s_xx + s_yy)/2))
            img_both.save("../_out/img/%s/s_%s.jpeg" % (self.ts, ts2))

        # build tensor with structure matrices 
        h, w = I.shape
        min_eig_values = np.zeros(I.shape)
        for i in range(0, h):
            for j in range(0, w):
                structure_matrix = np.zeros((2,2))
                structure_matrix[0][0] = s_xx[i][j]
                structure_matrix[1][1] = s_yy[i][j]
                structure_matrix[0][1] = d_xx[i][j] * d_yy[i][j]
                structure_matrix[1][0] = d_xx[i][j] * d_yy[i][j]
                min_eig_values[i][j] = np.trace(structure_matrix)/2 - 0.5*np.sqrt(np.trace(structure_matrix)**2 - 4*np.linalg.det(structure_matrix))
                
        corner_map = np.where(abs(min_eig_values) > thresh, 1, 0)

        if self.save:
            img_corner_map = Image.fromarray(np.uint8(np.where(corner_map == 1 , 255, 0)))
            img_corner_map.save("../_out/img/%s/corners_%s.jpeg" % (self.ts, ts2))

        corners = np.argwhere(corner_map == 1)

        # filter points with short distance
        if filter != None: 
            single_corners = []
            for i in corners:
                neighbors = [i]
                for j in corners: 
                    dist_vec = abs(i - j)
                    dist = np.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
                    if (dist < distance and dist != 0): 
                        neighbors.append(j)
                if len(neighbors) == 0:
                    single_corners.append(list(i))
                    continue
                # calc mean of neighbors 
                neighbors_sum = np.array([0, 0])
                for i in neighbors:
                    neighbors_sum += i
                mean = neighbors_sum / (len(neighbors) + np.finfo(float).eps)
                mean = list(np.around(mean, decimals=0))
                if(mean not in single_corners):
                    single_corners.append(mean)

            single_corners_map = np.zeros(I.shape)
            for corner in single_corners:
                single_corners_map[int(corner[0])][int(corner[1])] = 255

            if self.save: 
                img = Image.fromarray(np.uint8(single_corners_map))
                img.save("../img/%s/corners_dist_%s.jpeg" % (self.ts, ts2))
            corners = single_corners

        return corners

    def compute_matches(self, descr1, descr2, ord):
        # go over all keypoints and calc distance of the histograms
        distances = []
        matches = []
        dist_thresh = 0.6
        for i, histo_1 in enumerate(descr1): 
            smallest_dist = np.finfo(float).max
            smallest_idx = None
            for j, histo_2 in enumerate(descr2):
                dist = np.linalg.norm(histo_1 - histo_2, ord)
                if dist < smallest_dist:
                    smallest_dist = dist
                    smallest_idx = j
            if smallest_dist > dist_thresh: 
                continue
            
            match = (i, smallest_idx)
            distances.append([match, smallest_dist])
            matches.append(match) 
        
        # ---------outlier handling--------------
        # make matches unique
        matches = list(set(matches))

        # count how often a keypoint is used
        second_element = Counter([y for (x,y) in matches])
    
        # get keypoints that are matched multiple times 
        keypoints = dict((k, v) for k, v in second_element.items() if v > 1)
        keypoints = list(keypoints.keys())
        
        for i in keypoints: 
            tuples = [tuple for tuple in matches if tuple[1] == i]
            smallest_dist = np.finfo(float).max
            smallest_tuple = None
            for tuple in tuples: 
                dist = [element for element in distances if element[0] == tuple]
                if dist[0][1] < smallest_dist:
                    smallest_tuple = dist[0][0]
                
                matches.remove(tuple)
            matches.append(smallest_tuple)
        return matches


    def plot_matches(self, I1, I2, corners1, corners2, matches, title="Matches"):
        # plot matching corners individually 

        i1 = np.copy(I1)
        i2 = np.copy(I2)

        matches_1_mask = [tuple[0] for tuple in matches]
        matches_2_mask = [tuple[1] for tuple in matches]
        matched_corners_1 = [corners1[i] for i in matches_1_mask]
        matched_corners_2 = [corners2[i] for i in matches_2_mask]


        f, axarr = plt.subplots(1, 2)
        for i in matched_corners_1:
            cv2.circle(i1, (int(i[1]), int(i[0])),radius=3,color=(0,0,255),thickness=-1)
        ts2 = str(time.time())
        if self.save:
            img = Image.fromarray(np.uint8(i1))
            img.save("../_out/img/%s/image_1_matching_circles_%s.jpeg" % (ts, ts2))
        if self.show:
            axarr[0].imshow(i1)
        

        for i in matched_corners_2:
            cv2.circle(i2, (int(i[1]), int(i[0])),radius=3,color=(0,0,255),thickness=-1)
        ts2 = str(time.time())
        if self.save:
            img = Image.fromarray(np.uint8(i2))
            img.save("../_out/img/%s/image_2_matching_circles_%s.jpeg" % (self.ts, ts2))
        if self.show:
            axarr[1].imshow(i2)
            plt.show()

        h, w ,c = I1.shape   
        concatenated = np.concatenate((I1, I2), axis=1)
        for match in matches:
            cv2.circle(concatenated, (int(corners1[match[0]][1]), int(corners1[match[0]][0])),radius=3,color=(0,255,0),thickness=-1)
            cv2.circle(concatenated, (int(corners2[match[1]][1]) + w, int(corners2[match[1]][0])),radius=3,color=(0,255,0),thickness=-1)
            cv2.line(concatenated, (int(corners1[match[0]][1]), int(corners1[match[0]][0])), (int(corners2[match[1]][1]) + w, int(corners2[match[1]][0])), (0,255,0), 1)
        ts2 = str(time.time())
        if self.save:
            img = Image.fromarray(np.uint8(concatenated.copy()))
            img.save("../_out/img/%s/concatenated_%s.jpeg" % (self.ts, ts2))
        if self.show:
            plt.imshow(concatenated)
            plt.show()

    def stitch_images(self,img_paths, thresh, resize):
        homographies = []
        imgs = []
        for i in range(len(img_paths)):
            if (i == len(img_paths) - 1):
                img_1 = cv2.imread(img_paths[i])[:,:,::-1]
                width = int(img_1.shape[1] * resize / 100)
                height = int(img_1.shape[0] * resize / 100)
                dim = (width, height)
                resized_1 = cv2.resize(img_1, dim, interpolation = cv2.INTER_AREA)
                imgs.append(resized_1)
                break

            img_1 = cv2.imread(img_paths[i])[:,:,::-1]
            width = int(img_1.shape[1] * resize / 100)
            height = int(img_1.shape[0] * resize / 100)
            dim = (width, height)
            resized_1 = cv2.resize(img_1, dim, interpolation = cv2.INTER_AREA)
            imgs.append(resized_1)
            gray_1 = cv2.cvtColor(resized_1,cv2.COLOR_RGB2GRAY)

            img_2 = cv2.imread(img_paths[i+1])[:,:,::-1]
            width = int(img_2.shape[1] * resize / 100)
            height = int(img_2.shape[0] * resize / 100)
            dim = (width, height)
            resized_2 = cv2.resize(img_2, dim, interpolation = cv2.INTER_AREA)
            imgs.append(resized_2)
            gray_2 = cv2.cvtColor(resized_2,cv2.COLOR_RGB2GRAY)

            corners_1 = self.compute_corners(gray_1, thresh, filter='gauss', kernel_size = 3, distance=5)
            corners_2 = self.compute_corners(gray_2, thresh, filter='gauss', kernel_size = 3, distance=5)

            descriptors_1 = self.compute_descriptors(gray_1, corners_1)
            descriptors_2 = self.compute_descriptors(gray_2, corners_2)

            matches = self.compute_matches(descriptors_1, descriptors_2, 2)
            self.plot_matches(resized_1, resized_2, corners_1, corners_2, matches)

            src_points = []
            dst_points = []

            for match in matches:
                dst_points.append((int(corners_1[match[0]][1]), int(corners_1[match[0]][0])))
                src_points.append((int(corners_2[match[1]][1]), int(corners_2[match[1]][0])))


            M, mask = cv2.findHomography(np.array(src_points), np.array(dst_points), cv2.RANSAC, 5.0)
            homographies.append(M)
        

        width = imgs[0].shape[1] * 3
        height = imgs[0].shape[0] * 3
        result = imgs[-1]
        for i in range(2,0,-1):
            result = cv2.warpPerspective(result, homographies[i-1], (width, height))
            result[0:imgs[i-1].shape[0], 0:imgs[i-1].shape[1]] = imgs[i-1]
        
        return result

if __name__ == "__main__":
    try:
        save = False 
        show = True
        fd = FeatureDetector(show, save)

        ## Convert to grayscale 
        img_path = "images/checkerboard.png"
        img = cv2.imread(img_path)
        gray = fd.convert_to_grayscale(img_path)

        ## Estimate corners and draw circles
        corners = fd.compute_corners(gray, 15000, filter='gauss', kernel_size = 3, distance=10)
        for i in corners:
            cv2.circle(img, (int(i[1]), int(i[0])),2,(0,0,255),-1)
        
        ## Save image
        if save:
            img = Image.fromarray(np.uint8(img))
            img.save("../_out/img/checkerboard_circles.jpeg")
        if show:
            plt.imshow(img)
            plt.show()

        # load images 
        img_path_1 = "images/mountain_1.jpg"
        img_1 = cv2.imread(img_path_1)[:,:,::-1]
        gray_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY)


        ## Find matches in pictures
        img_path_2 = "images/mountain_2.jpg"
        img_2 = cv2.imread(img_path_2)[:,:,::-1]
        gray_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY)

        corners_1 = fd.compute_corners(gray_1, 10000, filter='gauss', kernel_size = 3, distance=5)
        corners_2 = fd.compute_corners(gray_2, 10000, filter='gauss', kernel_size = 3, distance=5)

        descriptors_1 = fd.compute_descriptors(gray_1, corners_1)
        descriptors_2 = fd.compute_descriptors(gray_2, corners_2)


        matches = fd.compute_matches(descriptors_1, descriptors_2, 2)
        fd.plot_matches(img_1, img_2,corners_1, corners_2, matches)

        ## Stitch images together 
        img_paths = ["images/ducati1.jpg", "images/ducati2.jpg", "images/ducati3.jpg"]
        result = fd.stitch_images(img_paths, 15000, 100)
        
        if save:
            img = Image.fromarray(np.uint8(result.copy()))
            img.save("../img/stitched_mountains.jpeg")
        if show:
            plt.imshow(result)
            plt.show()

        ## make use of SuperPoint
        from SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend

        weights_path = "SuperPointPretrainedNetwork/superpoint_v1.pth"
        spf = SuperPointFrontend(weights_path, nms_dist = 4, conf_thresh =0.015, nn_thresh=0.7, cuda =False)
        img_1 = cv2.imread("images/carla_0000.png")[:,:,::-1]
        gray_1 = cv2.cvtColor(img_1,cv2.COLOR_RGB2GRAY) / 255

        img_2 = cv2.imread("images/carla_0001.png")[:,:,::-1]
        gray_2 = cv2.cvtColor(img_2,cv2.COLOR_RGB2GRAY) / 255

        img_3 = cv2.imread("images/carla_0002.png")[:,:,::-1]
        gray_3 = cv2.cvtColor(img_3,cv2.COLOR_RGB2GRAY) / 255

        img_4 = cv2.imread("images/carla_0003.png")[:,:,::-1]
        gray_4 = cv2.cvtColor(img_4,cv2.COLOR_RGB2GRAY) / 255

        pts_1, desc_1, heatmap_1 = spf.run(gray_1.astype(np.float32))
        corners_1 = []
        for i in range(pts_1.shape[1]):
            corners_1.append((pts_1[1][i], pts_1[0][i]))

        pts_2, desc_2, heatmap_2 = spf.run(gray_2.astype(np.float32))
        corners_2 = []
        for i in range(pts_2.shape[1]):
            corners_2.append((pts_2[1][i], pts_2[0][i]))

        pts_3, desc_3, heatmap_3 = spf.run(gray_3.astype(np.float32))
        corners_3 = []
        for i in range(pts_3.shape[1]):
            corners_3.append((pts_3[1][i], pts_3[0][i]))

        pts_4, desc_4, heatmap_4 = spf.run(gray_4.astype(np.float32))
        corners_4 = []
        for i in range(pts_4.shape[1]):
            corners_4.append((pts_4[1][i], pts_4[0][i]))


        matches_12 = fd.compute_matches(np.swapaxes(desc_1, 0, 1), np.swapaxes(desc_2, 0, 1), 2)
        fd.plot_matches(img_1, img_2,corners_1, corners_2, matches_12)
        matches_34 = fd.compute_matches(np.swapaxes(desc_3, 0, 1), np.swapaxes(desc_4, 0, 1), 2)
        fd.plot_matches(img_3, img_4,corners_3, corners_4, matches_34)
    except: 
        print(traceback.format_exc())