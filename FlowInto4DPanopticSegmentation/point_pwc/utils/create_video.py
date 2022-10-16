import os
import glob
import cv2

"This is a short script to build a video out of multiple png images"

def main():
    img_list = []
    base_path = os.getcwd()
    temp = glob.glob("../_input/images/*.png")
    temp.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    size = None
    for filename in temp:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_list.append(img)
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter("../../_output/video/video.avi", fourcc, 2, size)
    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e: 
        print(e)
