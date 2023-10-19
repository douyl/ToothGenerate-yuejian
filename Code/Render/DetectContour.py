import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
import matplotlib
import os
from skimage import morphology

def Preprocess(img):
    # Input: cv2 image (BGR)
    img_copy = deepcopy(img)
    location = np.where((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0))
    img_copy[location[0], location[1]] = (255, 255, 255)
    # cv2.imwrite(r'C:\Users\douyl\Desktop\o3d\test.jpg',img_copy)
    return img_copy

def Postprocess(img, mouth_mask):
    img = mouth_mask * np.uint8(img)
    

def DetectContour(img):
    # Input: PIL image (RGB)
    img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 预处理，背景变成白色
    img_cv2 = Preprocess(img_cv2)

    # img_cv2 = cv2.GaussianBlur(img_cv2, (5,5), 0)

    # Canny Edge Detection
    threshold1 = 200
    threshold2 = 100
    img_contour = cv2.Canny(img_cv2, threshold1, threshold2)
    # 闭操作，闭合曲线
    # img_contour = cv2.morphologyEx(img_contour, cv2.MORPH_CLOSE, kernel=(2,2), iterations=1)
    # 膨胀
    kernel = np.ones((2,2), np.uint8)
    img_contour = cv2.dilate(img_contour, kernel, iterations=1)
    img_contour = np.flip(cv2.dilate(np.flip(img_contour), kernel, iterations=1))


    img_contour_PIL = Image.fromarray(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
    return img_contour_PIL

def DetectContour_fromMask(img, if_visual=False):
    img_rgb = np.asarray(img)
    id_color_dict = {
    '1': 'Orchid',
    '2': 'Firebrick',
    '3': 'SpringGreen',
    '4': 'BlueViolet',
    '5': 'RoyalBlue',
    '6': 'MidnightBlue',
    '7': 'Deepskyblue',
    '8': 'Purple',
    
    '9': 'Blue',
    '10': 'Magenta',
    '11': 'MediumBlue',
    '12': 'DarkViolet',
    '13': 'DarkTurquoise',
    '14': 'DeepPink',
    '15': 'SaddleBrown',
    '16': 'Cyan',

    '17': 'Aliceblue',
    '18': 'SlateGrey',
    '19': 'Maroon',
    '20': 'ForestGreen',
    '21': 'Sienna',
    '22': 'DarkSlateGray',
    '23': 'Orangered',
    '24': 'Red',

    '25': 'Green',
    '26': 'Brown',
    '27': 'DarkCyan',
    '28': 'Lime',
    '29': 'Darkblue',
    '30': 'Chocolate',
    '31': 'DarkGreen',
    '32': 'Tan'
    }

    img_contour = np.zeros(img_rgb.shape, dtype=np.uint8)
    for id in id_color_dict.keys():
        tooth = np.zeros(img_rgb.shape, dtype=np.uint8)
        color = matplotlib.colors.to_rgb(id_color_dict[id])
        color_RGB = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        where = np.where((img_rgb[:,:,0]==color_RGB[0]) & (img_rgb[:,:,1]==color_RGB[1]) & (img_rgb[:,:,2]==color_RGB[2]))
        if len(where[0]) == 0: 
            continue
        tooth[where[0],where[1],:] = (255,255,255)
        
        kernel = np.ones((5,5),np.uint8)
        # kernel = np.ones((3,3),np.uint8) 
        tooth_erode = cv2.erode(tooth, kernel, iterations=1)
        tooth_contour = tooth - tooth_erode
        img_contour += tooth_contour
    # img_contour = cv2.morphologyEx(img_contour, cv2.MORPH_CLOSE, kernel=(2,2), iterations=1)
    # img_contour = morphology.skeletonize(img_contour[:,:,0]/255) * 255     # 提取骨架
    # img_contour = np.expand_dims(img_contour, -1).repeat(3, axis=-1)

    if if_visual == True:
        cv2.imwrite(os.path.join('./result_vis', 'contour_1.png'), img_contour)
    # print(np.unique(img_contour))
    return img_contour

