import cv2
import os
import dlib
import numpy as np
from skimage.filters import gaussian
from .test import evaluate
import argparse
from copy import deepcopy
from PIL import Image



def mask(image, parsing, part=11, color=[139, 0, 139]):
	b, g, r = color      
	tar_color = np.zeros_like(image)
	tar_color[:, :, 0] = b
	tar_color[:, :, 1] = g
	tar_color[:, :, 2] = r

	changed = tar_color.copy()
	changed[parsing != part] = image[parsing != part]

	return changed


def DetectMouth(image):
	cp = './DetectMouth/cp/79999_iter.pth'
	# image = cv2.imread(image_path)

	origin_img = deepcopy(image)
	mouth_color = deepcopy(image)
	mouth_mask = np.zeros_like(image)

	parsing = evaluate(image, cp)
	parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

	part = 11  # mouth
	color = [255, 255, 255]
	mouth_color = mask(mouth_color, parsing, part, color)
	mouth_mask = mask(mouth_mask, parsing, part, color)

	return origin_img, mouth_mask, mouth_color



def CropMouth_1(teeth_real, teeth_proj, teeth_proj_trans, face, mouth_region, crop_size=(256,128)):

	location = np.where(mouth_region == 1.)
	center_u = int((np.min(location[1]) + np.max(location[1])) / 2)   # u-axis is along to width
	center_v = int((np.min(location[0]) + np.max(location[0])) / 2)   # v-axis is along to height

	mouth_region_crop = Image.fromarray(np.uint8(mouth_region*255))

	result = []
	for teeth in [teeth_real, teeth_proj, teeth_proj_trans, mouth_region_crop, face]:


		cropteeth = teeth.crop((center_u-crop_size[0]//2, center_v-crop_size[1]//2, center_u+crop_size[0]//2, center_v+crop_size[1]//2))
		result.append(cropteeth)

	info = {
        'coord_x': (center_u-crop_size[0]//2, center_u+crop_size[0]//2),
        'coord_y': (center_v-crop_size[1]//2, center_v+crop_size[1]//2),
        'new_size': crop_size,
    }
	return result[0], result[1], result[2], result[3], result[4], info


def CropMouth(mouth_region, crop_size=(256,128), *imgs):

	location = np.where(mouth_region == 255.)
	center_u = int((np.min(location[1]) + np.max(location[1])) / 2)   # u-axis is along to width
	center_v = int((np.min(location[0]) + np.max(location[0])) / 2)   # v-axis is along to height

	result = []
	# first crop the mouth region
	if type(mouth_region) is np.ndarray:
		img = Image.fromarray(np.uint8(mouth_region))      # Needn't change BGR to RGB
	crop_img = img.crop((center_u-crop_size[0]//2, center_v-crop_size[1]//2, center_u+crop_size[0]//2, center_v+crop_size[1]//2))
	crop_img = np.asarray(crop_img)                        # Also needn't change channels, still BGR
	result.append(crop_img)

	# then crop other images
	for img in imgs:
		if type(img) is np.ndarray:
			img = Image.fromarray(np.uint8(img))      # Needn't change BGR to RGB
		crop_img = img.crop((center_u-crop_size[0]//2, center_v-crop_size[1]//2, center_u+crop_size[0]//2, center_v+crop_size[1]//2))
		crop_img = np.asarray(crop_img)               # Also needn't change channels, still BGR
		result.append(crop_img)

	info = {
        'coord_x': (center_u-crop_size[0]//2, center_u+crop_size[0]//2),
        'coord_y': (center_v-crop_size[1]//2, center_v+crop_size[1]//2),
        'new_size': crop_size,
    }
	return result, info

if __name__ == '__main__':
	img_path = './DetectMouth/images/img10.jpg'
	img = cv2.imread(img_path)
	DetectMouth(img)

