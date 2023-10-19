from glob import glob
import os
import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from natsort import natsorted

import Optimization.Optim_matrix as Optim_matrix
import Render.Render_open3d as Render_open3d
from Render.DetectContour import DetectContour_fromMask

from DetectCoord.TeethDetect2D import TeethCoords2D, CalcuCoords
from DetectCoord.TeethDetect3D import TeethDetect3D
from DetectCoord.TeethRowMesh import save_to_obj
from DetectMouth.DetectMouth import DetectMouth, CropMouth
from DetectFace.DetectFace import DetectFace

from GenerateTooth.Network import Network
from Utils.utils import Restore, tensor2img



### setup
name = 'Company_1'

face_path = os.path.join('../Data', name, 'orig.png')
bf_mesh_path = os.path.join('../Data', name, 'teeth')
af_mesh_path = os.path.join('../Data', name, 'teeth_after')
save_path = os.path.join('../Data', name)

### detect the human face and resize to 512*512
ori_img, face, info_detectface = DetectFace(face_path, newsize=(512, 512))
face_PIL = Image.fromarray(cv2.cvtColor(np.asarray(face), cv2.COLOR_BGR2RGB))

### read 2D coords from txt and calculate related 2D coords because of function_'DetectFace'
pixel_coord, teeth_id = TeethCoords2D(name, '../Data/2D_coords.txt')
print("teeth_id", teeth_id)
pixel_coord = CalcuCoords(pixel_coord, face_PIL.size, info_detectface)
print("pixel_coord\n", pixel_coord)


### read 3D coords from Tooth Model, for registration
world_coord, teethrow_mesh = TeethDetect3D(bf_mesh_path, af_mesh_path, part_teeth='all', rotation=(-np.pi/2, np.pi/2, 0.), teeth_id=teeth_id)
print("world_coord\n", world_coord)

### registration
optim_result = Optim_matrix.train(pixel_coord, world_coord, rt_vec=[1e-10, 1e-10, 1e-10], size=face_PIL.size, epoch=3000)
center = [optim_result['c1'].item(), optim_result['c2'].item()]
t3 = optim_result['t3'].item()
rt_vec = np.array([optim_result['r1'].item(), optim_result['r2'].item(), optim_result['r3'].item()])

### render to obation the projection
teeth_proj, teeth_proj_trans = Render_open3d.render(teethrow_mesh, rt_vec, center, t3, face_PIL.size)     # Color: RGB

# ---------------visual the registration and render---------------
img_blend = Image.blend(face_PIL, Image.fromarray(np.uint8(teeth_proj)), 0.4)
img_blend.save(os.path.join(save_path, 'regist.png'))
img_blend_trans = Image.blend(face_PIL, Image.fromarray(np.uint8(teeth_proj_trans)), 0.4)
img_blend_trans.save(os.path.join(save_path, 'regist_trans.png'))
# ---------------visual the registration and render---------------

### detect the mouth region 
_, mouth_region, _ = DetectMouth(face)                                 # Mouth_region: 0 and 255

### mask with mouth_region
teeth_real = (mouth_region/255) * np.uint8(face)                       # image of real teeth  (BGR)
teeth_proj = (mouth_region/255) * np.uint8(teeth_proj)                 # image of teeth projection before treatment  (Color: RGB)
teeth_proj_trans = (mouth_region/255) * np.uint8(teeth_proj_trans)                 # image of teeth projection after treatment   (Color: RGB)

### crop mouth (256*128) from 512*512 images
crop_result, info_cropmouth = CropMouth(mouth_region, (256, 128), teeth_real, teeth_proj, teeth_proj_trans, face)
crop_mouth_region, crop_teeth_real, crop_teeth_proj, crop_teeth_proj_trans, crop_face = crop_result

## detect teeth contour from teeth projection(also called teeth mask)
crop_teeth_contour = np.uint8((crop_mouth_region/255) * DetectContour_fromMask(crop_teeth_proj))             # RGB to binary
crop_teeth_contour_trans = np.uint8((crop_mouth_region/255) * DetectContour_fromMask(crop_teeth_proj_trans)) # RGB to binary


### set the data
data = {'ori_face': ori_img,
        'detect_face': face,
        'info': {0: info_detectface, 1: info_cropmouth},
        'crop_teeth': crop_teeth_contour, 
        'crop_teeth_align': crop_teeth_contour_trans, 
        'crop_mouth': crop_teeth_real, 
        'crop_mask': crop_mouth_region, 
        'crop_face': crop_face}

### Generate the real tooth photo
from GenerateTooth.Generator import Contour2ToothGenerator_FaceColor_LightColor as Generator
with open("./GenerateTooth/config/config_Contour2Tooth_facecolor_lightcolor.yaml", 'r') as f:
    GeneratorConfig = yaml.load(f, Loader=yaml.SafeLoader)['GeneratorConfig']
# initialize the Network
netG = Network(GeneratorConfig['unet'], GeneratorConfig['beta_schedule'])
netG.load_state_dict(torch.load('./GenerateTooth/ckpt/ckpt_contour2tooth_v2_ContourSegm_facecolor_lightcolor_10000.pth'), strict=False)
netG.to(torch.device('cuda'))
netG.eval()
# initialize the Generator
generator = Generator(netG)
prediction, cond_teeth_color = generator.predict(data)       # tensor_BGR_float32 (-1to1)
mouth_align = tensor2img(prediction)                          # numpy_BGR_uint8 (0-255)





### restore the facial photo with original size (Result here!!!)
pred = Restore(mouth_align, data)
pred_face = pred['pred_ori_face']
cv2.imwrite(os.path.join(save_path, 'pred_face.png'), pred_face)

# ---------------visual the normalized and rotated 3D dental model---------------
save_to_obj(os.path.join(save_path, 'teethrow.obj'), teethrow_mesh.teethrow_mesh.v, teethrow_mesh.teethrow_mesh.f)
save_to_obj(os.path.join(save_path, 'teethrow_trans.obj'), teethrow_mesh.teethrow_transmesh.v, teethrow_mesh.teethrow_transmesh.f)
# ---------------visual the normalized and rotated 3D dental model---------------