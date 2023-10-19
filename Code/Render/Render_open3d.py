import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import cv2
import matplotlib
from PIL import Image
from DetectCoord.TeethRowMesh import TeethRowMesh

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


def render(mesh, rt_vec, center, t3, size):
    # camera参数
    w, h = size[0], size[1]
    intrinsic = np.array([[w*4/4.8, 0, center[0]],
                          [0, w*4/4.8, center[1]],
                          [0, 0, 1]])

    extrinsic = np.eye(4, 4)
    Rt = cv2.Rodrigues(rt_vec)[0]
    extrinsic[:3, :3] = Rt
    extrinsic[2, 3] = t3

    material = rendering.MaterialRecord()

    meshes = mesh.mesh_list
    transmeshes = mesh.transmesh_list
    ids = mesh.id_list

    # Render the before-transformation teeth model
    triangle_mesh = o3d.geometry.TriangleMesh()
    for i, mesh in enumerate(meshes):
        trimesh = o3d.geometry.TriangleMesh()
        trimesh.vertices = o3d.utility.Vector3dVector(mesh.v)
        trimesh.triangles = o3d.utility.Vector3iVector(mesh.f)
        # color = list(np.random.random(size=3))
        # color = matplotlib.colors.to_rgb('black')
        color = matplotlib.colors.to_rgb(id_color_dict[ids[i]])
        trimesh.paint_uniform_color(color)
        triangle_mesh += trimesh
   
    OffscreenRenderer = o3d.visualization.rendering.OffscreenRenderer(width=w, height=h)
    OffscreenRenderer.scene.add_geometry("", triangle_mesh, material, add_downsampled_copy_for_fast_rendering=False)
    OffscreenRenderer.scene.set_background([255,255,255,1])
    OffscreenRenderer.scene.view.set_antialiasing(False)
    OffscreenRenderer.scene.view.set_post_processing(False)
    OffscreenRenderer.setup_camera(intrinsic, extrinsic, w, h)
    mask = OffscreenRenderer.render_to_image()


    #Render the after-transformation teeth model
    triangle_transmesh = o3d.geometry.TriangleMesh()
    for i, transmesh in enumerate(transmeshes):
        trimesh_trans = o3d.geometry.TriangleMesh()
        trimesh_trans.vertices = o3d.utility.Vector3dVector(transmesh.v)
        trimesh_trans.triangles = o3d.utility.Vector3iVector(transmesh.f)
        color = matplotlib.colors.to_rgb(id_color_dict[ids[i]])
        trimesh_trans.paint_uniform_color(color)
        triangle_transmesh += trimesh_trans
   
    OffscreenRenderer_trans = o3d.visualization.rendering.OffscreenRenderer(width=w, height=h)
    OffscreenRenderer_trans.scene.add_geometry("trans", triangle_transmesh, material, add_downsampled_copy_for_fast_rendering=False)
    OffscreenRenderer_trans.scene.set_background([255,255,255,1])
    OffscreenRenderer_trans.scene.view.set_antialiasing(False)
    OffscreenRenderer_trans.scene.view.set_post_processing(False)
    OffscreenRenderer_trans.setup_camera(intrinsic, extrinsic, w, h)
    mask_trans = OffscreenRenderer_trans.render_to_image()

    return np.asarray(mask), np.asarray(mask_trans)
    # return np.asarray(mask), None
    o3d.io.write_image(r'C:\Users\douyl\Desktop\o3d\5.jpg', mask, quality=100)



if __name__ == '__main__':
    teethrow_mesh = TeethRowMesh(r'C:\IDEA_Lab\Project_tooth_video\2DRegist\FacePose\Data\DentalModel')
    # rt_vec = np.array([0.120009, -0.01133796, -0.04190715])
    rt_vec = np.array([0., 0., 0.])
    center = (512, 512)
    t3 = 3
    size = (1024, 1024)
    
    mask, mask_trans = render(teethrow_mesh, rt_vec, center, t3, size)
    mask = np.asarray(mask)
    mask = Image.fromarray(mask)
    mask.save(r'C:\IDEA_Lab\Project_tooth_video\2DRegist\FacePose\Data\visual\x_neg_pi_frac2.jpg')