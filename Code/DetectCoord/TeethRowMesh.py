import numpy as np
import cv2
from glob import glob
import os
from natsort import natsorted
import vtk
import json
import copy
# import open3d as o3d

class Mesh(object):
    def __init__(self, verts, faces):
        self.v = verts
        self.f = faces
        self.vc = verts*0. + 1


### 前后mesh数据
class TeethRowMesh(object):
    def __init__(self, bf_folder, af_folder, part_teeth="all", rotation=(0., 0., 0.)):
        ### folder of dental model before alignment
        if part_teeth == "up":
            self.filename_list = natsorted(glob(os.path.join(bf_folder, '[2-9].stl'))+\
                                           glob(os.path.join(bf_folder, '1[0-5].stl')))
        elif part_teeth == "low":
            self.filename_list = natsorted(glob(os.path.join(bf_folder, '1[8-9].stl'))+\
                                           glob(os.path.join(bf_folder, '2[0-9].stl'))+\
                                           glob(os.path.join(bf_folder, '3[0-1].stl')))
        elif part_teeth == "all":
            self.filename_list = natsorted(glob(os.path.join(bf_folder, '[2-9].stl'))+\
                                           glob(os.path.join(bf_folder, '1[0-5].stl'))+\
                                           glob(os.path.join(bf_folder, '1[8-9].stl'))+\
                                           glob(os.path.join(bf_folder, '2[0-9].stl'))+\
                                           glob(os.path.join(bf_folder, '3[0-1].stl')))
        
        ### folder of dental model after alignment
        if part_teeth == "up":
            self.filename_trans_list = natsorted(glob(os.path.join(af_folder, '[2-9].ply'))+\
                                           glob(os.path.join(af_folder, '1[0-5].ply')))
        elif part_teeth == "low":
            self.filename_trans_list = natsorted(glob(os.path.join(af_folder, '1[8-9].ply'))+\
                                           glob(os.path.join(af_folder, '2[0-9].ply'))+\
                                           glob(os.path.join(af_folder, '3[0-1].ply')))
        elif part_teeth == "all":
            self.filename_trans_list = natsorted(glob(os.path.join(af_folder, '[2-9].ply'))+\
                                           glob(os.path.join(af_folder, '1[0-5].ply'))+\
                                           glob(os.path.join(af_folder, '1[8-9].ply'))+\
                                           glob(os.path.join(af_folder, '2[0-9].ply'))+\
                                           glob(os.path.join(af_folder, '3[0-1].ply')))

        self.mesh_list = [self.load(file) for file in self.filename_list]              # "self.mesh_list"
        self.transmesh_list = [self.load(file) for file in self.filename_trans_list]   # "self.transmesh_list"
        self.teethrow_mesh = self.get_teethrow_mesh()                       # "self.teethrow_mesh"
        self.teethrow_transmesh = self.get_teethrow_transmesh()             # "self.teethrow_transmesh"

        self.id_list = [os.path.basename(file).split('.')[0] for file in self.filename_list]
        ### record the starting index of each tooth
        start_idx_list = []
        numVerts = 0
        for _, mesh in enumerate(self.mesh_list):
            start_idx_list.append(numVerts)
            numVerts += mesh.v.shape[0]
        ### record the id of each tooth corresponding to starting index
        teeth_id_list = []
        for _, filename in enumerate(self.filename_list):
            teeth_id_list.append(os.path.basename(filename).split('.')[0])
        self.teethId_startIdx_dict = dict(zip(teeth_id_list, start_idx_list))

        ### normalize and rotate
        self.rotation = rotation
        self.view_init_all()


    def get_teethrow_mesh(self):
        numVerts = 0
        verts_list =[]
        faces_list = []
        vc_list = []
        i = 0

        for mesh in self.mesh_list:
            if i == 0:
                mesh.vc = mesh.v*0 + [1, 0.85, 0.85]
                i = 1
            else:
                mesh.vc = mesh.v*0 + 1
            verts_list.append(mesh.v)
            faces_list.append(mesh.f + numVerts)
            vc_list.append(mesh.vc)
            numVerts += mesh.v.shape[0]

        teethrow_mesh = Mesh(np.vstack(verts_list), np.vstack(faces_list))
        teethrow_mesh.vc = np.vstack(vc_list)
        # print('TeethRow with #V={}, #F={}, #vc={}'.format(teethrow_mesh.v.shape, teethrow_mesh.f.shape, teethrow_mesh.vc.shape))
        return teethrow_mesh

    def rotate(self, rv):
        self.teethrow_mesh.v = self.teethrow_mesh.v.dot(cv2.Rodrigues(rv)[0])
        for mesh in self.mesh_list:
            mesh.v = mesh.v.dot(cv2.Rodrigues(rv)[0])


    def get_teethrow_transmesh(self):
        numVerts = 0
        verts_list =[]
        faces_list = []
        vc_list = []
        i = 0

        for mesh in self.transmesh_list:
            if i == 0:
                mesh.vc = mesh.v*0 + [1, 0.85, 0.85]
                i = 1
            else:
                mesh.vc = mesh.v*0 + 1
            verts_list.append(mesh.v)
            faces_list.append(mesh.f + numVerts)
            vc_list.append(mesh.vc)
            numVerts += mesh.v.shape[0]

        teethrow_transmesh = Mesh(np.vstack(verts_list), np.vstack(faces_list))
        teethrow_transmesh.vc = np.vstack(vc_list)
        # print('TeethTrans with #V={}, #F={}, #vc={}'.format(teethrow_transmesh.v.shape, teethrow_transmesh.f.shape, teethrow_transmesh.vc.shape))
        return teethrow_transmesh

    def rotate_ForTransform(self, rv):
        self.teethrow_transmesh.v = self.teethrow_transmesh.v.dot(cv2.Rodrigues(rv)[0])
        for mesh in self.transmesh_list:
            mesh.v = mesh.v.dot(cv2.Rodrigues(rv)[0])


    def view_init_all(self):
        mean_v = np.mean(self.teethrow_mesh.v, axis=0)
        max_v = np.max(self.teethrow_mesh.v)
        self.teethrow_mesh.v -= mean_v
        self.teethrow_mesh.v /= max_v
        for mesh in self.mesh_list:
            mesh.v -= mean_v
            mesh.v /= max_v

        self.teethrow_transmesh.v -= mean_v
        self.teethrow_transmesh.v /= max_v
        for mesh in self.transmesh_list:
            mesh.v -= mean_v
            mesh.v /= max_v
        
        self.rotate(np.array([self.rotation[0], 0, 0]))
        self.rotate(np.array([0, self.rotation[1], 0]))
        self.rotate(np.array([0, 0, self.rotation[2]]))

        self.rotate_ForTransform(np.array([self.rotation[0], 0, 0]))
        self.rotate_ForTransform(np.array([0, self.rotation[1], 0]))
        self.rotate_ForTransform(np.array([0, 0, self.rotation[2]]))


    @staticmethod
    def load(filename):
        verts, faces = ReadPolyData(filename)
        teethmesh = Mesh(verts, faces)
        return teethmesh



def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        print ('Warning: Unsupport file extension.')
        poly_data = None
    # return poly_data

    verts = []
    faces = []
    for i in range(poly_data.GetNumberOfPoints()):
        verts.append(np.array(poly_data.GetPoint(i), dtype=np.float64))
    verts = np.vstack(verts)

    triangles = poly_data.GetPolys().GetData()
    for i in range(poly_data.GetNumberOfCells()):
        faces.append(np.array([int(triangles.GetValue(j)) for j in range(4 * i + 1, 4 * i + 4)]))
    if len(faces)>0:
        faces = np.vstack(faces)
    else:
        faces=None
    return verts, faces

def read_TransMatrix_from_json(file_name):
    with open(file_name, 'r') as f:
        trans_matrix_set = json.loads(f.read())
    return trans_matrix_set

def read_TransMatrix_from_txt(file_name):
    with open(file_name, 'r') as f:
        trans_matrix_set = eval(f.read())
        return trans_matrix_set

def save_to_obj(filename, verts, faces):
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face+1))

