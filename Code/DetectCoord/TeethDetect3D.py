from DetectCoord.TeethRowMesh import TeethRowMesh
import numpy as np
import torch

def TeethDetect3D(bf_folder, af_folder, part_teeth, rotation, teeth_id):
    '''
    To: Generate the 3D coordinates of dental model, each tooth (column) is (x,y,z,1)
    Output example:
        tensor([[-0.5549, -0.3585, -0.1022,  0.1936,  0.4144,  0.5852],
                [ 0.0426,  0.0174,  0.0515,  0.0731,  0.0505,  0.0790],
                [-0.2387, -0.4327, -0.5550, -0.5457, -0.4219, -0.2151],
                [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]],dtype=torch.float64)
    '''
    teethrow_mesh = TeethRowMesh(bf_folder=bf_folder, af_folder=af_folder, part_teeth=part_teeth, rotation=rotation)
    teethId_startIdx_dict = teethrow_mesh.teethId_startIdx_dict

    teethtips3D = []

    for id in teeth_id:
        tooth_id = id
        if not tooth_id in teethId_startIdx_dict.keys():
            raise Exception('No such tooth in 3D dental model.')
        start_idx = teethId_startIdx_dict[tooth_id]
        if not str(int(tooth_id)+1) in teethId_startIdx_dict.keys():
            index = list(teethId_startIdx_dict.keys()).index(tooth_id)
            stop_idx = teethId_startIdx_dict[list(teethId_startIdx_dict.keys())[index+1]]
        else:
            stop_idx = teethId_startIdx_dict[str(int(tooth_id)+1)]

        verts = teethrow_mesh.teethrow_mesh.v[start_idx: stop_idx, :]

        ### sorted by y-axis and select the 5% biggest verts
        sort_idx = np.lexsort((verts[:, 0], verts[:, 2], verts[:, 1]))
        verts = verts[sort_idx]
        verts_big = verts[int(0.95*len(verts)):]  #set confidence

        toothtip3D = np.mean(verts_big, axis=0)
        toothtip3D = np.concatenate([toothtip3D, [1]])
        teethtips3D.append(toothtip3D)
        
        
    teethtips3D = np.asarray(teethtips3D).T
    return torch.from_numpy(teethtips3D), teethrow_mesh
        




if __name__ == '__main__':
    TeethDetect3D(r'C:\IDEA_Lab\Project_tooth_alignment\myNet\Data\TeethData_500\00ab7a70199634666af1c142797674a8')