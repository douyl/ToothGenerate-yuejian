from math import cos, sin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from tqdm import tqdm

# class model(nn.Module):
#     def __init__(self, init_var, rt_vec, size):
#         super().__init__()
#         self.var = init_var
#         for key,value in self.var.items():
#             setattr(self, key, nn.Parameter(torch.tensor(value).float()))

#         rotation = cv2.Rodrigues(rt_vec)[0]
#         self.rotation = torch.Tensor(np.hstack([rotation, np.array([[0.,0.,0.]]).T]))
#         print(self.rotation)

#         translation = np.zeros((3,4))
#         translation[2,3] = 1.
#         self.translation = torch.Tensor(translation)
        
#         focal = np.eye(3)
#         focal[0,0] = size[0]*4/4.8
#         focal[1,1] = size[0]*4/4.8
#         self.focal = torch.Tensor(focal)
        
#         center_x = np.zeros((3,3))
#         center_x[0,2] = 1.
#         center_y = np.zeros((3,3))
#         center_y[1,2] = 1.
#         self.center_x = torch.Tensor(center_x)
#         self.center_y = torch.Tensor(center_y)

#     def forward(self, pixel_coord, world_coord):
#         pixel_coord = pixel_coord.float()
#         world_coord = world_coord.float()
#         coord_0 = self.rotation@world_coord + self.t3*self.translation@world_coord
#         coord_1 = coord_0 / coord_0[2,:]
#         coord_2 = self.focal@coord_1 + self.c1*self.center_x@coord_1 + self.c2*self.center_y@coord_1
#         loss = F.l1_loss(coord_2[:2,:], pixel_coord[:2,:])

#         return loss

# def train(pixel_coord, world_coord, rt_vec, size, epoch):
#     w, h = size[0], size[1]
#     # set the initial variable
#     init_var = {
#         'c1':w/2,
#         'c2':h/2,
#         't3':7,
#     }

#     best_loss = torch.tensor(10**10)
#     best_var = {
#         'c1':None,
#         'c2':None,
#         't3':None,
#     }

#     net = model(init_var, rt_vec, size)
#     net.train()

#     centers = list(map(id, net.parameters()))[:2]
#     center_params = filter(lambda p: id(p) in centers, net.parameters())
#     trans_params = filter(lambda p: id(p) not in centers, net.parameters())
#     optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},
#                                 {'params':trans_params, 'lr':0.01}])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)


#     for e in tqdm(range(epoch)):
#         loss = net(pixel_coord, world_coord)
#         # print(loss)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if loss.item() < best_loss.item():
#             best_loss = loss
#             best_var['c1'] = net.c1
#             best_var['c2'] = net.c2
#             best_var['t3'] = net.t3
        
#         scheduler.step()
#     # print(optimizer.state_dict()['param_groups'][0]['lr'])
#     # print(optimizer.state_dict()['param_groups'][1]['lr'])

#     print('LOSS: {}'.format(best_loss.item()))
#     print("PARAM: c1: {}, c2: {}, t3: {}".format(best_var['c1'].item(), best_var['c2'].item(), best_var['t3'].item()))
#     return best_var

# ===================================================



def differentiable_rotation_vector_to_matrix(rotation_vector):
    # Convert a differentiable rotation vector to a rotation matrix
    # Input:
    #   - rotation_vector: A 3D rotation vector (PyTorch tensor)
    # Output:
    #   - rotation_matrix: A 3x3 rotation matrix (PyTorch tensor)
    

    # Ensure the rotation_vector is a PyTorch tensor
    # rotation_vector = torch.tensor(rotation_vector, dtype=torch.float, requires_grad=True)
    
    # Calculate the rotation matrix
    angle = torch.norm(rotation_vector)
    axis = rotation_vector / (angle + 1e-6)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    rotation_matrix = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device) * cos_angle

    # Method1
    # skew_symmetric = torch.tensor([
    #     [0, -axis[2], axis[1]],
    #     [axis[2], 0, -axis[0]],
    #     [-axis[1], axis[0], 0]
    # ], dtype=rotation_vector.dtype, device=rotation_vector.device)

    # Method2
    # Manually compute the gradient for skew_symmetric
    # skew_symmetric = torch.zeros(3, 3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    # skew_symmetric[0, 1] = -axis[2]
    # skew_symmetric[0, 2] = axis[1]
    # skew_symmetric[1, 0] = axis[2]
    # skew_symmetric[1, 2] = -axis[0]
    # skew_symmetric[2, 0] = -axis[1]
    # skew_symmetric[2, 1] = axis[0]

    # Method3 
    skew_symmetric = construct_K(axis)

    rotation_matrix = rotation_matrix + sin_angle * skew_symmetric
    rotation_matrix = rotation_matrix + (1.0 - cos_angle) * torch.ger(axis, axis)
    
    return rotation_matrix


def construct_K(axis):
    Z = torch.tensor([[0, -1, 0],
                     [1, 0, 0],
                     [0, 0, 0]])
    Y = torch.tensor([[0, 0, 1],
                     [0, 0, 0],
                     [-1, 0, 0]])
    X = torch.tensor([[0, 0, 0],
                     [0, 0, -1],
                     [0, 1, 0]])
    K = axis[0] * X + axis[1] * Y + axis[2] * Z
    return K
    

class model(nn.Module):
    def __init__(self, init_var, rt_vec, size):
        super().__init__()
        self.var = init_var
        for key,value in self.var.items():
            setattr(self, key, nn.Parameter(torch.tensor(value).float()))

        # rotation = cv2.Rodrigues(rt_vec)[0]
        # self.rotation = torch.Tensor(np.hstack([rotation, np.array([[0.,0.,0.]]).T]))
        # self.rt_vec = rt_vec

        translation = np.zeros((3,4))
        translation[2,3] = 1.
        self.translation = torch.Tensor(translation)
        
        focal = np.eye(3)
        focal[0,0] = size[0]*4/4.8
        focal[1,1] = size[0]*4/4.8
        self.focal = torch.Tensor(focal)
        
        center_x = np.zeros((3,3))
        center_x[0,2] = 1.
        center_y = np.zeros((3,3))
        center_y[1,2] = 1.
        self.center_x = torch.Tensor(center_x)
        self.center_y = torch.Tensor(center_y)

    def forward(self, pixel_coord, world_coord):
        # print('c1', self.c1.grad)
        # print('r1', self.r1.grad)
        

        # rotation_vector = torch.tensor([self.r1, self.r2, self.r3], requires_grad=True)
        rotation_vector = torch.cat([self.r1.unsqueeze(0), self.r2.unsqueeze(0), self.r3.unsqueeze(0)], 0)
        
        # rotation_vector = torch.zeros(3, dtype=torch.float, requires_grad=True)
        # rotation_vector[0].data.add_(self.r1)
        # rotation_vector[1].data.add_(self.r2)
        # rotation_vector[2].data.add_(self.r3)


        rotation = differentiable_rotation_vector_to_matrix(rotation_vector)
        self.rotation = torch.cat([rotation, torch.tensor([[0.],[0.],[0.]])], -1)
        # print(self.rotation)

        pixel_coord = pixel_coord.float()
        world_coord = world_coord.float()
        coord_0 = self.rotation@world_coord + self.t3*self.translation@world_coord
        coord_1 = coord_0 / coord_0[2,:]
        coord_2 = self.focal@coord_1 + self.c1*self.center_x@coord_1 + self.c2*self.center_y@coord_1
        loss = F.l1_loss(coord_2[:2,:], pixel_coord[:2,:])
        # loss = F.mse_loss(coord_2[:2,:], pixel_coord[:2,:])

        return loss

def train(pixel_coord, world_coord, rt_vec, size, epoch):
    w, h = size[0], size[1]
    # set the initial variable
    init_var = {
        'c1':w/2,
        'c2':h/2,
        't3':5,            # 7
        'r1':rt_vec[0],
        'r2':rt_vec[1],
        'r3':rt_vec[2],
    }

    best_loss = torch.tensor(10**10)
    best_var = {
        'c1':None,
        'c2':None,
        't3':None,
        'r1':None,
        'r2':None,
        'r3':None,
    }

    net = model(init_var, rt_vec, size)
    net.train()

    # -----------
    # centers = list(map(id, net.parameters()))[:2]
    # translations = [list(map(id, net.parameters()))[2]]
    # rotations = list(map(id, net.parameters()))[-3:]
    # center_params = filter(lambda p: id(p) in centers, net.parameters())
    # trans_params = filter(lambda p: id(p) in translations, net.parameters())
    # rotat_params = filter(lambda p: id(p) in rotations, net.parameters())
    # # optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},
    # #                             {'params':trans_params, 'lr':0.01}])
    # optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},
    #                             {'params':trans_params, 'lr':0.01},
    #                             {'params':rotat_params, 'lr':1e-6}])    # 1e-6
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    # -----------

    centers = list(map(id, net.parameters()))[:2]
    translations = [list(map(id, net.parameters()))[2]]
    rotation_x = [list(map(id, net.parameters()))[-3]]
    rotation_yz = list(map(id, net.parameters()))[-2:]
    center_params = filter(lambda p: id(p) in centers, net.parameters())
    trans_params = filter(lambda p: id(p) in translations, net.parameters())
    rotat_x_params = filter(lambda p: id(p) in rotation_x, net.parameters())
    rotat_yz_params = filter(lambda p: id(p) in rotation_yz, net.parameters())
    # optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},
    #                             {'params':trans_params, 'lr':0.01}])
    # ------------------------------------------
    # optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},      # 0.2
    #                             {'params':trans_params, 'lr':0.01},
    #                             {'params':rotat_x_params, 'lr':1e-6},      # 1e-5
    #                             {'params':rotat_yz_params, 'lr':1e-6}])    # 1e-6
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)   # 0.9
    # ------------------------------------------
    optimizer = torch.optim.SGD([{'params':center_params, 'lr':0.2},      # 0.2
                                {'params':trans_params, 'lr':0.01},
                                {'params':rotat_x_params, 'lr':1e-5},      # 1e-5
                                {'params':rotat_yz_params, 'lr':1e-6}])    # 1e-6
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)   # 0.9



    for e in tqdm(range(epoch)):
        loss = net(pixel_coord, world_coord)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        # print('c1', net.c1.grad)
        # print('c2', net.c2.grad)
        # print('t3', net.t3.grad)
        # print('r1', net.r1.grad)
        # print('r2', net.r2.grad)
        # print('r3', net.r3.grad)

        optimizer.step()

        if loss.item() < best_loss.item():
            best_loss = loss
            best_var['c1'] = net.c1
            best_var['c2'] = net.c2
            best_var['t3'] = net.t3
            best_var['r1'] = net.r1
            best_var['r2'] = net.r2
            best_var['r3'] = net.r3
        
        # print("c1:{}, c2:{}, t3:{}, r1:{}, r2:{}, r3:{}".format(net.c1.item(), net.c2.item(), net.t3.item(), net.r1.item(), net.r2.item(), net.r3.item()))

        scheduler.step()

        # if e == 2: quit()


    # print(optimizer.state_dict()['param_groups'][0]['lr'])
    # print(optimizer.state_dict()['param_groups'][1]['lr'])

    print('LOSS: {}'.format(best_loss.item()))
    print("PARAM: c1:{}, c2:{}, t3:{}, r1:{}, r2:{}, r3:{}".format(net.c1.item(), net.c2.item(), net.t3.item(), net.r1.item(), net.r2.item(), net.r3.item()))
    return best_var

