import numpy as np
import torch

def TeethCoords2D(img_name, label_txt_path):
    with open(label_txt_path, 'r') as f:
        labels_list = f.readlines()
    img_names = [label.split('------')[0] for label in labels_list]
    teeth_infos = [eval(label.split('------')[1].strip('\n')) for label in labels_list]
    names_infos_dict = dict(zip(img_names, teeth_infos))   #{name: info_dict}

    teethtips2D = None
    if img_name not in names_infos_dict.keys():
        assert ValueError
    else:
        teeth_info = names_infos_dict[img_name]
        teethtips2D = []
        for id in teeth_info.keys():
            toothtip2D = np.asarray(list(teeth_info[id][0].values()))
            toothtip2D = np.concatenate([toothtip2D, [1]])
            teethtips2D.append(toothtip2D)
        teethtips2D = np.asarray(teethtips2D, dtype=np.float32).T

    return  torch.from_numpy(teethtips2D), list(teeth_info.keys())


def CalcuCoords(coords, size, info_detectface):
    new_coords = np.ones_like(coords)
    x1, x2 = info_detectface['coord_x']
    y1, y2 = info_detectface['coord_y']
    size_x, size_y = info_detectface['face_size']
    for i in range(coords.shape[1]):
        coord_x = coords[0, i]
        coord_y = coords[1, i]
        new_coord_x = int((coord_x - x1) * size[0] / size_x)
        new_coord_y = int((coord_y - y1) * size[1] / size_y)
        new_coords[0, i] = new_coord_x
        new_coords[1, i] = new_coord_y
    return torch.from_numpy(new_coords).float()


