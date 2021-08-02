import numpy as np
import pyntcloud
import matplotlib.pyplot as plt
import torch
import pickle

import torch.nn.functional as F

from pointnet import PointNet

with open('src/classes.pickle', 'rb') as f:
    class_dict = pickle.load(f)

PATH = 'src/pointnet.pth'
device = torch.device('cpu')
model = PointNet()
model.load_state_dict(torch.load(PATH, map_location=device))


def get_file(path):
    pc = pyntcloud.PyntCloud.from_file(path)
    pc = pc.get_sample("mesh_random", n=10000, rgb=False, normals=True, as_PyntCloud=True)
    xyz = pc.points[['x', 'y', 'z']].values
    return torch.swapdims(torch.tensor(np.expand_dims(xyz, 0), dtype=torch.float32), -1, -2)


def print_result(result, class_dict):
    fig1, ax1 = plt.subplots()
    p1 = ax1.bar([x + 1 for x in class_dict.values()], np.squeeze(result * 100))
    ax1.set_xticks([x + 1 for x in np.arange(len(class_dict.keys()))])
    ax1.set_xticklabels(class_dict.keys())
    plt.show()


if __name__ == '__main__':
    print('Введите путь к файлу .obj: ')
    path = input()
    xyz = get_file(path)
    model.eval()

    with torch.no_grad():
        result, _, _, _ = model(xyz)
        result = F.softmax(result).numpy()

    print_result(result, class_dict)