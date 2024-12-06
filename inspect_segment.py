import os
import re
import cv2
import copy
import glob
import shutil
import argparse
import numpy as np

from core.MeshBVH import MeshBVH
from core.math.Triangle import Triangle
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutDivide, cutBounding, re_index

# cut OBJ into multi chunks along z-axis
def preprocess(data, folderName):
    layerMin = np.min(data['vertices'][:, 2])
    layerMax = np.max(data['vertices'][:, 2])

    layerMin = int(layerMin // 100) * 100
    layerMax = int(layerMax // 100) * 100
    gap = layerMax - layerMin

    if (gap <= 100):
        name = os.path.join(folderName, f'z{layerMin}_d{gap}.obj')
        save_obj(name, data)
    else:
        cutZ = int(((layerMin + layerMax) / 2) // 100) * 100
        left, right = cutDivide(data, cutZ, align = False)
        preprocess(left, folderName)
        preprocess(right, folderName)

def cut_node(data, node):
    offset = node._offset
    count = node._count

    data_copy = copy.deepcopy(data)
    data_copy['faces'] = data_copy['faces'][offset : offset + count]
    re_index(data_copy)

    return data_copy

def cut_box(data, boundingData):
    boxMin = boundingData[:3]
    boxMax = boundingData[3:]

    data_copy = copy.deepcopy(data)
    cutBounding(data_copy, boxMin, boxMax, align = False)

    return data_copy

def save_n(data, data2, node, depth):
    d_list = []
    uv_list = []

    save_node(data, data2, node, depth, '', d_list, uv_list)

    d = np.concatenate(d_list, axis=0)
    uv = np.concatenate(uv_list, axis=0)

    return d, uv

def save_node(data, data2, node, depth, name, d_list, uv_list):

    def cal(node, mode, name):
        if (mode == 'left'): name += '0'
        if (mode == 'right'): name += '1'
        
        c_data2 = cut_box(data2, node.boundingData)

        if (depth == 0):
            c_data = cut_node(data, node)
            # save_obj(f'{ name }.obj', c_data)
            # save_obj(f'{ name }_.obj', c_data2)
            c_d = d_cal(c_data, c_data2)
            c_uv = c_data['uvs']

            if not np.isinf(np.max(c_d)):
                d_list.append(c_d)
                uv_list.append(c_uv)
        else:
            save_node(data, c_data2, node, depth, name, d_list, uv_list)

    depth -= 1
    if hasattr(node, 'left'):
        cal(node.left, 'left', name)
    if hasattr(node, 'right'):
        cal(node.right, 'right', name)

# nearest triangle distance calculation
def d_cal(data_A, data_B):
    point = data_A['vertices']
    n, _ = point.shape  

    closestPoint = np.full((n, 3), 0)
    closestPointIndex = np.full((n), 0)
    closestDistance = np.full((n), float('inf'))

    if (data_B['faces'].shape[0] == 0): return closestDistance

    triVertices = data_B['vertices'][data_B['faces'][:,:,0] - 1]
    tri = Triangle(triVertices)

    for i in range(triVertices.shape[0]):
        target = tri.closestPointToPointGPU(point, i)
        d = np.linalg.norm(point - target, axis=1)
        closestDistance = np.minimum(closestDistance, d)
        closestPoint = np.where((closestDistance == d)[..., np.newaxis], target, closestPoint)
        closestPointIndex = np.where((closestDistance == d), i, closestPointIndex)

    # sign: above or below the surface (1 or -1)
    triNormals = data_B['normals'][data_B['faces'][:,:,0] - 1]
    nor = triNormals[closestPointIndex][:,0]
    dir = point - closestPoint
    t = np.sum(nor * dir, axis=1)
    sign = np.where(t < 0, -1, 1)
    closestDistance *= sign

    return closestDistance

def plot(d, uv, w, h, filename):
    # d_max = np.max(np.abs(d))
    d_max = 10
    dotSize = 3
    d = d / d_max

    image = np.zeros((h, w, 3), dtype=np.uint8)

    for d, uv in zip(d, uv):
        u, v = uv
        u = int(w * u)
        v = int(h * (1-v))

        value = min(255 * abs(d), 255)

        if d > 0: color = (0, value, 0)
        if not d > 0: color = (0, 0, value)
        # color = (value, value, value)
        cv2.circle(image, (u, v), dotSize, color, -1)

    cv2.imshow('distance', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(filename, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation inspection / evaluation')
    parser.add_argument('--mode', type=str, help='preprocess, inspect, plot')
    parser.add_argument('--name', type=str, help='output folder name', default='')
    parser.add_argument('--path', type=str, help='segment OBJ path', default='')
    parser.add_argument('--A', type=str, help='segment A folder name', default='')
    parser.add_argument('--B', type=str, help='segment B folder name', default='')
    parser.add_argument('--i', type=str, help='input path', default='')
    parser.add_argument('--o', type=str, help='output path', default='')
    parser.add_argument('--w', type=int, help='image width', default=0)
    parser.add_argument('--h', type=int, help='image height', default=0)
    args = parser.parse_args()

    # cut the segment
    if (args.mode == 'preprocess'):
        folderName = os.path.join('output', args.name)
        shutil.rmtree(folderName, ignore_errors=True)
        os.makedirs(folderName)
        
        data = parse_obj(args.path)
        preprocess(data, folderName)

    # inspect
    if (args.mode == 'inspect'):
        dList = []
        uvList = []

        def sort_by_layer(filename):
            match = re.search(r'z(\d+)_d', filename)
            layer = int(match.group(1))
            return layer

        files = glob.glob(os.path.join('output', args.A, '*.obj'))
        sorted_filenames = sorted(files, key=sort_by_layer)
        # sorted_filenames = sorted_filenames[:3]

        for path in sorted_filenames:
            name = os.path.basename(path)

            path_A = os.path.join('output', args.A, name)
            path_B = os.path.join('output', args.B, name)

            if not os.path.exists(path_A) or not os.path.exists(path_B): continue
            print(f'processing { name }')

            data_A = parse_obj(path_A)
            data_B = parse_obj(path_B)

            bvh = MeshBVH(data_A)
            data = bvh.data

            node = bvh._roots[0]
            sub_d, sub_uv = save_n(data, data_B, node, depth=5)

            dList.append(sub_d)
            uvList.append(sub_uv)

        d = np.concatenate(dList, axis=0)
        uv = np.concatenate(uvList, axis=0)
        d = np.around(d, decimals=5)

        np.savez(args.o, d=d, uv=uv)

    # inspect
    if (args.mode == 'plot'):
        data = np.load(args.i)

        d = data['d']
        uv = data['uv']

        plot(d, uv, args.w, args.h, args.o)




