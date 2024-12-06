import os
import tifffile
import numpy as np
from path_utils import grid_folder

def getGridName(boxMin, boxMax):
    lower = (boxMin // 500 + 1).astype('int')
    upper = (boxMax // 500 + 1).astype('int')
    nameList = []

    for x in range(lower[0], upper[0] + 1):
        for y in range(lower[1], upper[1] + 1):
            for z in range(lower[2], upper[2] + 1):
                start = boxMin - np.array([x - 1, y - 1, z - 1]) * 500
                end = boxMax - np.array([x - 1, y - 1, z - 1]) * 500
                start = np.maximum(np.minimum(start, 500), 0).astype('int')
                end = np.maximum(np.minimum(end, 500), 0).astype('int')

                name = 'cell_yxz_{:03d}_{:03d}_{:03d}.tif'.format(y, x, z)
                nameList.append(name)

    return nameList

def calculateVolume(boxMin, boxMax):
    lower = (boxMin // 500 + 1).astype('int')
    upper = (boxMax // 500 + 1).astype('int')

    xStack = []
    for x in range(lower[0], upper[0] + 1):
        yStack = []
        for y in range(lower[1], upper[1] + 1):
            zStack = []
            for z in range(lower[2], upper[2] + 1):
                start = boxMin - np.array([x - 1, y - 1, z - 1]) * 500
                end = boxMax - np.array([x - 1, y - 1, z - 1]) * 500
                start = np.maximum(np.minimum(start, 500), 0).astype('int')
                end = np.maximum(np.minimum(end, 500), 0).astype('int')

                name = 'cell_yxz_{:03d}_{:03d}_{:03d}.tif'.format(y, x, z)
                path = os.path.join(grid_folder, name)
                # z, y, x
                data = tifffile.imread(path)
                data = data[start[2]:end[2], start[1]:end[1], start[0]:end[0]]

                zStack.append(data)
            yStack.append(np.concatenate(zStack, axis=0))
        xStack.append(np.concatenate(yStack, axis=1))

    volumeStack = np.concatenate(xStack, axis=2)
    volumeStack = volumeStack / np.max(volumeStack)
    return volumeStack
