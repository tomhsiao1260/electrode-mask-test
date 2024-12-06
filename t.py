import os
import re
import nrrd
import glob
import shutil
import tifffile
import numpy as np

from tqdm import tqdm
from core.MeshBVH import MeshBVH
from core.math.Triangle import Triangle
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutDivide, cutBounding, re_index

from utils.sdf import closestPointToPoint

def sort_by_layer(filename):
    match = re.search(r'z(\d+)_d', filename)
    layer = int(match.group(1))
    return layer

# distance field calculation
def calculateSDF(data, boxMin, boxMax, factor):
    center = (boxMin + boxMax) / 2
    sampling = (1/factor * (boxMax - boxMin)).astype('int')
    windowSize = 1.0 * (boxMax - boxMin)
    maxDistance = np.max(windowSize[:2])

    pStack = []
    iStack = []
    dStack = []

    i, j = np.meshgrid(np.arange(sampling[0]), np.arange(sampling[1]), indexing='ij')

    # for layer in tqdm(range(int(boxMin[2]) + 50, int(boxMin[2]) + 52, 1)):
    for layer in tqdm(range(int(boxMin[2]), int(boxMax[2]), factor)):
        x = center[0] - windowSize[0] / 2 + (i + 0.5) * windowSize[0] / sampling[0]
        y = center[1] - windowSize[1] / 2 + (j + 0.5) * windowSize[1] / sampling[1]
        z = layer * np.full_like(x, 1)
        p = np.stack((x, y, z), axis=-1)

        closestPoint, closestPointIndex, closestDistance = closestPointToPoint(data, p, layer)

        pStack.append(closestPoint)
        iStack.append(closestPointIndex)
        dStack.append(closestDistance / maxDistance)

    return np.array(pStack), np.array(iStack), np.array(dStack)

if __name__ == "__main__":
    z, y, x, size = 3513, 1900, 3400, 768
    factor, label, init = 8, 1, False

    # obj_path = './output/2.obj' # label 2
    obj_path = './output/0.obj' # label 1
    nrrd_path = f'./output/o.nrrd'
    tif_path = f'./output/o.tif'

    if (init):
        nrrdStack = np.zeros((size//factor, size//factor, size//factor), dtype=np.uint8)
        imageStack = np.zeros((size//factor, size//factor, size//factor), dtype=np.uint8)

        nrrd.write(nrrd_path, nrrdStack)
        tifffile.imwrite(tif_path, imageStack)

    imageStack = tifffile.imread(tif_path)
    nrrdStack, header = nrrd.read(nrrd_path)
    nrrdStack = np.asarray(nrrdStack)

    # processing
    data = parse_obj(obj_path)

    boxSize = np.array((size, size, size))
    boxMin = np.array((x, y, z))
    boxMax = boxMin + boxSize

    pStack, iStack, dStack = calculateSDF(data, boxMin, boxMax, factor)

    # z, x, y -> z, y, x
    mask = (np.transpose(dStack, (0, 2, 1)) < 0.01)
    nrrdStack[mask] = label
    # z, x, y -> z, y, x
    mask = (np.transpose(dStack, (0, 2, 1)) < 0.01)
    imageStack[mask] = 255

    nrrd.write(nrrd_path, nrrdStack)
    tifffile.imwrite(tif_path, imageStack)

    # back to original shape
    expanded_nrrd = np.zeros((768, 768, 768), dtype=np.uint8)
    expanded_tif = np.zeros((768, 768, 768), dtype=np.uint8)

    for i in range(factor):
        for j in range(factor):
            for k in range(factor):
                expanded_nrrd[i::factor, j::factor, k::factor] = nrrdStack

    expanded_tif[expanded_nrrd > 0] = 255
    nrrd.write(f'./output/{z:05}_{y:05}_{x:05}_mask.nrrd', expanded_nrrd)
    tifffile.imwrite(f'./output/{z:05}_{y:05}_{x:05}_mask.tif', expanded_tif)


