import os
import re
import nrrd
import glob
import shutil
import tifffile
import argparse
import numpy as np

from tqdm import tqdm
from core.MeshBVH import MeshBVH
from core.math.Triangle import Triangle
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutDivide, cutBounding, re_index
from utils.sdf import closestPointToPoint

# python c2.py --x 2432 --y 2304 --z 10624 --chunk 768
# python c2.py --x 2990 --y 1800 --z 1200 --chunk 768 # casey pi

# python c2.py --x 2860 --y 2853 --z 9657 --chunk 768
# python c2.py --x 2860 --y 3621 --z 9657 --chunk 768
# python c2.py --x 2973 --y 2769 --z 8889 --chunk 768
# python c2.py --x 2973 --y 3537 --z 8889 --chunk 768
# python c2.py --x 3140 --y 2700 --z 8121 --chunk 768
# python c2.py --x 3140 --y 3468 --z 8121 --chunk 768
# python c2.py --x 3264 --y 2666 --z 7353 --chunk 768
# python c2.py --x 3264 --y 3434 --z 7353 --chunk 768
# python c2.py --x 3360 --y 2666 --z 6585 --chunk 768
# python c2.py --x 3360 --y 3434 --z 6585 --chunk 768
# python c2.py --x 3360 --y 2373 --z 5817 --chunk 768
# python c2.py --x 3360 --y 3141 --z 5817 --chunk 768

# python c2.py --x 3380 --y 2533 --z 5049 --chunk 768
# python c2.py --x 3380 --y 1765 --z 5049 --chunk 768
# python c2.py --x 3380 --y 2533 --z 4281 --chunk 768
# python c2.py --x 3380 --y 1765 --z 4281 --chunk 768
# python c2.py --x 3400 --y 1900 --z 3513 --chunk 768
# python c2.py --x 3413 --y 1831 --z 2736 --chunk 768
# python c2.py --x 3424 --y 1860 --z 1968 --chunk 768

# python c2.py --x 3490 --y 1537 --z 1200 --chunk 768
# python c2.py --x 3490 --y 2305 --z 1200 --chunk 768
# python c2.py --x 3574 --y 1693 --z 432 --chunk 768
# python c2.py --x 3574 --y 2461 --z 432 --chunk 768
# python c2.py --x 3674 --y 1722 --z 0 --chunk 768
# python c2.py --x 3674 --y 2490 --z 0 --chunk 768

# 1b

# python c2.py --x 4008 --y 2551 --z 6039 --chunk 768
# python c2.py --x 4008 --y 1783 --z 6039 --chunk 768
# python c2.py --x 4075 --y 2815 --z 5271 --chunk 768
# python c2.py --x 4075 --y 2047 --z 5271 --chunk 768
# python c2.py --x 4380 --y 3154 --z 4503 --chunk 768
# python c2.py --x 4380 --y 2386 --z 4503 --chunk 768
# python c2.py --x 4347 --y 3780 --z 3903 --chunk 768
# python c2.py --x 4400 --y 2963 --z 3903 --chunk 768

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
    parser = argparse.ArgumentParser(description='Transform OBJ into Mask')
    parser.add_argument('--x', type=int, help='minimium x')
    parser.add_argument('--y', type=int, help='minimium y')
    parser.add_argument('--z', type=int, help='minimium z')
    parser.add_argument('--chunk', type=int, default=256, help='chunk size')
    args = parser.parse_args()

    chunk = args.chunk
    xmin, ymin, zmin = args.x, args.y, args.z
    factor, label_list = 8, [1, 2]
    # factor, label_list = 8, [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # directory = f"./output/{zmin:05}_{ymin:05}_{xmin:05}" # use 1a
    directory = f"./output/b{zmin:05}_{ymin:05}_{xmin:05}" # use 1b
    nrrd_path = os.path.join(directory, f"{zmin:05}_{ymin:05}_{xmin:05}_mask.nrrd")
    tif_path = os.path.join(directory, f"{zmin:05}_{ymin:05}_{xmin:05}_mask.tif")

    # turn 1.obj, 2.obj, ... into 1, 2, ... in nrrd mask label value
    for label in label_list:
        # obj that you want to label
        obj_path = os.path.join(directory, f"{label}.obj")
        if not os.path.exists(obj_path): continue

        # processing
        print('processing label ', label, ' ...')

        data = parse_obj(obj_path)
        nrrdStack = np.zeros((chunk//factor, chunk//factor, chunk//factor), dtype=np.uint8)
        imageStack = np.zeros((chunk//factor, chunk//factor, chunk//factor), dtype=np.uint8)

        boxSize = np.array((chunk, chunk, chunk))
        boxMin = np.array((xmin, ymin, zmin))
        boxMax = boxMin + boxSize
        pStack, iStack, dStack = calculateSDF(data, boxMin, boxMax, factor)

        # z, x, y -> z, y, x
        mask = (np.transpose(dStack, (0, 2, 1)) < 0.01)
        nrrdStack[mask] = label
        imageStack[mask] = 255

        # init if mask data don't exist
        if not os.path.exists(nrrd_path):
            emptyStack = np.zeros((chunk, chunk, chunk), dtype=np.uint8)
            nrrd.write(nrrd_path, emptyStack)

        # back to original shape
        expanded_nrrd = np.zeros((chunk, chunk, chunk), dtype=np.uint8)

        for i in range(factor):
            for j in range(factor):
                for k in range(factor):
                    expanded_nrrd[i::factor, j::factor, k::factor] = nrrdStack

        original_nrrd, header = nrrd.read(nrrd_path)
        original_nrrd[expanded_nrrd == label] = label
        nrrd.write(nrrd_path, original_nrrd)

        original_tif = np.zeros((chunk, chunk, chunk), dtype=np.uint8)
        original_tif[original_nrrd > 0] = 255
        tifffile.imwrite(tif_path, original_tif)


