import argparse
import numpy as np
from utils.volume import getGridName
from core.utils.loader import parse_obj
from path_utils import obj_path

# python find_grid.py --u 0.521 --v 0.492
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate which grids are required.')
    parser.add_argument('--u', type=float, help='U value in UV coordinate')
    parser.add_argument('--v', type=float, help='V value in UV coordinate')
    args = parser.parse_args()

    data = parse_obj(obj_path)
    uv = np.array([ args.u, args.v ])
    d = np.sum((data['uvs'] - uv) ** 2, axis=1)
    index = np.argmin(d)

    center = data['vertices'][index]
    boxSize = np.array([ 256, 256, 256 ])
    boxMin = center - boxSize / 2
    boxMax = boxMin + boxSize
    boxMin = boxMin.astype(int)
    boxMax = boxMax.astype(int)
    nameList = getGridName(boxMin, boxMax)

    print('The following grids are required:')
    for grid in nameList: print(grid)
    print('')
    print('Volume Center (x, y, z):')
    print(f'{center[0]} {center[1]} {center[2]}')
    print('')
    print('Volume Size (w, h, d):')
    print(f'{boxSize[0]} {boxSize[1]} {boxSize[2]}')
    print('')
    print('Volume Minimum (x, y, z):')
    print(f'{boxMin[0]} {boxMin[1]} {boxMin[2]}')
    print('')



