import nrrd
import shutil
import tifffile
import argparse
import numpy as np
from path_utils import label_path
from utils.draw import drawImage
from core.utils.loader import parse_obj
from utils.sdf import calculateSDF, calculateUV, calculateLabel

# distance field calculation
def getSDF(data, boxMin, boxMax):
    pStack, iStack, dStack = calculateSDF(data, boxMin, boxMax)

    # z, x, y -> x, y, z
    nrrdStack = np.transpose(dStack * 255, (1, 2, 0)).astype(np.uint8)
    nrrd.write('output/sdf.nrrd', nrrdStack)
    shutil.copy('output/sdf.nrrd' , 'client/public')
    # z, x, y -> z, y, x
    imageStack = np.transpose(dStack * 65536, (0, 2, 1)).astype(np.uint16)
    tifffile.imwrite('output/sdf.tif', imageStack)
    # debug
    # drawImage('output/sdf.tif')

    return pStack, iStack, dStack

# inklabels 3D mapping
def getLabel(data, pStack, iStack, path):
    indices = data['faces'][iStack][:, :, :, :, 0] - 1
    triUVs = data['uvs'][indices]
    triVertices = data['vertices'][indices]

    uvStack = calculateUV(pStack, triVertices, triUVs)
    labelStack = calculateLabel(path, uvStack)

    # z, x, y -> x, y, z
    nrrdStack = np.transpose(labelStack * 255, (1, 2, 0)).astype(np.uint8)
    nrrd.write('output/label.nrrd', nrrdStack)
    shutil.copy('output/label.nrrd' , 'client/public')
    # z, x, y -> z, y, x
    imageStack = np.transpose(labelStack * 65535, (0, 2, 1)).astype(np.uint16)
    tifffile.imwrite('output/label.tif', imageStack)
    # debug
    # drawImage('output/label.tif')

    return labelStack

# python get_sdf.py --i output/segment.obj --min 3400 1900 3513 --size 768 768 768
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SDF & Inklabels via boudning box')
    parser.add_argument('--i', type=str, help='OBJ input path')
    parser.add_argument('--min', nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='bounding box minimum (x, y, z)')
    parser.add_argument('--size', nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='bounding box size (w, h, d)')
    args = parser.parse_args()

    boxSize = np.array(args.size)
    boxMin = np.array(args.min)
    boxMax = boxMin + boxSize

    # extract sdf
    data = parse_obj(args.i)
    print('Distance field calculating ...')
    pStack, iStack, dStack = getSDF(data, boxMin, boxMax)

    # extract inklabels
    print('Inklabels calculating ...')
    getLabel(data, pStack, iStack, label_path)
