import argparse
import numpy as np
from path_utils import obj_path
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutLayer, cutBounding

# python cut_obj.py --o output/segment.obj --min 3400 1900 3513 --size 768 768 768
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate volume via boudning box')
    parser.add_argument('--o', type=str, help='OBJ output path')
    parser.add_argument('--min', nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='bounding box minimum (x, y, z)')
    parser.add_argument('--size', nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='bounding box size (w, h, d)')
    args = parser.parse_args()

    boxSize = np.array(args.size)
    boxMin = np.array(args.min)
    boxMax = boxMin + boxSize

    data = parse_obj(obj_path)
    # cut a given .obj along z-axis
    cutLayer(data, layerMin = boxMin[2], layerMax = boxMax[2])
    # cut a given .obj along bounding box
    cutBounding(data, boxMin, boxMax)

    save_obj(args.o, data)