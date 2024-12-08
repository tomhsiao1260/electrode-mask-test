import os
import argparse
import numpy as np
from path_utils import obj_path
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutLayer, cutBounding

# python cut_obj.py --x 3380 --y 2533 --z 5049 --chunk 768
# python cut_obj.py --x 3380 --y 1765 --z 5049 --chunk 768
# python cut_obj.py --x 3380 --y 2533 --z 4281 --chunk 768
# python cut_obj.py --x 3380 --y 1765 --z 4281 --chunk 768
# python cut_obj.py --x 3400 --y 1900 --z 3513 --chunk 768
# python cut_obj.py --x 3413 --y 1831 --z 2736 --chunk 768
# python cut_obj.py --x 3424 --y 1860 --z 1968 --chunk 768

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate volume via boudning box')
    parser.add_argument('--o', type=str, help='OBJ output path')
    parser.add_argument('--x', type=int, help='minimium x')
    parser.add_argument('--y', type=int, help='minimium y')
    parser.add_argument('--z', type=int, help='minimium z')
    parser.add_argument('--chunk', type=int, default=256, help='chunk size')
    args = parser.parse_args()

    chunk = args.chunk
    xmin, ymin, zmin = args.x, args.y, args.z

    boxSize = np.array([chunk, chunk, chunk])
    boxMin = np.array([xmin, ymin, zmin])
    boxMax = boxMin + boxSize

    data = parse_obj(obj_path)
    # cut a given .obj along z-axis
    cutLayer(data, layerMin = boxMin[2], layerMax = boxMax[2])
    # cut a given .obj along bounding box
    cutBounding(data, boxMin, boxMax)

    directory = f"./output/{zmin:05}_{ymin:05}_{xmin:05}"
    filename = f"{zmin:05}_{ymin:05}_{xmin:05}_20230702185753.obj"

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    save_obj(path, data)

