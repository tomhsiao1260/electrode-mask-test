import os
import argparse
import numpy as np
from path_utils import obj_path
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import cutLayer, cutBounding

# python cut_obj.py --x 2432 --y 2304 --z 10624 --chunk 768
# python cut_obj.py --x 2990 --y 1800 --z 1200 --chunk 768 # casey pi

# python cut_obj.py --x 2860 --y 2853 --z 9657 --chunk 768
# python cut_obj.py --x 2860 --y 3621 --z 9657 --chunk 768
# python cut_obj.py --x 2973 --y 2769 --z 8889 --chunk 768
# python cut_obj.py --x 2973 --y 3537 --z 8889 --chunk 768
# python cut_obj.py --x 3140 --y 2700 --z 8121 --chunk 768
# python cut_obj.py --x 3140 --y 3468 --z 8121 --chunk 768
# python cut_obj.py --x 3264 --y 2666 --z 7353 --chunk 768
# python cut_obj.py --x 3264 --y 3434 --z 7353 --chunk 768
# python cut_obj.py --x 3360 --y 2666 --z 6585 --chunk 768
# python cut_obj.py --x 3360 --y 3434 --z 6585 --chunk 768
# python cut_obj.py --x 3360 --y 2373 --z 5817 --chunk 768
# python cut_obj.py --x 3360 --y 3141 --z 5817 --chunk 768

# python cut_obj.py --x 3380 --y 2533 --z 5049 --chunk 768
# python cut_obj.py --x 3380 --y 1765 --z 5049 --chunk 768
# python cut_obj.py --x 3380 --y 2533 --z 4281 --chunk 768
# python cut_obj.py --x 3380 --y 1765 --z 4281 --chunk 768
# python cut_obj.py --x 3400 --y 1900 --z 3513 --chunk 768
# python cut_obj.py --x 3413 --y 1831 --z 2736 --chunk 768
# python cut_obj.py --x 3424 --y 1860 --z 1968 --chunk 768

# python cut_obj.py --x 3490 --y 1537 --z 1200 --chunk 768
# python cut_obj.py --x 3490 --y 2305 --z 1200 --chunk 768
# python cut_obj.py --x 3574 --y 1693 --z 432 --chunk 768
# python cut_obj.py --x 3574 --y 2461 --z 432 --chunk 768
# python cut_obj.py --x 3674 --y 1722 --z 0 --chunk 768
# python cut_obj.py --x 3674 --y 2490 --z 0 --chunk 768

# 1b

# python cut_obj.py --x 4008 --y 2551 --z 6039 --chunk 768
# python cut_obj.py --x 4008 --y 1783 --z 6039 --chunk 768
# python cut_obj.py --x 4075 --y 2815 --z 5271 --chunk 768
# python cut_obj.py --x 4075 --y 2047 --z 5271 --chunk 768
# python cut_obj.py --x 4380 --y 3154 --z 4503 --chunk 768
# python cut_obj.py --x 4380 --y 2386 --z 4503 --chunk 768
# python cut_obj.py --x 4347 --y 3780 --z 3903 --chunk 768
# python cut_obj.py --x 4400 --y 2963 --z 3903 --chunk 768


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

    # data = parse_obj(obj_path) # use 1a
    data = parse_obj('/Users/yao/Desktop/full-scrolls/Scroll1/PHercParis4.volpkg/paths/20240101215220/20240101215220.obj') # use 1b
    # cut a given .obj along z-axis
    cutLayer(data, layerMin = boxMin[2], layerMax = boxMax[2])
    # cut a given .obj along bounding box
    cutBounding(data, boxMin, boxMax)

    # directory = f"./output/{zmin:05}_{ymin:05}_{xmin:05}" # use 1a
    # filename = f"{zmin:05}_{ymin:05}_{xmin:05}_20230702185753.obj" # use 1a

    directory = f"./output/b{zmin:05}_{ymin:05}_{xmin:05}" # use 1b
    filename = f"{zmin:05}_{ymin:05}_{xmin:05}_20240101215220.obj" # use 1b

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    save_obj(path, data)

