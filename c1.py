# to use open3d, use hello-world-geometry env

import os
import argparse
import numpy as np
import open3d as o3d
from core.utils.loader import parse_obj, save_obj
from core.utils.cut import re_index

# python c1.py --x 2432 --y 2304 --z 10624 --chunk 768
# python c1.py --x 2990 --y 1800 --z 1200 --chunk 768 # casey pi

# python c1.py --x 2860 --y 2853 --z 9657 --chunk 768
# python c1.py --x 2860 --y 3621 --z 9657 --chunk 768
# python c1.py --x 2973 --y 2769 --z 8889 --chunk 768
# python c1.py --x 2973 --y 3537 --z 8889 --chunk 768
# python c1.py --x 3140 --y 2700 --z 8121 --chunk 768
# python c1.py --x 3140 --y 3468 --z 8121 --chunk 768
# python c1.py --x 3264 --y 2666 --z 7353 --chunk 768
# python c1.py --x 3264 --y 3434 --z 7353 --chunk 768
# python c1.py --x 3360 --y 2666 --z 6585 --chunk 768
# python c1.py --x 3360 --y 3434 --z 6585 --chunk 768
# python c1.py --x 3360 --y 2373 --z 5817 --chunk 768
# python c1.py --x 3360 --y 3141 --z 5817 --chunk 768

# python c1.py --x 3380 --y 2533 --z 5049 --chunk 768
# python c1.py --x 3380 --y 1765 --z 5049 --chunk 768
# python c1.py --x 3380 --y 2533 --z 4281 --chunk 768
# python c1.py --x 3380 --y 1765 --z 4281 --chunk 768
# python c1.py --x 3400 --y 1900 --z 3513 --chunk 768
# python c1.py --x 3413 --y 1831 --z 2736 --chunk 768
# python c1.py --x 3424 --y 1860 --z 1968 --chunk 768

# python c1.py --x 3490 --y 1537 --z 1200 --chunk 768
# python c1.py --x 3490 --y 2305 --z 1200 --chunk 768
# python c1.py --x 3574 --y 1693 --z 432 --chunk 768
# python c1.py --x 3574 --y 2461 --z 432 --chunk 768
# python c1.py --x 3674 --y 1722 --z 0 --chunk 768
# python c1.py --x 3674 --y 2490 --z 0 --chunk 768

# 1b

# python c1.py --x 4008 --y 2551 --z 6039 --chunk 768
# python c1.py --x 4008 --y 1783 --z 6039 --chunk 768
# python c1.py --x 4075 --y 2815 --z 5271 --chunk 768
# python c1.py --x 4075 --y 2047 --z 5271 --chunk 768
# python c1.py --x 4380 --y 3154 --z 4503 --chunk 768
# python c1.py --x 4380 --y 2386 --z 4503 --chunk 768
# python c1.py --x 4347 --y 3780 --z 3903 --chunk 768
# python c1.py --x 4400 --y 2963 --z 3903 --chunk 768

def cluster_obj(filename):
    data = parse_obj(filename)
    mesh = o3d.io.read_triangle_mesh(filename)

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    cluster_data_list = []
    cluster_uv_list = []
    for i in range(len(cluster_n_triangles)):
        cluster_data = data.copy()
        cluster_data['faces'] = data['faces'][triangle_clusters == i]

        if(cluster_data['faces'].shape[0] < 100): continue

        re_index(cluster_data)
        cluster_data_list.append(cluster_data)
        cluster_uv_list.append(np.mean(cluster_data['uvs'], axis=0))

    sorted_id = np.argsort(np.array(cluster_uv_list)[:, 0])
    # sorted_id = sorted_id if not REVERSE else np.max(sorted_id) - sorted_id

    for i, cluster_data in enumerate(cluster_data_list):
        dirname = os.path.dirname(filename)
        save_obj(os.path.join(dirname, f'{i}.obj'), cluster_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clustering OBJ File')
    parser.add_argument('--o', type=str, help='OBJ output path')
    parser.add_argument('--x', type=int, help='minimium x')
    parser.add_argument('--y', type=int, help='minimium y')
    parser.add_argument('--z', type=int, help='minimium z')
    parser.add_argument('--chunk', type=int, default=256, help='chunk size')
    args = parser.parse_args()

    chunk = args.chunk
    xmin, ymin, zmin = args.x, args.y, args.z

    directory = f"./output/b{zmin:05}_{ymin:05}_{xmin:05}" # use 1b
    filename = f"{zmin:05}_{ymin:05}_{xmin:05}_20240101215220.obj" # use 1b

    # directory = f"./output/{zmin:05}_{ymin:05}_{xmin:05}" # use 1a
    # filename = f"{zmin:05}_{ymin:05}_{xmin:05}_20230702185753.obj" # use 1a
    path = os.path.join(directory, filename)

    cluster_obj(path)




