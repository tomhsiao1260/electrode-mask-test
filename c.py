import numpy as np
import open3d as o3d
from core.utils.loader import parse_obj, save_obj

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
        filename = f'./output/{i}.obj'
        save_obj(filename, cluster_data)

# to use open3d, use hello-world-geometry env
if __name__ == "__main__":
    cluster_obj('./output/segment.obj')




