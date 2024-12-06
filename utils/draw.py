import os
import cv2
import tifffile
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
from core.objects.MeshBVHHelper import MeshBVHHelper

def drawImage(path):
    data = tifffile.imread(path)

    for layer in range(data.shape[0]):
        image = data[layer, :, :]

        cv2.imshow('Volume', image)
        # cv2.waitKey(0)
        cv2.waitKey(10)
        cv2.destroyAllWindows()

# def drawUV(bvh):
#     w, h = 869, 675
#     image = np.zeros((h, w, 3), dtype=np.uint8)

#     # drawUVNode(image, bvh, bvh._roots[0].left, color = (255, 0, 0))
#     # drawUVNode(image, bvh, bvh._roots[0].right, color = (0, 255, 0))
#     drawUVNode(image, bvh, bvh._roots[0].left.left.left.left, color = (0, 0, 255))
#     # drawUVNode(image, bvh, bvh._roots[0].left.left.left.right, color = (0, 255, 255))

#     cv2.imshow('UV', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def drawUVs(data):
#     w, h = 869, 675
#     color = (255, 255, 255)
#     image = np.zeros((h, w, 3), dtype=np.uint8)

#     triIndices = data['faces']
#     triUVs = data['uvs'][triIndices[:,:,0] - 1]

#     triUVs[:, :, 0] = w * triUVs[:, :, 0]
#     triUVs[:, :, 1] = h * (1 - triUVs[:, :, 1])
#     triUVs = triUVs.astype(int)

#     for tri in triUVs:
#         cv2.line(image, tuple(tri[0]), tuple(tri[1]), color, 1)
#         cv2.line(image, tuple(tri[1]), tuple(tri[2]), color, 1)
#         cv2.line(image, tuple(tri[2]), tuple(tri[0]), color, 1)

#     # cv2.imshow('UV', image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     silhouette_scores = []
#     for n_clusters in range(2, 11):
#         kmeans = KMeans(n_clusters=n_clusters)
#         cluster_labels = kmeans.fit_predict(data['uvs'])
#         silhouette_avg = silhouette_score(data['uvs'], cluster_labels)
#         silhouette_scores.append(silhouette_avg)

#     best_n_clusters = np.argmax(silhouette_scores) + 2
#     print("Best number of clusters:", best_n_clusters)

#     kmeans = KMeans(n_clusters=best_n_clusters)
#     cluster_labels = kmeans.fit_predict(data['uvs'])

#     cluster_0_data = data['uvs'][np.where(cluster_labels == 0)[0]]
#     cluster_1_data = data['uvs'][np.where(cluster_labels == 1)[0]]

#     box_0_min = np.min(cluster_0_data, axis=0)
#     box_0_max = np.max(cluster_0_data, axis=0)

#     c = (box_0_max + box_0_min) / 2
#     l = box_0_max - box_0_min

#     inklabelPath = 'model/SPOILER_20230702185753.png'
#     labelImage = cv2.imread(inklabelPath, cv2.IMREAD_UNCHANGED)
#     hh, ww = labelImage.shape[:2]

#     v = cluster_0_data
#     v[:, 0] = ww * v[:, 0]
#     v[:, 1] = hh * (1 - v[:, 1])
#     v = v.astype(int)

#     o_min = np.min(v, axis=0)
#     o_max = np.max(v, axis=0)

#     image = labelImage[o_min[0]:o_max[0], o_min[1]:o_max[1]]
#     h, w = image.shape[:2]

#     cv2.imshow('UV', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     uvs = data['uvs']
#     uvs -= c
#     uvs /= l
#     uvs += 0.5
#     uvs[:, 0] = w * uvs[:, 0]
#     uvs[:, 1] = h * (1 - uvs[:, 1])
#     uvs = uvs.astype(int)

#     for tri in data['faces'][:,:,0] - 1:
#         if (cluster_labels[tri[0]] != 0): continue
#         u = uvs[tri]

#         cv2.line(image, tuple(u[0]), tuple(u[1]), color, 1)
#         cv2.line(image, tuple(u[1]), tuple(u[2]), color, 1)
#         cv2.line(image, tuple(u[2]), tuple(u[0]), color, 1)

#     cv2.imshow('UV', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def drawUVNode(image, bvh, node, color = (255, 255, 255)):
    h, w, _ = image.shape
    offset = node._offset
    count = node._count

    triIndices = bvh.data['faces'][offset: offset + count]
    triUVs = bvh.data['uvs'][triIndices[:,:,0] - 1]

    # triUVs = triUVs[triUVs[:,0,1] < 0.6]

    # print(np.max(np.max(triUVs, axis=1), axis=0))
    # print(np.min(np.min(triUVs, axis=1), axis=0))

    triUVs[:, :, 0] = w * triUVs[:, :, 0]
    triUVs[:, :, 1] = h * (1 - triUVs[:, :, 1])
    triUVs = triUVs.astype(int)

    for tri in triUVs:
        cv2.line(image, tuple(tri[0]), tuple(tri[1]), color, 1)
        cv2.line(image, tuple(tri[1]), tuple(tri[2]), color, 1)
        cv2.line(image, tuple(tri[2]), tuple(tri[0]), color, 1)

def drawBoxes(bvh):
    vertices = bvh.data['vertices']
    center = np.mean(vertices, axis=0)
    boxMin = np.min(vertices, axis=0)
    boxMax = np.max(vertices, axis=0)
    windowSize = 2 * np.max(np.maximum(boxMin - center, boxMax - center))
    windowSize = int(1.5 * windowSize)

    depth = 0
    helper = MeshBVHHelper(bvh)

    while (True):
        if (helper.draw(center, windowSize, depth)): depth += 1
        else: break