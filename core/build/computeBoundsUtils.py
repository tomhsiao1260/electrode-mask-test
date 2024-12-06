import numpy as np
from core.Constants import FLOAT32_EPSILON

# computes the union of the bounds of all of the given triangles
def getBounds(data, triangleBounds, offset, count, target, centroidTarget):
    box_centers = triangleBounds[0][offset : offset + count]
    box_half_sizes = triangleBounds[1][offset : offset + count]

    box_min = box_centers - box_half_sizes
    box_max = box_centers + box_half_sizes

    target_min = np.min(box_min, axis=0)
    target_max = np.max(box_max, axis=0)
    target[:] = np.hstack((target_min, target_max))

    centroidTarget_min = np.min(box_centers, axis=0)
    centroidTarget_max = np.max(box_centers, axis=0)
    centroidTarget[:] = np.hstack((centroidTarget_min, centroidTarget_max))

# precomputes the bounding box for each triangle; required for quickly calculating tree splits
def computeTriangleBounds(data):
    vertices = data.get('vertices', np.array([]))
    faces    = data.get('faces'   , np.array([]))[:, :, 0]

    p = vertices[faces - 1]
    box_min = np.min(p, axis=1)
    box_max = np.max(p, axis=1)

    box_centers = (box_min + box_max) / 2
    box_half_sizes = (box_max - box_min) / 2 * (1 + FLOAT32_EPSILON * 10)

    return box_centers, box_half_sizes
