import numpy as np

def closestPointToPoint(bvh, point, minThreshold, maxThreshold):
    # early out if under minThreshold
    # skip checking if over maxThreshold
    # set minThreshold = maxThreshold to quickly check if a point is within a threshold
    # returns Infinity if no value found
    minThresholdSq = minThreshold * minThreshold
    maxThresholdSq = maxThreshold * maxThreshold
    closestPoint = None
    closestDistanceSq = float('inf')
    closestDistanceTriIndex = None

    def boundsTraverseOrder(boxMin, boxMax):
        q = np.minimum(point, boxMax)
        q = np.maximum(q, boxMin)
        return np.sum((point - q)**2)

    def intersectsBounds(boxMin, boxMax, isLeaf, score):
        return score < closestDistanceSq and score < maxThresholdSq

    def intersectsTriangle(tri, triIndex):
        nonlocal minThresholdSq, maxThresholdSq
        nonlocal closestPoint, closestDistanceSq, closestDistanceTriIndex

        q = tri.closestPointToPoint(point, triIndex)
        distSq = np.sum((point - q)**2)
        if (distSq < closestDistanceSq):
            closestPoint = q
            closestDistanceSq = distSq
            closestDistanceTriIndex = triIndex

            if (distSq < minThresholdSq): return True
            else: return False

    bvh.shapecast({
        'boundsTraverseOrder': boundsTraverseOrder,
        'intersectsBounds': intersectsBounds,
        'intersectsTriangle': intersectsTriangle
    })

    if (closestDistanceSq == float('inf')): return None
    point = closestPoint
    distance = np.sqrt(closestDistanceSq)
    faceIndex = closestDistanceTriIndex

    return point, distance, faceIndex
