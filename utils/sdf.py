import cv2
import numpy as np
from tqdm import tqdm
from core.math.Triangle import Triangle

# distance field calculation
def calculateSDF(data, boxMin, boxMax):
    center = (boxMin + boxMax) / 2
    sampling = (1.0 * (boxMax - boxMin)).astype('int')
    windowSize = 1.0 * (boxMax - boxMin)
    maxDistance = np.max(windowSize[:2])

    pStack = []
    iStack = []
    dStack = []

    i, j = np.meshgrid(np.arange(sampling[0]), np.arange(sampling[1]), indexing='ij')

    # for layer in tqdm(range(int(boxMin[2]) + 50, int(boxMin[2]) + 52, 1)):
    for layer in tqdm(range(int(boxMin[2]), int(boxMax[2]), 1)):
        x = center[0] - windowSize[0] / 2 + (i + 0.5) * windowSize[0] / sampling[0]
        y = center[1] - windowSize[1] / 2 + (j + 0.5) * windowSize[1] / sampling[1]
        z = layer * np.full_like(x, 1)
        p = np.stack((x, y, z), axis=-1)

        closestPoint, closestPointIndex, closestDistance = closestPointToPoint(data, p, layer)

        pStack.append(closestPoint)
        iStack.append(closestPointIndex)
        dStack.append(closestDistance / maxDistance)

    return np.array(pStack), np.array(iStack), np.array(dStack)

def closestPointToPoint(data, point, layer):
    w, h, _ = point.shape    
    closestPoint = np.full((w, h, 3), 0)
    closestPointIndex = np.full((w, h), 0)
    closestDistance = np.full((w, h), float('inf'))

    gap = 30
    triVertices = data['vertices'][data['faces'][:,:,0] - 1]
    lowerBound = triVertices[:,0,2] > (layer - gap)
    upperBound = triVertices[:,0,2] < (layer + gap)
    mask = np.logical_and(lowerBound, upperBound)
    index = np.nonzero(mask)[0]
    triVertices = triVertices[mask]
    tri = Triangle(triVertices)

    for i in range(triVertices.shape[0]):
        target = tri.closestPointToPointGPU(point, i)
        d = np.linalg.norm(point - target, axis=2)
        closestDistance = np.minimum(closestDistance, d)
        closestPoint = np.where((closestDistance == d)[:, :, np.newaxis], target, closestPoint)
        closestPointIndex = np.where((closestDistance == d), index[i], closestPointIndex)

    return closestPoint, closestPointIndex, closestDistance

# uv interpolation
def calculateUV(point, triVertices, triUVs):
    d = np.linalg.norm(triVertices - point[:, :, :, np.newaxis, :], axis=-1)
    d += 1e-5
    d = 1 / d
    d = d / np.sum(d, axis=-1)[:, :, :, np.newaxis]

    uvStack = np.sum(triUVs * d[..., np.newaxis], axis=-2)
    # uvStack = triUVs[:, :, :, 0]

    return uvStack

# extract label from 2d uv to 3d stack
def calculateLabel(path, uvStack):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = image / np.max(image)
    wPixel = uvStack[:, :, :, 0]
    hPixel = 1 - uvStack[:, :, :, 1]
    hPixel = (image.shape[0] * hPixel).astype(int)
    wPixel = (image.shape[1] * wPixel).astype(int)
    labelStack = image[hPixel, wPixel][:,:,:]

    return labelStack
