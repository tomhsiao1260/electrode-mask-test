def getOptimalSplit(centroidBoundingData):
    axis = -1
    pos = 0

    # center strategy
    axis = getLongestEdgeIndex(centroidBoundingData)

    if (axis != -1):
        pos = (centroidBoundingData[axis] + centroidBoundingData[axis + 3]) / 2

    return axis, pos

def getLongestEdgeIndex(bounds):
    splitDimIdx = -1
    splitDist = -float('inf')

    for i in range(3):
        dist = bounds[i + 3] - bounds[i]
        if (dist > splitDist):
            splitDist = dist
            splitDimIdx = i

    return splitDimIdx