import numpy as np
from core.MeshBVHNode import MeshBVHNode
from core.build.splitUtils import getOptimalSplit
from core.build.sortUtils import partition
from core.build.computeBoundsUtils import getBounds, computeTriangleBounds

def buildTree(data, triangleBounds, offset, count):
    centroidTarget = np.zeros(6, dtype=np.float32)

    root = MeshBVHNode()
    setattr(root, "_offset", offset)
    setattr(root, "_count", count)

    getBounds(data, triangleBounds, offset, count, root.boundingData, centroidTarget)
    splitNode(data, triangleBounds, root, offset, count, centroidTarget)

    return root

# either recursively splits the given node, creating left and right subtrees for it, or makes it a leaf node,
# recording the offset and count of its triangles and writing them into the reordered geometry index.
def splitNode(data, triangleBounds, node, offset, count, centroidBoundingData, depth = 0):
    maxLeafTris = 5

    # early out if we've met our capacity
    if (count <= maxLeafTris):
        setattr(node, "offset", offset)
        setattr(node, "count", count)
        return node

    split = getOptimalSplit(centroidBoundingData)
    axis, pos = split

    splitOffset = partition(data, triangleBounds, offset, count, split)

    setattr(node, "splitAxis", axis)

    # create the left child and compute its bounding box
    left = MeshBVHNode()
    lstart = offset
    lcount = splitOffset - offset
    setattr(node, "left", left)
    setattr(node.left, "_offset", lstart)
    setattr(node.left, "_count", lcount)

    getBounds(data, triangleBounds, lstart, lcount, left.boundingData, centroidBoundingData)
    splitNode(data, triangleBounds, left, lstart, lcount, centroidBoundingData, depth + 1)

    # repeat for right
    right = MeshBVHNode()
    rstart = splitOffset
    rcount = count - lcount
    setattr(node, "right", right)
    setattr(node.right, "_offset", rstart)
    setattr(node.right, "_count", rcount)

    getBounds(data, triangleBounds, rstart, rcount, right.boundingData, centroidBoundingData)
    splitNode(data, triangleBounds, right, rstart, rcount, centroidBoundingData, depth + 1)

def buildPackedTree(data):
    triangleBounds = computeTriangleBounds(data)

    offset = 0
    count = triangleBounds[0].shape[0]

    root = buildTree(data, triangleBounds, offset, count)

    return root
