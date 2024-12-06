import numpy as np

_box1Min = np.zeros(3, dtype=np.float32)
_box1Max = np.zeros(3, dtype=np.float32)
_box2Min = np.zeros(3, dtype=np.float32)
_box2Max = np.zeros(3, dtype=np.float32)

def shapecast(
	self,
	intersectsBounds,
	intersectsRange,
	boundsTraverseOrder,
):
    # setup
    root = self._roots[0]
    _box1Min[:] = 0
    _box1Max[:] = 0
    _box2Min[:] = 0
    _box2Max[:] = 0
    
    result = shapecastTraverse(
		root,
		intersectsBounds,
		intersectsRange,
		boundsTraverseOrder,
	)

    return result

def shapecastTraverse(
    node,
    intersectsBoundsFunc,
    intersectsRangeFunc,
    nodeScoreFunc = None,
    depth = 0
):
    isLeaf = hasattr(node, 'count')
    if (isLeaf):
        offset = getattr(node, 'offset')
        count = getattr(node, 'count')
        return intersectsRangeFunc(offset, count)
    else:
        left = node.left
        right = node.right
        c1 = left
        c2 = right

        score1, score2 = None, None
        box1Min, box1Max, box2Min, box2Max = None, None, None, None

        if (nodeScoreFunc):
            box1Min = left.boundingData[:3]
            box1Max = left.boundingData[3:]
            box2Min = right.boundingData[:3]
            box2Max = right.boundingData[3:]

            score1 = nodeScoreFunc(box1Min, box1Max)
            score2 = nodeScoreFunc(box2Min, box2Max)

            if (score2 < score1):
                c1 = right
                c2 = left

                temp = score1
                score1 = score2
                score2 = temp

                box1Min = box2Min
                box1Max = box2Max
                # box2 is always set before use below

        # Check box 1 intersection
        if (box1Min is None):
            box1Min = c1.boundingData[:3]
            box1Max = c1.boundingData[3:]

        isC1Leaf = hasattr(c1, 'count')
        c1Intersection = intersectsBoundsFunc(
			box1Min,
            box1Max,
			isC1Leaf,
			score1,
		)
        c1StopTraversal = c1Intersection and shapecastTraverse(
			c1,
			intersectsBoundsFunc,
			intersectsRangeFunc,
			nodeScoreFunc,
			depth + 1
		)
        if (c1StopTraversal): return True

        # Check box 2 intersection
		# cached box2 will have been overwritten by previous traversal
        box2Min = c2.boundingData[:3]
        box2Max = c2.boundingData[3:]

        isC2Leaf = hasattr(c2, 'count')
        c2Intersection = intersectsBoundsFunc(
			box2Min,
            box2Max,
			isC2Leaf,
			score2,
		)
        c2StopTraversal = c2Intersection and shapecastTraverse(
			c2,
			intersectsBoundsFunc,
			intersectsRangeFunc,
			nodeScoreFunc,
			depth + 1
		)
        if (c2StopTraversal): return True
        return False