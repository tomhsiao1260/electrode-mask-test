def iterateOverTriangles(
	offset,
	count,
	intersectsTriangleFunc,
	tri
):
	for i in range(offset, offset + count):
		if (intersectsTriangleFunc(tri, i)): return True
	return False
