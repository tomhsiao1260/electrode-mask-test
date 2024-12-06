# reorders `tris` such that for `count` elements after `offset`, elements on the left side of the split
# will be on the left and elements on the right side of the split will be on the right. returns the index
# of the first element on the right side, or offset + count if there are no elements on the right side.
def partition(data, triangleBounds, offset, count, split):
    box_centers, box_half_sizes = triangleBounds
    left = offset
    right = offset + count - 1
    axis, pos = split

    # hoare partitioning, see e.g. https://en.wikipedia.org/wiki/Quicksort#Hoare_partition_scheme
    while (True):
        while (left <= right and box_centers[left][axis] < pos):
            left += 1

        # if a triangle center lies on the partition plane it is considered to be on the right side
        while (left <= right and box_centers[right][axis] >= pos): right -= 1

        if (left < right):
            # we need to swap all of the information associated with the triangles at index
			# left and right; that's the verts in the geometry index, the bounds,
			# and perhaps the SAH planes

            # swap bounds
            data['faces'][[right, left]] = data['faces'][[left, right]]
            box_centers[[right, left]] = box_centers[[left, right]]
            box_half_sizes[[right, left]] = box_half_sizes[[left, right]]
                
            left += 1
            right -= 1
        else:
            return left