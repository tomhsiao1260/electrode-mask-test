import cv2
import numpy as np 

class MeshBVHHelper:
    def __init__(self, bvh):
        self.bvh = bvh
        self.height, self.width = 500, 500
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def draw(self, center, windowSize, depth = 0):
        self.clear()
        root, canvas = self.bvh._roots[0], self.canvas
        needToDraw = False

        def drawTraverse(node, depth):
            nonlocal needToDraw
            height, width, canvas = self.height, self.width, self.canvas

            if (depth == 0):
                self.drawBox(node.boundingData, canvas, width, height, cx = center[0], cy = center[1], size = windowSize)
                needToDraw = True

            if (hasattr(node, 'left')): drawTraverse(node.left, depth - 1)
            if (hasattr(node, 'right')): drawTraverse(node.right, depth - 1)

        drawTraverse(root, depth)

        if (needToDraw):
            cv2.imshow('Box', canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True

        return False

    def drawBox(self, boundingData, canvas, width, height, cx, cy, size):
        xmin, ymin, zmin, xmax, ymax, zmax = boundingData[:6]

        w = abs(xmax - xmin)
        h = abs(ymax - ymin)

        box_width = int(w / size * width)
        box_height = int(h / size * height)
        box_x = int((xmin - (cx - size / 2)) / size * width)
        box_y = int((ymin - (cy - size / 2)) / size * height)

        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), thickness=1)

    def clear(self):
        self.canvas[:] = np.zeros_like(self.canvas)
