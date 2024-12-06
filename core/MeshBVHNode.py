import numpy as np 

class MeshBVHNode:
    def __init__(self):
        self.boundingData = np.zeros(6, dtype=np.float32)