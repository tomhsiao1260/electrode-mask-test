import numpy as np

class Triangle:
    def __init__(self, tri):
        self.tri = tri

    def closestPointToPoint(self, p, i):
        a = self.tri[i][0]
        b = self.tri[i][1]
        c = self.tri[i][2]

        # algorithm thanks to Real-Time Collision Detection by Christer Ericson,
		# published by Morgan Kaufmann Publishers, (c) 2005 Elsevier Inc.,
		# under the accompanying license; see chapter 5.1.5 for detailed explanation.
		# basically, we're distinguishing which of the voronoi regions of the triangle
		# the point lies in with the minimum amount of redundant computation.

        _vab = b - a
        _vac = c - a
        _vap = p - a
        d1 = np.dot(_vab, _vap)
        d2 = np.dot(_vac, _vap)
        if (d1 <=0 and d2 <= 0):
            # vertex region of A; barycentric coords (1, 0, 0)
            return a
        
        _vbp = p - b
        d3 = np.dot(_vab, _vbp)
        d4 = np.dot(_vac, _vbp)
        if (d3 >= 0 and d4 <= d3):
            # vertex region of B; barycentric coords (0, 1, 0)
            return b
        
        vc = d1 * d4 - d3 * d2
        if (vc <= 0 and d1 >= 0 and d3 <= 0 ):
            v = d1 / ( d1 - d3 )
			# edge region of AB; barycentric coords (1-v, v, 0)
            return a + _vab * v
        
        _vcp = p - c
        d5 = np.dot(_vab, _vcp)
        d6 = np.dot(_vac, _vcp)
        if (d6 >= 0 and d5 <= d6):
			# vertex region of C; barycentric coords (0, 0, 1)
            return c
        
        vb = d5 * d2 - d1 * d6
        if ( vb <= 0 and d2 >= 0 and d6 <= 0 ):
            w = d2 / ( d2 - d6 )
            # edge region of AC; barycentric coords (1-w, 0, w)
            return a + _vac * w
        
        va = d3 * d6 - d5 * d4
        if ( va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0):
            _vbc = c - b
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            # edge region of BC; barycentric coords (0, 1-w, w)
            return b + _vbc * w # edge region of BC
        
        # face region
        denom = 1 / (va + vb + vc)
        # u = va * denom
        v = vb * denom
        w = vc * denom
        
        return a + _vab * v + _vac * w

    # https://www.shadertoy.com/view/ttfGWl
    def closestPointToPointGPU(self, p, i):
        v0 = self.tri[i][0]
        v1 = self.tri[i][1]
        v2 = self.tri[i][2]

        v10 = v1 - v0
        v21 = v2 - v1
        v02 = v0 - v2

        p0 = p - v0
        p1 = p - v1
        p2 = p - v2

        nor = np.cross(v10, v02)
        
        # method 2, in barycentric space
        q = np.cross(nor, p0)
        n = np.dot(nor, nor)
        d = 1 / (n + 1e-5)
        u = d * np.dot(q, v02)
        v = d * np.dot(q, v10)
        w = 1 - u - v

        # pick up the region
        spaceU = u < 0
        spaceV = np.logical_and(np.logical_not(spaceU), v < 0)
        spaceW = np.logical_and(np.logical_not(np.logical_or(spaceU, spaceV)), w < 0)

        # for space u
        n = np.dot(v02, v02)
        wp = np.minimum(np.maximum(np.dot(p2, v02) / (n + 1e-5), 0), 1)
        up = 0
        vp = 1 - wp

        u = np.where(spaceU, up, u)
        v = np.where(spaceU, vp, v)
        w = np.where(spaceU, wp, w)

        # for space v
        n = np.dot(v10, v10)
        up = np.minimum(np.maximum(np.dot(p0, v10) / (n + 1e-5), 0), 1)
        vp = 0
        wp = 1 - up

        u = np.where(spaceV, up, u)
        v = np.where(spaceV, vp, v)
        w = np.where(spaceV, wp, w)

        # for space w
        n = np.dot(v21, v21)
        vp = np.minimum(np.maximum(np.dot(p1, v21) / (n + 1e-5), 0), 1)
        wp = 0
        up = 1 - vp

        u = np.where(spaceW, up, u)
        v = np.where(spaceW, vp, v)
        w = np.where(spaceW, wp, w)

        u = u[..., np.newaxis]
        v = v[..., np.newaxis]
        w = w[..., np.newaxis]

        return u * v1 + v * v2 + w * v0
