import numpy as np
import unittest
import geom

class TestPointSeg(unittest.TestCase):

    def setUp(self):
        self.ar = np.array([-0.4, 0.05, -0.21])
        self.u = np.array([0.2, -0.014, -0.43])
        self.u_perp = np.cross(self.ar, self.u)

    def test_touching_start(self):
        br1 = self.ar
        br2 = self.ar + self.u
        self.assertAlmostEqual(geom.point_seg_sep_sq(self.ar, br1, br2), 0.0)

    def test_touching_inside(self):
        br1 = self.ar - self.u / 2.0
        br2 = self.ar + self.u
        self.assertAlmostEqual(geom.point_seg_sep_sq(self.ar, br1, br2), 0.0)

    def test_touching_end(self):
        br1 = self.ar - self.u
        br2 = self.ar
        self.assertAlmostEqual(geom.point_seg_sep_sq(self.ar, br1, br2), 0.0)

    def test_just_off_start(self):
        br1 = self.ar + self.u
        br2 = self.ar + 2.0 * self.u
        self.assertAlmostEqual(geom.point_seg_sep_sq(self.ar, br1, br2), np.sum(np.square(self.u)))

    def test_just_off_end(self):
        br1 = self.ar + 2.0 * self.u
        br2 = self.ar + self.u
        self.assertAlmostEqual(geom.point_seg_sep_sq(self.ar, br1, br2), np.sum(np.square(self.u)))

    def test_just_off_middle(self):
        br1 = -self.u
        br2 = self.u
        ar = self.u_perp
        self.assertAlmostEqual(geom.point_seg_sep_sq(ar, br1, br2), np.sum(np.square(self.u_perp)))

class TestSegs(unittest.TestCase):

    def setUp(self):
        self.ar1 = np.array([0.0, 0.0, 0.0])
        self.u = np.array([1.0, 0.0, 0.0])
        self.ar2 = self.ar1 + self.u
        self.test_func = geom.segs_sep_sq

    def test_parallel_touch(self):
        br1 = self.ar1.copy()
        br2 = self.ar2.copy()
        self.assertAlmostEqual(self.test_func(self.ar1, self.ar2, br1, br2), 0.0)

    def test_parallel_notouch(self):
        br1 = self.ar1 + 2.0 * self.u
        br2 = self.ar2 + 3.0 * self.u
        self.assertAlmostEqual(self.test_func(self.ar1, self.ar2, br1, br2), 1.0)

class TestSegsFast(TestSegs):
    def setUp(self):
        TestSegs.setUp(self)
        self.test_func = geom.segs_sep_sq_fast

if __name__ == '__main__':
    unittest.main()