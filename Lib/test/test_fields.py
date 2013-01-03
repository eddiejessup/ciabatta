import timeit
import numpy as np
import _fields as ff
import field_numerics as fp

l = 2.0
m = 1000
dx = l / m
l_half = l / 2.0
n = 200

setup_base = '''
import _fields as ff
import field_numerics as fp
import numpy as np
ap = np.random.uniform(size=2*(%i,))
af = np.asarray(ap, order='F')
dx = %f
''' % (m, dx)

def rms(a):
    return np.sqrt(np.mean(np.square(a)))

def test_grad_2d_speed():
    print('Grad 2D speeds:')
    setup = setup_base + 'gp = np.zeros([%i, %i, 2], dtype=np.float64)' % (m, m)
    print('Python: %f' % timeit.timeit('gp = fp.grad(ap, dx)', setup=setup, number=100))
    print('Fortran: %f' % timeit.timeit('gf = ff.fields.grad_2d(af, dx)', setup=setup, number=100))

def test_laplace_2d_speed():
    print('Laplace 2D speeds:')
    print('Python: %f' % timeit.timeit('lp = fp.laplace(ap, dx)', setup=setup_base, number=50))
    print('Fortran: %f' % timeit.timeit('lf = ff.fields.laplace_2d(af, dx)', setup=setup_base, number=50))

def test_density_2d_speed():
    print('Density 2D speeds:')
    setup = setup_base + 'rp = np.random.uniform(-%f, %f, size=(%i, 2))' % (l_half, l_half, n)
    setup += '\nrf = np.asarray(rp.T, dtype=np.float64, order="F")'
    setup += '\ndensp = np.zeros([%i, %i], dtype=np.int)' % (m,m)
    setup += '\ndensf = np.asarray(densp, dtype=np.int32, order="F")'
    print('Python: %f' % timeit.timeit('fp.density_2d(rp, %f, densp)' % l, setup=setup, number=200))
    print('Fortran: %f' % timeit.timeit('ff.fields.density_2d(rf, %f, densf)' % l, setup=setup, number=200))

def test_consistency():
    l = 2.0
    m = 100
    dx = l / m

    a=np.random.uniform(size=(m,m))

    gf = ff.fields.grad_2d(a, dx)
    gfp=np.transpose(gf,axes=(1,2,0))
    gp=np.zeros([m,m,2],dtype=np.float)
    fp.grad_2d(a, gp, dx)
    error = 100.0 * np.abs(gfp-gp).max() / np.mean([rms(gfp), rms(gp)])
    assert error < 1e-12
    print('Grad 2D works!')

    df = ff.fields.div_2d(gf, dx)
    dp = np.zeros([m,m])
    fp.div_2d(gp, dp, dx)
    error = 100.0 * np.abs(df-dp).max() / np.mean([rms(df), rms(dp)])
    assert error < 1e-12
    print('Div 2D works!')

    lf = ff.fields.laplace_2d(a, dx)
    lp = np.zeros([m,m], dtype=np.float64)
    fp.laplace_2d(a, lp, dx)
    error = 100.0 * np.abs(lf-lp).max() / np.mean([rms(lf), rms(lp)])
    assert error < 1e-12, 'Percent error is: %s' % error
    print('Laplace 2D works!')

    r = np.random.uniform(-1.0, 1.0, size=(200, 2))
    densf = np.zeros([m,m], dtype=np.int32, order='F')
    ff.fields.density_2d(r.T, 2.0, densf)
    densp = np.zeros([m,m], dtype=np.int)
    fp.density_2d(r, 2.0, densp)
    assert np.abs(densp - densf).max() == 0
    print('Density 2D works!')

    gif = ff.fields.grad_i_2d(a, r.T, l)
    gifp = gif.T
    gip = np.zeros(r.shape, dtype=np.float)
    inds = fp.r_to_i(r, l, dx)
    fp.grad_i_2d(a, inds, gip, dx)
    error = 100.0 * np.abs(gifp-gip).max() / np.mean([rms(gifp), rms(gip)])
    assert error < 1e-12
    print('Grad i 2D works!')

    print('2D Done!')

if __name__ == '__main__':
    test_consistency()
    test_grad_2d_speed()
#    test_laplace_2d_speed()
#    test_density_2d_speed()
