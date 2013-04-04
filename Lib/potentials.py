import numpy as np
import utils

def LJ(r_0, U_0):
    '''
    Lennard-Jones with minimum at (r_0, -U_0).
    '''
    r_0_6 = r_0 ** 6
    def func(r_sq):
        six_term = r_0_6 / r_sq ** 3
        return U_0 * (six_term ** 2 - 2.0 * six_term)
    return func

def step(r_0, U_0):
    '''
    Potential Well at r with U(r < r_0) = 0, U(r > r_0) = U_0.
    '''
    def func(r_sq):
        return np.where(r_sq < r_0 ** 2, U_0, 0.0)
    return func

def inv_sq(k):
    '''
    Inverse-square law, U(r) = -k / r.
    '''
    def func(r_sq):
        return -k / np.sqrt(r_sq)
    return func

def harm_osc(k):
    '''
    Harmonic oscillator, U(r) = k * (r ** 2) / 2.0.
    '''
    def func(r_sq):
        return 0.5 * k * r_sq ** 2
    return func

def harm_osc_F(k):
    '''
    Harmonic oscillator, F(r) = -k * r.
    '''
    def func(r):
        return -k * r
    return func

def logistic(r_0, U_0, k):
    ''' Logistic approximation to step function. '''
    def func(r_sq):
        return 0.5 * U_0 * (1.0 + np.tanh(k * (np.sqrt(r_sq) - r_0)))
    return func

def logistic_F(r_0, U_0, k):
    def func(r):
        r_sq = utils.vector_mag(r)
        return -U_0 * utils.vector_unit_nonull(r) * (1.0 - np.square(np.tanh(k * (np.sqrt(r_sq) - r_0))))[:, np.newaxis]
    return func

def anis_wrap(func_iso):
    '''
    Wrap an isotropic potential in an anisotropic envelope
    '''
    def func_anis(r_sq, theta):
        return func_iso(r_sq) * (0.5 + np.cos(0.5*theta) ** 2)
    return func_anis
