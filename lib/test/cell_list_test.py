import numpy as np

import cell_list

N = 5000
DIM = 2
L = 1600.0
R_cell = 10.0
r = np.random.uniform(-L/2.0, L/2.0, (N, DIM))
#c = cell_list.CellList(DIM, L, R_cell)

#def test_cl():
#    for _ in range(10):
#        inters = c.get_interacts(r)
#    return sum([sum(entry) for entry in inters])
#
#def test_dir():
#    for _ in range(10):
#        inters = c.get_interacts_direct(r, R_cell)
#    return sum([sum(entry) for entry in inters])    

def test_cl_f():
    for _ in range(100):
        inters = cell_list.get_interacts(r, L, R_cell)
    return sum([sum(entry) for entry in inters])

def test_dir_f():
    for _ in range(100):
        inters = cell_list.get_interacts_direct(r, L, R_cell)
    return sum([sum(entry) for entry in inters])    


def main():
#    inter = test_cl()
#    interd = test_dir()
    
    inter2 = test_cl_f()
#    interd2 = test_dir_f()
#    assert inter2 == interd2

if __name__ == '__main__':
    import cProfile as prof; prof.run('main()')    
    #main()