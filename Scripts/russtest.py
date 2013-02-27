import sys
import numpy as np
import matplotlib.mlab as mlb

d = mlb.csv2rec(sys.argv[1])

def ask(q, a):
    print('\t\t%s...' % q, end='')
    input()
    print('\t\t\t\t%s' % a)

def prompt(i):
    if sys.argv[3] == 'r':
        ask(d[i]['spelling'], d[i]['meaning'])
    elif sys.argv[3] == 'e':
        ask(d[i]['meaning'], d[i]['spelling'])
    else:
        raise Exception

def main():
    if sys.argv[2] == 'i':
        for i in range(len(d)):
            prompt(i)
    elif sys.argv[2] == 'r':
        while True:
            prompt(np.random.randint(len(d)))
    else:
        raise Exception

main()