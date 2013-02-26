#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
Created on 31 Oct 2011

@author: elliot
'''

import numpy as np

ENTRIES_LEN = 5000

foreplay = """
********************************************************************************

TescoAdd

Input entries in pence. Fractional values are allowed if you must.

Things will go wrong if you have more than %i entries, so don't.

Special input:
    u - Undo last entry, can be done repeatedly if you make multiple mistakes.
    r - Reset sum and start from beginning. Note this cannot be undone.
    x - Exit.
    h - Show this introduction if you weren't paying attention the first time.
    e - Enter expression. Don't get clever though, with great power etc.
        Note expressions are evaluated using 2.x-style division.

********************************************************************************
"""

def main():
    print(foreplay % ENTRIES_LEN)
    entries = np.zeros([ENTRIES_LEN], dtype=np.float)
    i_entry = 0
    exit_flag = False
    while i_entry < len(entries):
        inputt = input('£%6.2f >> ' % (entries.sum() / 100))

        if inputt in ['u', 'undo']:
            if i_entry >= 1: 
                i_entry -= 1
            entries[i_entry] = 0.0

        elif inputt in ['r', 'reset']:
            entries[:] = 0.0
            i_entry = 0

        elif inputt in ['x', 'exit']:
            while True:
                confirm = input('\t\t\tIncluded promotions? Included delivery? (y/n) >> ')
                if confirm in ['y', 'yes']:
                    exit_flag = True
                    break
                elif confirm in ['n', 'no']:
                    break
                else:
                    print("\t\t\tCome now, play by the rules. Let's try again...")

        elif inputt in ['h', 'help']:
            print(foreplay % ENTRIES_LEN)

        else:
            if inputt in ['e', 'expression']:
                inputt = eval(input('\t\t\tEnter expression >> '))
                print('\t\t\tExpression evaluates to %f' % inputt)

            try:
                entry = float(inputt)
            except:
                print("\t\t\tI think you entered something silly. Let's pretend that didn't happen...")
                continue

            valid_entry_flag = True
            if entry % 1.0 != 0.0:
                while True:
                    confirm = input('\t\t\tInput seems to have fractional part. Sure about this? (y/n) >> ')
                    if confirm in ['y', 'yes']:
                        break
                    elif confirm in ['n', 'no']:
                        valid_entry_flag = False
                        break
                    else:
                        print("\t\t\tCome now, play by the rules. Let's try again...")
            if valid_entry_flag:
                entries[i_entry] = entry
                i_entry += 1

        if exit_flag:
            break

    return entries

if __name__ == '__main__':
    main()
